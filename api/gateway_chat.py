"""Default-off Hermes Gateway bridge for browser-originated chat turns."""
from __future__ import annotations

import http.client
import json
import logging
import os
import socket
import threading
from pathlib import Path
import time
import uuid
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from api.config import (
    AGENT_INSTANCES,
    CANCEL_FLAGS,
    STREAM_GOAL_RELATED,
    STREAMS,
    STREAMS_LOCK,
    STREAM_LAST_EVENT_ID,
    _get_session_agent_lock,
    _parse_provider_qualified_model_id,
    clear_session_writeback_owner_if_owned,
    resolve_session_reasoning_effort,
    peek_stream,
    register_active_run,
    release_stream_owned_registries,
    unregister_active_run,
    unregister_active_run_if_owned,
    unregister_stream_owner,
)
from api.gateway_projection import LiveTurnSnapshot
from api.gateway_transport import TurnResult
from api.helpers import _redact_text
from api.models import clear_process_wakeup_pause, get_session, merge_session_messages_append_only
from api.run_journal import RunJournalWriter
from api.turn_journal import append_turn_journal_event_for_stream

logger = logging.getLogger(__name__)

# Maps stream_id -> gateway run_id for approval response relay.
_STREAM_RUN_IDS: dict[str, str] = {}
_STREAM_RUN_LIFECYCLE: dict[str, dict[str, Any]] = {}
_STREAM_RUN_STARTING_CONDITION = threading.Condition()
GATEWAY_RUN_ID_WAIT_TIMEOUT = 5.0
# stream_id -> (base_url, api_key) of the Gateway the stream's run lives on, for Stop
# and approval replies; a reattached run's Gateway need not be the process profile's.
_STREAM_ENDPOINTS: dict[str, tuple[str, str]] = {}
# #7293: streams whose resumable Gateway run identity (the exact request before
# admission, then the run id) has been durably saved in the session. Only these
# survive a WebUI restart (``_resume_gateway_run_for_session`` reattaches them),
# so only these are left out of restart blocking and shutdown-drain accounting.
# A legacy /v1/chat/completions relay never records one, and a Runs API launch
# has none until its first successful save; both count as WebUI-owned work.
_DURABLE_RUN_STREAMS: set[str] = set()
_DURABLE_RUN_STREAMS_LOCK = threading.Lock()


def _mark_gateway_run_durable(stream_id: str) -> None:
    stream_id = str(stream_id or "").strip()
    if stream_id:
        with _DURABLE_RUN_STREAMS_LOCK:
            _DURABLE_RUN_STREAMS.add(stream_id)


class GatewayHandoffCancelled(RuntimeError):
    """Stop or a server settle cancelled the stream before its durable handoff (#7293)."""


def _publish_gateway_run_durable(stream_id: str) -> bool:
    """Mark ``stream_id`` durable iff it is still registered and not cancelled.

    Runs on the STREAMS_LOCK edge where Stop and the server settle publish a
    cancellation, so exactly one of the two wins (#7293).
    """
    with STREAMS_LOCK:
        flag = CANCEL_FLAGS.get(stream_id)
        if stream_id not in STREAMS or (flag is not None and flag.is_set()):
            return False
        _mark_gateway_run_durable(stream_id)
        return True


def gateway_run_durably_resumable(stream_id: str) -> bool:
    """True once ``stream_id``'s Gateway run identity is saved for reattachment.

    Stays true until the stream's Gateway state is released: a later save that
    clears the record also persists the turn's terminal state, so the run is
    either reattachable from disk or already settled there.
    """
    stream_id = str(stream_id or "").strip()
    if not stream_id:
        return False
    with _DURABLE_RUN_STREAMS_LOCK:
        return stream_id in _DURABLE_RUN_STREAMS


def gateway_run_endpoint(run_id: str) -> tuple[str, str]:
    """URL and key of the Gateway that owns run_id; the process Gateway if no live stream holds it."""
    run_id = str(run_id or "").strip()
    with _STREAM_RUN_STARTING_CONDITION:
        for stream_id, mapped in list(_STREAM_RUN_IDS.items()):
            if run_id and mapped == run_id and stream_id in _STREAM_ENDPOINTS:
                return _STREAM_ENDPOINTS[stream_id]
    from api.config import get_config

    return _gateway_base_url(get_config()), _gateway_api_key()


def gateway_stream_steer_target(stream_id: str) -> tuple[str, str, str] | None:
    """(run_id, base_url, api_key) of stream_id's live run, read together under one lock.

    None when the stream has no published run or its owning endpoint is gone:
    a steer must never fall back to another Gateway than the one running it.
    """
    stream_id = str(stream_id or "").strip()
    with _STREAM_RUN_STARTING_CONDITION:
        run_id = str(_STREAM_RUN_IDS.get(stream_id) or "").strip()
        if not run_id:
            state = _STREAM_RUN_LIFECYCLE.get(stream_id) or {}
            if str(state.get("phase") or "").strip().lower() == "ready":
                run_id = str(state.get("run_id") or "").strip()
        endpoint = _STREAM_ENDPOINTS.get(stream_id)
        if not run_id or not endpoint:
            return None
        return run_id, endpoint[0], endpoint[1]


def _mark_gateway_run_starting(stream_id: str) -> None:
    with _STREAM_RUN_STARTING_CONDITION:
        _STREAM_RUN_IDS.pop(stream_id, None)
        _STREAM_RUN_LIFECYCLE[stream_id] = {
            "phase": "pending",
            "run_id": "",
            "waiters": 0,
            "owner_done": False,
        }


def _publish_gateway_run_id(stream_id: str, run_id: str) -> None:
    with _STREAM_RUN_STARTING_CONDITION:
        _STREAM_RUN_IDS[stream_id] = run_id
        state = _STREAM_RUN_LIFECYCLE.get(stream_id) or {}
        _STREAM_RUN_LIFECYCLE[stream_id] = {
            "phase": "ready",
            "run_id": run_id,
            "waiters": int(state.get("waiters") or 0),
            "owner_done": bool(state.get("owner_done")),
        }
        _STREAM_RUN_STARTING_CONDITION.notify_all()


def _finish_gateway_run_starting(stream_id: str, *, result: str = "failed") -> None:
    with _STREAM_RUN_STARTING_CONDITION:
        state = _STREAM_RUN_LIFECYCLE.get(stream_id) or {}
        if str(state.get("phase") or "").strip().lower() == "ready":
            return
        _STREAM_RUN_IDS.pop(stream_id, None)
        _STREAM_RUN_LIFECYCLE[stream_id] = {
            "phase": "fallback" if result == "fallback" else "failed",
            "run_id": "",
            "waiters": int(state.get("waiters") or 0),
            "owner_done": bool(state.get("owner_done")),
        }
        _STREAM_RUN_STARTING_CONDITION.notify_all()


def _retire_gateway_run_starting_if_done(stream_id: str) -> bool:
    state = _STREAM_RUN_LIFECYCLE.get(stream_id)
    if not state:
        return False
    if int(state.get("waiters") or 0) > 0:
        return False
    if not bool(state.get("owner_done")):
        return False
    _STREAM_RUN_LIFECYCLE.pop(stream_id, None)
    _STREAM_RUN_IDS.pop(stream_id, None)
    return True


def _clear_gateway_run_starting(stream_id: str) -> None:
    with _STREAM_RUN_STARTING_CONDITION:
        state = _STREAM_RUN_LIFECYCLE.get(stream_id)
        if state:
            state["owner_done"] = True
        _retire_gateway_run_starting_if_done(stream_id)
        _STREAM_RUN_STARTING_CONDITION.notify_all()


def release_gateway_stream_state(stream_id: str, *, finish_pending: bool = True) -> None:
    """No-op-safe release of the Gateway-owned rows for ``stream_id``.

    THE release path for ``_STREAM_RUN_LIFECYCLE`` / ``_STREAM_RUN_IDS`` /
    ``_STREAM_ENDPOINTS``: the canonical Gateway worker teardown and the
    chat/start orphan recovery both call this, so a dead stream cannot leak a
    lifecycle row, a run-id mapping or an endpoint that no worker will ever free
    (#7302 re-gate).

    Uses the existing lifecycle/waiter protocol instead of popping rows raw: a
    request parked in ``wait_for_gateway_run_id`` is never stranded -- it wakes on
    the terminal phase, and retires the row itself on the way out
    (``_retire_gateway_run_starting_if_done``), which is why the row can be left
    in place while ``waiters`` is non-zero.

    ``finish_pending`` publishes the terminal phase for a run that never reached
    ``ready``. It is a pure no-op for a stream that never touched the Gateway
    (no lifecycle row, no run id, no endpoint).
    """
    stream_id = str(stream_id or "").strip()
    if not stream_id:
        return
    if finish_pending and gateway_run_id_pending(stream_id):
        _finish_gateway_run_starting(stream_id)
    _clear_gateway_run_starting(stream_id)
    with _STREAM_RUN_STARTING_CONDITION:
        _STREAM_ENDPOINTS.pop(stream_id, None)
    with _DURABLE_RUN_STREAMS_LOCK:
        _DURABLE_RUN_STREAMS.discard(stream_id)


def gateway_run_id_pending(stream_id: str) -> bool:
    with _STREAM_RUN_STARTING_CONDITION:
        return str((_STREAM_RUN_LIFECYCLE.get(stream_id) or {}).get("phase") or "").strip().lower() == "pending"


def wait_for_gateway_run_id(stream_id: str, timeout: float) -> tuple[bool, str | None]:
    deadline = time.monotonic() + max(0.0, float(timeout))
    with _STREAM_RUN_STARTING_CONDITION:
        state = _STREAM_RUN_LIFECYCLE.get(stream_id)
        if state:
            state["waiters"] = int(state.get("waiters") or 0) + 1
        try:
            while True:
                state = _STREAM_RUN_LIFECYCLE.get(stream_id)
                phase = str((state or {}).get("phase") or "").strip().lower()
                if phase == "fallback":
                    return False, None
                if phase == "failed":
                    return True, None
                run_id = str(_STREAM_RUN_IDS.get(stream_id) or "").strip()
                if phase == "ready":
                    stored_run_id = str((state or {}).get("run_id") or "").strip()
                    return True, run_id or stored_run_id or None
                if run_id:
                    return True, run_id
                if not state:
                    return False, None
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return True, None
                _STREAM_RUN_STARTING_CONDITION.wait(timeout=remaining)
        finally:
            state = _STREAM_RUN_LIFECYCLE.get(stream_id)
            if state:
                waiters = max(0, int(state.get("waiters") or 0) - 1)
                state["waiters"] = waiters
                if _retire_gateway_run_starting_if_done(stream_id):
                    _STREAM_RUN_STARTING_CONDITION.notify_all()

_WEBUI_CHAT_BACKEND_ENV = "HERMES_WEBUI_CHAT_BACKEND"
_WEBUI_GATEWAY_BASE_URL_ENV = "HERMES_WEBUI_GATEWAY_BASE_URL"
_WEBUI_GATEWAY_API_KEY_ENV = "HERMES_WEBUI_GATEWAY_API_KEY"
_WEBUI_GATEWAY_USE_RUNS_API_ENV = "HERMES_WEBUI_GATEWAY_USE_RUNS_API"
_GATEWAY_CHAT_BACKENDS = {"gateway", "api_server", "api-server"}
# Backend tag of the in-process WebUI runtime. Local workers register their
# active run with it; cache-only Steer only enqueues on this explicit value.
WEBUI_LOCAL_CHAT_BACKEND = "legacy"


def _gateway_model_field(model: str | None) -> str:
    """Return the bare model name to put in a gateway request body.

    The picker and ``_resolve_compatible_session_model_state`` intentionally
    keep the full ``@provider:model`` string for internal routing (#1253), but
    the gateway expects a bare model name and carries the provider separately.
    Sent verbatim, ``@ollama-cloud:minimax-m2.7`` is forwarded to the upstream
    provider API, which 404s on the ``@``-prefixed string (#6722).

    Parsing is delegated to ``config._parse_provider_qualified_model_id()`` so
    a multi-segment custom provider ID (``@custom:backup:model-a``) yields the
    real model (``model-a``) instead of a positional-split fragment.
    """
    if not model:
        return ""
    value = str(model).strip()
    parsed = _parse_provider_qualified_model_id(value)
    if parsed:
        return str(parsed[0] or "").strip()
    return value


# Total byte-silence budget (seconds) for the gateway SSE socket, applied via
# ``urlopen(timeout=...)``. A stream that emits ANY byte within the window never
# trips it, so a genuinely alive (if slow) token stream is untouched; only more
# than this much *total* byte-silence is treated as a dead/stalled gateway.
#
# This is a TERMINAL budget, not a per-read grace: CPython's ``socket.makefile``
# latches ``_timeout_occurred`` on the first ``socket.timeout``, after which every
# further read raises a bare ``OSError`` — the connection cannot be resumed. So a
# read timeout ends the turn (surfacing Stop if pressed). It replaces the old flat
# 600s timeout, under which a half-open gateway (TCP open, zero bytes) pinned the
# worker for the full 10 minutes and ignored Stop (cancel is only re-checked
# between SSE lines). We KEEP the 600s default budget: there is no Gateway
# protocol heartbeat guaranteeing sub-600s progress bytes, so a legitimately
# long/fully-silent server-side tool call must not be terminated early — reducing
# the default below 600s would kill currently-working turns (gate finding, #5789).
# The win here is that a read timeout is now TERMINAL and Stop-honoring (the old
# flat timeout ignored Stop on a half-open gateway); the budget itself stays 600s
# for backward compatibility. Deployments that want a tighter dead-gateway cap can
# lower ``HERMES_WEBUI_GATEWAY_READ_TIMEOUT``.
def webui_chat_backend_mode(config_data=None, environ: dict[str, str] | None = None) -> str:
    """Return the explicitly selected browser chat backend.

    The default remains the in-process WebUI runtime. Only explicit gateway
    values opt browser chat into the Hermes API server bridge; generic truthy
    strings are deliberately ignored so deployments do not change execution
    ownership by accident.
    """
    source = os.environ if environ is None else environ
    cfg = config_data if isinstance(config_data, dict) else {}
    raw = str(
        source.get(_WEBUI_CHAT_BACKEND_ENV)
        or cfg.get("webui_chat_backend")
        or ""
    ).strip().lower()
    if raw in _GATEWAY_CHAT_BACKENDS:
        return "gateway"
    return WEBUI_LOCAL_CHAT_BACKEND


def webui_gateway_chat_enabled(config_data=None, environ: dict[str, str] | None = None) -> bool:
    return webui_chat_backend_mode(config_data, environ) == "gateway"


class GatewayExternalSessionReadOnly(PermissionError):
    pass


def gateway_session_chat_read_only_reason(session: dict, config_data=None) -> str:
    """Transport eligibility only; never an Agent authorization grant or saved flag.

    Importing a WebUI sidecar does not change the Agent's source/API binding.
    Keep in-process continuation and newly created WebUI/fork sessions unchanged.
    Resolve the owning profile even on workers without request context.
    """
    if session.get("session_source") == "fork":
        return ""
    source = str(session.get("raw_source") or session.get("source_tag")
                 or session.get("session_source") or "").strip().lower()
    if source in {"webui", "api_server", "api", "bot_room", "fork"} or (
        not source and not session.get("is_cli_session")
    ):
        return ""
    from api.gateway_sessions import configured, config_for_session, external_session
    if config_data is None:
        config_data = config_for_session(session)
    if not webui_gateway_chat_enabled(config_data):
        return ""
    if configured(session, config_data):
        return ""
    if not external_session(session):
        return "Read-only in WebUI."
    client = {
        "desktop": "Desktop", "cli": "CLI", "tui": "TUI", "gui": "GUI",
        "acp": "ACP", "local": "the original client", "telegram": "Telegram",
        "discord": "Discord", "slack": "Slack", "whatsapp": "WhatsApp",
        "signal": "Signal", "matrix": "Matrix", "mattermost": "Mattermost",
        "teams": "Teams", "line": "LINE", "feishu": "Feishu",
    }.get(source, "the original client")
    return f"Read-only in WebUI. Continue in {client}."


def _gateway_base_url(config_data=None, environ: dict[str, str] | None = None) -> str:
    source = os.environ if environ is None else environ
    cfg = config_data if isinstance(config_data, dict) else {}
    raw = str(
        source.get(_WEBUI_GATEWAY_BASE_URL_ENV)
        or cfg.get("webui_gateway_base_url")
        or "http://127.0.0.1:8642"
    ).strip()
    return raw.rstrip("/") or "http://127.0.0.1:8642"


def _gateway_api_key(environ: dict[str, str] | None = None) -> str:
    source = os.environ if environ is None else environ
    return str(
        source.get(_WEBUI_GATEWAY_API_KEY_ENV)
        or source.get("API_SERVER_KEY")
        or ""
    ).strip()


def _gateway_use_runs_api_enabled(config_data=None, environ: dict[str, str] | None = None) -> bool:
    """Return True only when the operator has explicitly opted into the runs API path."""
    source = os.environ if environ is None else environ
    cfg = config_data if isinstance(config_data, dict) else {}
    raw = str(
        source.get(_WEBUI_GATEWAY_USE_RUNS_API_ENV)
        or cfg.get("webui_gateway_use_runs_api")
        or ""
    ).strip().lower()
    return raw in ("1", "true", "yes", "on")


def _gateway_reasoning_effort_for_request(
    cfg, *, model=None, model_provider=None, session_effort=None
):
    """Resolve and coerce the session effort, falling back to profile config."""
    try:
        coerced = resolve_session_reasoning_effort(
            cfg,
            session_effort=session_effort,
            model_id=model,
            provider_id=model_provider,
        )
        # Preserve explicit "none" while still omitting absent or invalid effort.
        return None if not coerced else str(coerced)
    except Exception:
        return None


def _gateway_reasoning_body_fields(reasoning_effort) -> dict:
    """Request-body fields that carry a resolved effort to the Gateway.

    The Gateway decodes effort only from ``model_options`` (structured
    ``reasoning``, or the legacy ``reasoning_effort`` key inside it) and ignores
    a top-level ``reasoning_effort``; that key is still sent for any Gateway that
    reads it. ``None`` or ``""`` (the session's "Default") sends nothing, so the
    Gateway applies its own profile config: its protocol has no "provider
    default, ignore config" value, and ``{"enabled": true}`` alone means a
    medium effort on several providers.
    """
    effort = str(reasoning_effort or "").strip().lower()
    if not effort:
        return {}
    reasoning = {"enabled": False} if effort == "none" else {"enabled": True, "effort": effort}
    return {"reasoning_effort": effort, "model_options": {"reasoning": reasoning}}


def _gateway_session_yolo_enabled(session_id: str) -> bool:
    """Return the WebUI-owned, in-memory YOLO state for a browser session."""
    try:
        from tools.approval import is_session_yolo_enabled

        return bool(is_session_yolo_enabled(str(session_id or "")))
    except Exception:
        return False


def _settle_gateway_run_approval(
    session_id: str,
    approval_data: dict,
    base_url: str,
    api_key: str,
    *,
    cancel_event=None,
) -> tuple[bool, dict | None, int]:
    """Auto-approve or mirror one run approval at a session-linearized point.

    ``cancel_event`` is the run's retained Stop signal, for relays that can still
    receive an approval after Stop (the native session transport keeps reading
    its socket until the Gateway confirms the interrupt). A cancelled run gets
    no auto-approval and no card: the flag is checked before either, and again
    under the approval registry lock by ``submit_gateway_pending_mirror``, the
    same lock Stop retires the run's cards under (#8634). A card published in
    the instant before Stop is retired here as a second line of defence. The
    return is then ``(False, None, 0)``; the caller must not surface a card.
    """
    from api.route_approvals import (
        gateway_yolo_handoff,
        settle_gateway_pending_run,
        submit_gateway_pending_mirror,
    )

    run_id = str(approval_data.get("run_id") or "").strip()
    identity_v1 = bool(approval_data.get("_gateway_agent_identity_v1"))
    execution_generation = approval_data.get("_gateway_execution_generation")
    with gateway_yolo_handoff(session_id):
        if cancel_event is not None and cancel_event.is_set():
            return False, None, 0
        if _gateway_session_yolo_enabled(session_id):
            try:
                _auto_approve_gateway_run(
                    base_url,
                    api_key,
                    run_id,
                    approval_data["approval_id"] if identity_v1 else "",
                    **(
                        {"execution_generation": execution_generation}
                        if identity_v1 and execution_generation is not None
                        else {}
                    ),
                )
                return True, None, 0
            except Exception:
                # Fail closed: if remote approval fails, surface the real card
                # before allowing a same-session toggle to pass the handoff.
                logger.warning(
                    "WebUI YOLO could not auto-approve run %s; showing approval card",
                    run_id,
                    exc_info=True,
                )
        # Only the cancel-aware relays pass the flag; the existing Runs API caller
        # keeps its exact call shape.
        head, total = submit_gateway_pending_mirror(
            session_id,
            approval_data,
            **({"cancel_event": cancel_event} if cancel_event is not None else {}),
        )
    if cancel_event is not None and cancel_event.is_set():
        if run_id:
            settle_gateway_pending_run(
                session_id, run_id, reason="Gateway run was cancelled before approval resolution",
            )
        return False, None, 0
    return False, head, total


def _auto_approve_gateway_run(
    base_url: str,
    api_key: str,
    run_id: str,
    approval_id: str,
    *,
    execution_generation: int | None = None,
) -> None:
    """Resolve one Runs API prompt using only the shipped approval contract.

    This is a WebUI-owned compatibility path: the Runs API does not yet expose
    session YOLO, so WebUI answers each approval request while its own session
    flag is enabled. Native Agent-side YOLO would be preferable because it can
    bypass gates before they pause and also covers Agent-owned computer-use
    policy; https://github.com/NousResearch/hermes-agent/pull/61946 tracks that
    API capability. Until then, send only the fields the gateway itself
    published for this prompt (its id and, when present, its generation).
    """
    from api.runner_client import HttpRunnerClient

    from api.gateway_sessions import PREFIX, control
    if run_id.startswith(PREFIX):
        control(run_id, "approval", approval_id=approval_id, choice="once")
        return
    HttpRunnerClient(base_url=base_url, api_key=api_key).respond_approval(
        run_id,
        approval_id,
        "once",
        **({"execution_generation": execution_generation} if execution_generation is not None else {}),
    )


def gateway_chat_config_status(config_data=None, environ: dict[str, str] | None = None) -> dict:
    """Return redacted Gateway-backed chat configuration status."""
    mode = webui_chat_backend_mode(config_data, environ)
    base_url = _gateway_base_url(config_data, environ)
    return {
        "enabled": mode == "gateway",
        "backend": mode,
        "base_url_configured": bool(base_url),
        "api_key_configured": bool(_gateway_api_key(environ)),
    }


def _gateway_http_error_event(exc: urllib.error.HTTPError, err_body: str | None = None, *, api_key_configured: bool) -> dict:
    if err_body is None:
        try:
            err_body = exc.read(2048).decode("utf-8", errors="replace")
        except Exception:
            err_body = ""
    err_body = (err_body or "")[:2048]
    safe = _redact_text(err_body or str(exc), _enabled=True)[:500]
    if exc.code == 401:
        return {
            "label": "Gateway authentication failed",
            "type": "gateway_auth_error",
            "message": "Gateway rejected the WebUI API key (HTTP 401).",
            "hint": (
                "Set HERMES_WEBUI_GATEWAY_API_KEY to the same value as the Hermes Gateway "
                "API_SERVER_KEY, or disable HERMES_WEBUI_CHAT_BACKEND=gateway."
                if not api_key_configured
                else "Check that HERMES_WEBUI_GATEWAY_API_KEY matches the Hermes Gateway API_SERVER_KEY."
            ),
        }
    event = {
        "label": "Gateway request failed",
        "type": "gateway_http_error",
        "message": f"Gateway returned HTTP {exc.code}.",
        "hint": safe or "Check the configured Gateway API server.",
    }
    try:
        error = json.loads(err_body or "").get("error")
    except (ValueError, AttributeError):
        error = None
    if isinstance(error, dict):
        code = error.get("code")
        message = error.get("message")
        if isinstance(code, str) and code:
            event["code"] = _redact_text(code[:128], _enabled=True)
        if isinstance(message, str) and message:
            detail = _redact_text(message[:2048], _enabled=True)[:500]
            event["message"] = f"Gateway returned HTTP {exc.code}: {detail}"
        if event.get("code") and event["code"] not in event["message"]:
            event["message"] += f" ({event['code']})"
        if code == "permission_denied":
            event["hint"] = (
                "The Gateway did not authorize this turn. Reading or importing a conversation "
                "does not grant permission to continue it through the Gateway API."
            )
    return event


def _gateway_prompt_identity(payload) -> str:
    """The exact per-prompt id the Gateway published.

    In-process approvals carry ``request_id`` (``status["approval"]`` and the
    ``approval.request`` event); session-authority prompts carry ``prompt_id``
    (``pending_controls`` and their control events).
    """
    return str(payload.get("request_id") or payload.get("prompt_id") or "").strip()


def _gateway_runs_approval_event(payload: dict) -> dict | None:
    """Map a runs-API approval.request payload to the WebUI approval contract."""
    if not isinstance(payload, dict):
        return None
    tool = str(payload.get("tool") or payload.get("function_name") or payload.get("pattern_key") or "").strip()
    command = str(payload.get("command") or "").strip()
    description = str(payload.get("description") or "").strip()
    pattern_keys = payload.get("pattern_keys") if isinstance(payload.get("pattern_keys"), list) else []
    pattern_key = str(payload.get("pattern_key") or "").strip()
    args = payload.get("args") if isinstance(payload.get("args"), (list, dict)) else []
    run_id = str(payload.get("run_id") or "").strip()
    # Every Gateway prompt carries its id; a session-authority prompt is also
    # fenced by an integer ``execution_generation``. A prompt without an id
    # cannot be answered exactly, so it is not surfaced (fail closed).
    approval_id = _gateway_prompt_identity(payload)
    if not approval_id:
        return None
    execution_generation = payload.get("execution_generation")
    if type(execution_generation) is not int:
        execution_generation = None
    risk = str(payload.get("risk_level") or "high").strip()
    choices = payload.get("choices") if isinstance(payload.get("choices"), list) else []
    allow_permanent = payload.get("allow_permanent")
    if allow_permanent is None:
        allow_permanent = "always" in choices
    if not (tool or command or description):
        return None
    return {
        "tool": tool,
        "command": command,
        "description": description,
        "pattern_key": pattern_key,
        "pattern_keys": pattern_keys or ([pattern_key] if pattern_key else []),
        "args": args,
        "risk_level": risk,
        "run_id": run_id,
        "approval_id": approval_id,
        "_gateway_execution_generation": execution_generation,
        "choices": choices,
        "allow_permanent": bool(allow_permanent),
    }


def _gateway_approval_key(payload) -> str:
    """Stable id for one gateway approval, shared by the status probe and the event relay.

    The execution generation is part of the identity: a run that re-publishes the
    same prompt id at a newer generation is a different prompt, and answering the
    old generation is refused by the gateway. Keying on the id alone deduped that
    replay away and left the mirror pointing at a dead generation.
    """
    identity = _gateway_prompt_identity(payload)
    generation = payload.get("execution_generation")
    if identity and type(generation) is int:
        return f"{identity}#{generation}"
    return identity


def _gateway_status_approval(status: dict) -> dict | None:
    """The pending approval a durable run status describes, in either projection shape.

    The session bridge answers ``status["approval"]`` with the latest single
    prompt; the canonical authority projection answers ``pending_controls``,
    where the run's own queue order puts the prompt that must be answered first
    at the head.
    """
    approval = status.get("approval")
    if isinstance(approval, dict):
        return approval
    controls = status.get("pending_controls")
    if not isinstance(controls, list):
        return None
    for control in controls:
        if isinstance(control, dict) and control.get("kind") == "approval":
            return control
    return None


def _gateway_approval_identity_v1(approval_data: dict) -> bool:
    """Decide once, at card-creation time, whether this card is answered by exact identity.

    Shared by every card origin (Runs API relay, ``/v1/chat/completions``
    stream, durable status), so ``/api/approval/respond`` never re-derives it.
    A session-authority prompt is fenced by its ``execution_generation`` and is
    answered with ``{request_id, execution_generation, choice}``; an in-process
    approval has no generation and is answered by the in-process run handler.
    """
    return (approval_data or {}).get("_gateway_execution_generation") is not None


def stop_gateway_run(run_id: str) -> bool:
    """Request gateway interruption and report whether it was acknowledged."""
    run_id = str(run_id or "").strip()
    if not run_id:
        return False
    from api.gateway_sessions import PREFIX, control
    if run_id.startswith(PREFIX):
        try:
            return bool(control(run_id, "stop"))
        except ValueError:
            return False
    base_url, api_key = gateway_run_endpoint(run_id)
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/runs/{urllib.parse.quote(run_id, safe='')}/stop",
        data=b"{}",
        headers=headers,
        method="POST",
    )
    class _NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None

    try:
        opener = urllib.request.build_opener(_NoRedirect)
        with opener.open(req, timeout=10) as response:
            final_url = str(getattr(response, "geturl", lambda: req.full_url)() or "")
            status = int(getattr(response, "status", getattr(response, "code", 0)) or 0)
            return 200 <= status < 300 and final_url == req.full_url
    except (urllib.error.HTTPError, urllib.error.URLError, OSError, ValueError):
        logger.debug("Gateway stop failed for run %s", run_id, exc_info=True)
        return False


GATEWAY_STEER_ACCEPTED = "accepted"
GATEWAY_STEER_REFUSED = "refused"
GATEWAY_STEER_UNCERTAIN = "uncertain"
GATEWAY_STEER_TIMEOUT_SECS = 10
# Failures that prove no connection existed, so no request byte reached the Gateway.
_GATEWAY_NOT_CONNECTED_ERRORS = (ConnectionRefusedError, socket.gaierror)


def steer_gateway_run(run_id: str, text: str, base_url: str, api_key: str) -> str:
    """Forward steer text to the Gateway run.

    ``accepted``: 2xx with ``accepted: true``. ``refused``: the Gateway answered
    with a 2xx without acceptance or a 3xx/4xx, or no connection was ever made (refused, DNS, unreachable), so a
    re-queue cannot double-deliver. ``uncertain``: anything else — urllib wraps
    send-phase errors (reset/timeout while writing) in the same URLError, and the
    Gateway may already hold the text.
    """
    run_id = str(run_id or "").strip()
    text = str(text or "").strip()
    from api.gateway_sessions import PREFIX, control
    if str(run_id).startswith(PREFIX):
        try:
            result = control(run_id, "steer", text=text)
            return GATEWAY_STEER_ACCEPTED if result.get("status") in {"steered", "queued", "accepted"} else GATEWAY_STEER_REFUSED
        except ValueError:
            return GATEWAY_STEER_UNCERTAIN
    if not run_id or not text or not base_url:
        return GATEWAY_STEER_REFUSED
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/runs/{urllib.parse.quote(run_id, safe='')}/steer",
        data=json.dumps({"input": text}).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    class _NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None

    try:
        opener = urllib.request.build_opener(_NoRedirect)
        with opener.open(req, timeout=GATEWAY_STEER_TIMEOUT_SECS) as response:
            final_url = str(getattr(response, "geturl", lambda: req.full_url)() or "")
            status = int(getattr(response, "status", getattr(response, "code", 0)) or 0)
            if not (200 <= status < 300 and final_url == req.full_url):
                return GATEWAY_STEER_REFUSED
            payload = json.loads(response.read() or b"{}")
            accepted = isinstance(payload, dict) and payload.get("accepted") is True
            return GATEWAY_STEER_ACCEPTED if accepted else GATEWAY_STEER_REFUSED
    except urllib.error.HTTPError as exc:
        # A 5xx (502/504 from a proxy, a crashed handler) does not say the steer
        # was rejected: the Gateway may have taken it before the reply was lost.
        # Only an explicit 3xx/4xx answer proves it was not accepted.
        if int(getattr(exc, "code", 0) or 0) >= 500:
            logger.debug("Gateway steer outcome unknown for run %s", run_id, exc_info=True)
            return GATEWAY_STEER_UNCERTAIN
        logger.debug("Gateway steer refused for run %s", run_id, exc_info=True)
        return GATEWAY_STEER_REFUSED
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, _GATEWAY_NOT_CONNECTED_ERRORS):
            logger.debug("Gateway steer not sent for run %s", run_id, exc_info=True)
            return GATEWAY_STEER_REFUSED
        logger.debug("Gateway steer outcome unknown for run %s", run_id, exc_info=True)
        return GATEWAY_STEER_UNCERTAIN
    except (OSError, ValueError, http.client.HTTPException):
        logger.debug("Gateway steer outcome unknown for run %s", run_id, exc_info=True)
        return GATEWAY_STEER_UNCERTAIN


_REATTACH_SCAN_HEAD_BYTES = 16 * 1024


def _record_gateway_run(session_id: str, stream_id: str, run_id: str, request=None, **extra) -> None:
    """Persist the run id (or, before admission, the exact request) so a restarted WebUI can reattach.

    The request record is the run's durable handoff (#7293): when the stream was
    cancelled or detached first, it is withdrawn and ``GatewayHandoffCancelled``
    stops the admission.
    """
    try:
        with _get_session_agent_lock(session_id):
            session = get_session(session_id)
            if not _stream_writeback_is_current(session, stream_id):
                if not run_id:
                    raise GatewayHandoffCancelled(stream_id)
                return
            previous = session.gateway_run
            session.gateway_run = {"run_id": run_id, "stream_id": stream_id, **extra}
            if not run_id:
                session.gateway_run["request"] = request
            session.save(touch_updated_at=False)
            # Only now can a restart reattach this run (#7293 drain accounting).
            if not _publish_gateway_run_durable(stream_id) and not run_id:
                # Cancelled or detached while saving: withdraw the replayable request.
                session.gateway_run = previous
                try:
                    session.save(touch_updated_at=False)
                finally:  # even a failed withdrawal never admits the run
                    raise GatewayHandoffCancelled(stream_id)
    except GatewayHandoffCancelled:
        raise
    except Exception:
        logger.warning("Failed to persist gateway run %s for session %s", run_id, session_id, exc_info=True)
        if not run_id:
            raise  # never admit a run a restart could not recover


def resume_gateway_runs_after_restart() -> list[str]:
    """Reattach pending turns whose gateway run outlived the previous WebUI process; never raises.

    Call before serving so stale-pending repair does not mark these turns interrupted.
    """
    from api import models as _models

    try:
        candidates = _sidecars_with_active_stream(_models.SESSION_DIR)
    except Exception:
        logger.warning("gateway reattach: could not scan session sidecars", exc_info=True)
        return []
    resumed: list[str] = []
    for sid in candidates:
        try:
            if _resume_gateway_run_for_session(_models.Session.load_metadata_only(sid)):
                resumed.append(sid)
        except Exception:
            logger.warning("gateway reattach failed for session %s", sid, exc_info=True)
    if resumed:
        logger.info("Reattached %d gateway run(s) that outlived the previous WebUI process", len(resumed))
    return resumed


def _sidecars_with_active_stream(session_dir) -> list[str]:
    """Ids of sidecars (not the index, which can lag a save) whose header shows an active stream."""
    ids = []
    for path in sorted(session_dir.glob("*.json")):
        if path.name.startswith("_"):
            continue
        try:
            with path.open("rb") as fp:
                head = fp.read(_REATTACH_SCAN_HEAD_BYTES)
        except OSError:
            continue
        if b'"active_stream_id": null' not in head:
            ids.append(path.stem)
    return ids


def _profile_config_path(home):
    """config.yaml of a profile home. For the ACTIVE profile that is the authoritative ambient config file
    (HERMES_CONFIG_PATH may point outside the profile home), the same rule get_config_for_profile_home() uses;
    any other profile reads its own home, so a divergent profile never inherits the active custom config."""
    from api import profiles as _profiles
    from api.config import _get_config_path
    from api.workspace import _safe_resolve

    target = _safe_resolve(Path(home).expanduser())
    try:
        if _safe_resolve(Path(_profiles.get_active_hermes_home()).expanduser()) == target:
            return _get_config_path()
    except Exception:
        pass
    try:
        ambient = _get_config_path()
        if _safe_resolve(ambient.parent) == target:
            return ambient
    except Exception:
        pass
    return Path(home) / "config.yaml"


def _process_profile_is_root() -> bool:
    """True when the .env the process mirrored into os.environ is the default profile's (~/.hermes/.env)."""
    from api import profiles as _profiles
    from api.workspace import _safe_resolve

    try:
        return (_safe_resolve(Path(_profiles.get_process_profile_home()).expanduser())
                == _safe_resolve(Path(_profiles.get_hermes_home_for_profile("")).expanduser()))
    except Exception:
        return False  # fail closed: treat the mirrored alias as per-profile


def _env_value_is_profile_sourced(profiles_mod, key, value) -> bool:
    """True when os.environ's *value* for *key* came from a profile's .env rather than the launcher (#8432).

    Two sources put a profile's values into the process environment: the profile loader's mirror (recorded in
    ``_loaded_profile_env_keys``) and a shell launcher that sourced a NAMED launch home's ``.env`` before Python
    started (ctl.sh does this for ``$HERMES_HOME/.env``), recorded by the startup snapshot. A value equal to what the
    named launch home's .env set at startup is that profile's, even if the launcher exported it too (fail closed).
    The root home's .env is shared by definition, so its values are not profile-sourced.
    """
    value = str(value or "").strip()
    if not value:
        return False  # an empty value configures nothing
    if key in (getattr(profiles_mod, "_loaded_profile_env_keys", ()) or ()):
        return True
    launch_env = getattr(profiles_mod, "_INITIAL_HERMES_HOME_DOTENV", None) or {}
    if str(launch_env.get(key) or "").strip() != value:
        return False
    launch_home = str(getattr(profiles_mod, "_INITIAL_HERMES_HOME", "") or "").strip()
    try:
        return not _same_profile_home(launch_home, profiles_mod.get_hermes_home_for_profile(""))
    except Exception:
        return True  # fail closed


def _config_url_may_belong_to_a_profile(profiles_mod, process_home) -> bool:
    """True unless the ambient config's Gateway URL comes from an operator config file outside every profile (#8432).

    Without ``HERMES_CONFIG_PATH`` (or with one a profile .env chose) the ambient config is the process profile's own
    config.yaml. A launcher-provided file outside every profile home is operator configuration, shared, unless its
    URL is a ``${VAR}`` template filled from a profile's .env.
    """
    import re as _re

    from api.config import _load_yaml_config_file_raw
    from api.workspace import _safe_resolve

    override = str(os.environ.get("HERMES_CONFIG_PATH") or "").strip()
    if not override or _env_value_is_profile_sourced(profiles_mod, "HERMES_CONFIG_PATH", override):
        return True
    cfg_dir = _safe_resolve(Path(override).expanduser()).parent
    try:
        own_home = _safe_resolve(Path(process_home).expanduser())
        if cfg_dir == own_home or own_home in cfg_dir.parents:
            return True  # inside the process profile's own home (any layout, isolated included)
    except Exception:
        return True  # fail closed
    try:
        roots = [_safe_resolve(Path(profiles_mod.get_hermes_home_for_profile("")).expanduser() / "profiles")]
        canonical = getattr(profiles_mod, "_profiles_root", None)
        if callable(canonical):
            roots.append(_safe_resolve(Path(canonical())))
    except Exception:
        return True  # fail closed
    if any(root in cfg_dir.parents for root in roots):
        return True  # strictly below the profiles root: inside a named profile's directory
    raw = _load_yaml_config_file_raw(Path(override).expanduser(), _copy=False)
    raw_url = str(raw.get("webui_gateway_base_url") or "") if isinstance(raw, dict) else ""
    for ref in _re.findall(r"\$\{([^}]+)\}", raw_url):
        name = ref.split(":-", 1)[0].strip()
        value = os.environ.get(name)
        if value is not None and _env_value_is_profile_sourced(profiles_mod, name, value):
            return True
    return False


def _shared_url_may_belong_to_process_profile(session_home) -> bool:
    """True unless the shared Gateway URL is provably the launcher's, under a NAMED process profile (#8432).

    With the root profile as the process profile, os.environ and the ambient config are the shared settings: False.
    The process profile's own sessions are never affected: False. Nothing configured anywhere (the default URL)
    belongs to no profile: False. Otherwise the URL may be a profile's (True) when it comes from:

    - os.environ, and the value is profile-sourced (mirrored by the loader, or preloaded from a named launch home);
    - the ambient config, and that file is the process profile's own config.yaml, a file inside a profile home,
      a file a profile .env chose, or an operator file whose ``${VAR}`` URL is filled from a profile's .env;
    - anything undeterminable (an exception): fail closed.
    """
    from api import profiles as _profiles

    if _process_profile_is_root():
        return False
    try:
        process_home = _profiles.get_process_profile_home()
        if _same_profile_home(session_home, process_home):
            return False
    except Exception:
        process_home = None
    raw_env_url = str(os.environ.get(_WEBUI_GATEWAY_BASE_URL_ENV) or "").strip()
    try:
        if raw_env_url:
            return _env_value_is_profile_sourced(_profiles, _WEBUI_GATEWAY_BASE_URL_ENV, raw_env_url)
        from api.config import get_config

        cfg = get_config()
        if not (isinstance(cfg, dict) and str(cfg.get("webui_gateway_base_url") or "").strip()):
            return False
        if process_home is None:
            return True
        return _config_url_may_belong_to_a_profile(_profiles, process_home)
    except Exception:
        return True


def _same_profile_home(a, b) -> bool:
    from api.workspace import _safe_resolve

    return _safe_resolve(Path(a).expanduser()) == _safe_resolve(Path(b).expanduser())


def _gateway_endpoint_for_profile(profile_name) -> tuple[str, str]:
    """URL and key for a session's profile, never those of the process-active profile.

    The shared pair is what the WebUI process was given (launcher environment, default profile's .env, ambient
    config): exactly what a single-profile deployment sends. A profile may override each half on its own:

    - URL: its own HERMES_WEBUI_GATEWAY_BASE_URL in its .env, or webui_gateway_base_url in its config.yaml
      (${VAR} expanded from that profile's .env, read fresh from the file, never from the process environment or
      the cached ambient config); without one it keeps the shared URL.
    - Key, on a URL that is not the shared one: the profile's own HERMES_WEBUI_GATEWAY_API_KEY or API_SERVER_KEY,
      or none; the launcher's and the shared key are never sent to another Gateway.
    - Key, on the shared URL, for another profile: its own HERMES_WEBUI_GATEWAY_API_KEY, then the shared key
      (what v0.52.492 sends), then its API_SERVER_KEY only when there is no shared key at all. A profile's
      API_SERVER_KEY is the key of its OWN Gateway route: the Agent's API server picks the profile from the URL
      and accepts only that profile's key, so on the shared URL it would be a 401.
    - Key, on the shared URL, for the process profile's own sessions: its own HERMES_WEBUI_GATEWAY_API_KEY, then
      the shared alias (the launcher's export, or the root profile's .env), then its API_SERVER_KEY (read fresh
      from its .env), then the shared key.
    - Exception (#8432): under a NAMED process profile, a profile with a key but no URL gets the process pair
      (URL and key from one source, as v0.52.492 sends) unless the shared URL is provably the launcher's (an
      export that no profile .env supplied, or an operator config file outside every profile home), because that
      URL may be a profile's own Gateway and must never receive another profile's key.
    """
    from api import profiles as _profiles
    from api.config import _expand_env_vars, _load_yaml_config_file_raw, _thread_ctx, get_config

    shared_url, shared_key = _gateway_base_url(get_config(), os.environ), _gateway_api_key(os.environ)

    home = Path(_profiles.get_hermes_home_for_profile(str(profile_name or "").strip()))
    own_env = _profiles.filter_runtime_env_for_gateway_parity(_profiles.get_profile_runtime_env(home))
    raw_cfg = _load_yaml_config_file_raw(_profile_config_path(home), _copy=False)
    own_url_cfg = raw_cfg.get("webui_gateway_base_url") if isinstance(raw_cfg, dict) else None
    own_url_env = str(own_env.get(_WEBUI_GATEWAY_BASE_URL_ENV) or "").strip()
    own_alias = str(own_env.get(_WEBUI_GATEWAY_API_KEY_ENV) or "").strip()
    own_key = own_alias or str(own_env.get("API_SERVER_KEY") or "").strip()
    # The launcher's explicit WebUI alias (an export, not a value mirrored from a profile's .env) is the key
    # for the shared URL: there it keeps outranking a profile's API_SERVER_KEY, which is the key of that
    # profile's OWN Gateway (docs: HERMES_WEBUI_GATEWAY_API_KEY first, API_SERVER_KEY as the fallback).
    # An alias the profile loader mirrored from a NAMED active profile's .env is that profile's, not shared; one
    # mirrored from the root profile's ~/.hermes/.env is the shared alias.
    shared_alias = str(os.environ.get(_WEBUI_GATEWAY_API_KEY_ENV) or "").strip()
    if _WEBUI_GATEWAY_API_KEY_ENV in _profiles._loaded_profile_env_keys and not _process_profile_is_root():
        shared_alias = ""

    if not (own_url_env or own_url_cfg or own_key):
        return shared_url, shared_key

    url = shared_url
    if own_url_env or own_url_cfg:
        environ = {k: v for k, v in os.environ.items() if k not in _profiles._loaded_profile_env_keys}
        environ.update(own_env)
        previous_env = getattr(_thread_ctx, "env", {})
        previous_block = bool(getattr(_thread_ctx, "block_process_env_fallback", False))
        _thread_ctx.env = dict(environ)
        _thread_ctx.block_process_env_fallback = True
        try:
            expanded_url = _expand_env_vars(own_url_cfg)
        finally:
            _thread_ctx.env = previous_env
            _thread_ctx.block_process_env_fallback = previous_block
        # a launcher-exported URL still outranks config.yaml, as everywhere else
        url_env = own_url_env or str(environ.get(_WEBUI_GATEWAY_BASE_URL_ENV) or "").strip()
        url = _gateway_base_url({"webui_gateway_base_url": expanded_url} if expanded_url else {},
                                {_WEBUI_GATEWAY_BASE_URL_ENV: url_env} if url_env else {})
        if "${" in url:
            raise RuntimeError(
                f"gateway URL for profile {profile_name or 'default'!r} still contains an unresolved variable "
                f"({url}); set it in that profile's .env"
            )
    if url != shared_url:
        return url, own_key  # the profile's own Gateway: its own key or none, never the shared one
    if not own_url_env and not own_url_cfg and own_key and _shared_url_may_belong_to_process_profile(home):
        # #8432: this profile set a key but no URL, and the "shared" URL may be the NAMED process profile's own
        # (mirrored from its .env or read from its config). Never send this profile's key there; use the process
        # pair, URL and key from one source, exactly what v0.52.492 sends for every fresh turn.
        return shared_url, shared_key
    if _session_is_process_profile(home):
        # The process profile's own sessions: the shared key IS this profile's key, read fresh from its .env.
        return url, (own_alias or shared_alias or own_key or shared_key)
    # Another profile on the shared URL. Its API_SERVER_KEY is the key of its OWN Gateway route (the Agent's API
    # server picks the profile from the URL, its default listener or /p/<profile> under gateway.multiplex_profiles,
    # and accepts only that profile's key), so it never replaces the shared key here: that was a 401 where
    # v0.52.492 sends the shared key. Only an explicit WebUI alias set for this profile
    # (HERMES_WEBUI_GATEWAY_API_KEY in its .env) outranks the shared key; its API_SERVER_KEY is used only when the
    # WebUI was given no shared key at all. A profile that wants its own route sets its own URL.
    return url, (own_alias or shared_key or own_key)


def _session_is_process_profile(session_home) -> bool:
    """True when the session's profile home is the profile the WebUI process runs on (fail: False)."""
    from api import profiles as _profiles

    try:
        return _same_profile_home(session_home, _profiles.get_process_profile_home())
    except Exception:
        return False


def _gateway_config_for_profile(profile_name) -> dict | None:
    """Config of a named non-root profile; None means the process default applies."""
    from api import profiles as _profiles
    from api.config import get_config_for_profile_home

    name = str(profile_name or "").strip()
    if not name or _profiles._is_root_profile(name):
        return None
    return get_config_for_profile_home(_profiles.get_hermes_home_for_profile(name))


def _settle_delegation_reservation(session_id, stream_id, accepted: bool) -> None:
    from api.gateway_delegation_wakeup import settle_reservation

    settle_reservation(session_id, stream_id, accepted)


def _resume_gateway_run_for_session(session) -> bool:
    from api.config import create_stream_channel, register_session_writeback_owner, register_stream_owner

    run = (session.gateway_run if session is not None else None) or {}
    stream_id = str(run.get("stream_id") or "")
    if getattr(session, "_sidecar_names_other_session", None):
        # #8852: the run would be reattached, and its answer written back,
        # under the id this file's body names: another session.
        return False
    # A persisted run is the authority whatever the process default backend is.
    if not stream_id or not (run.get("run_id") or run.get("request")) or session.active_stream_id != stream_id:
        return False
    sid = session.session_id
    endpoint = _gateway_endpoint_for_profile(session.profile)
    # Review #7302 (nesquena, 01-Oct-2026) finding 2 -- "CORE": publish the
    # reattached run's ownership (ACTIVE_RUNS row + retained cancel signal) on the
    # SAME STREAMS_LOCK -> ACTIVE_RUNS_LOCK edge that creates the STREAMS entry,
    # BEFORE the worker is scheduled. A run restored after a restart carries the
    # PREVIOUS process's session.pending_started_at, so the fresh-pending guard in
    # chat/start cannot cover this window: with the stream registered but no
    # ACTIVE_RUNS row, the orphan check classifies the reattach as an orphan,
    # clears the stream plus the Gateway state and admits a SECOND start while the
    # remote run keeps going. The edge and the lock order are the ones the worker
    # (below), Stop and Steer use.
    with STREAMS_LOCK:
        if stream_id in STREAMS:
            return False
        # The record being reattached was loaded from the saved session: a restart
        # can reattach it again, so it is not WebUI-owned drain work (#7293). Durable
        # before the drain can see the stream, so no settle ever selects it.
        _mark_gateway_run_durable(stream_id)
        STREAMS[stream_id] = create_stream_channel()
        CANCEL_FLAGS[stream_id] = CANCEL_FLAGS.get(stream_id, threading.Event())
        # Ownership token: Stop can detach this stream (and a successor could even
        # register the same id) before the worker admits it, so the early-return
        # teardown must retire exactly THIS claim and never someone else's row.
        claim_token = uuid.uuid4().hex
        register_active_run(
            stream_id,
            session_id=sid,
            started_at=time.time(),
            phase="gateway-reattached",
            claim_token=claim_token,
            workspace=str(session.workspace or ""),
            model=session.model,
            provider=session.model_provider,
            backend="gateway",
        )
    register_stream_owner(stream_id, sid)
    register_session_writeback_owner(sid, stream_id)
    _mark_gateway_run_starting(stream_id)
    try:
        threading.Thread(
            target=_run_gateway_chat_streaming,
            args=(sid, session.pending_user_message or "", session.model, session.workspace,
                  stream_id, list(session.pending_attachments or [])),
            kwargs={
                "model_provider": session.model_provider,
                "goal_related": bool(run.get("goal_related")),
                "regeneration": bool(run.get("regeneration")),
                "reattach_run": run,
                "reattach_endpoint": endpoint,
                "reattach_claim_token": claim_token,
            },
            name=f"gateway-reattach-{stream_id[:12]}",
            daemon=True,
        ).start()
    except Exception:
        # A reattach whose worker could not be scheduled must not leave the
        # ownership claim behind (the caller is told to retry/fall back instead of
        # reporting a live reattach). Mirror the worker's own early-return
        # teardown, plus the ACTIVE_RUNS row published above.
        logger.warning(
            "gateway reattach: could not start the worker for stream %s", stream_id,
            exc_info=True,
        )
        _finish_gateway_run_starting(stream_id, result="failed")
        _clear_gateway_run_starting(stream_id)
        try:
            unregister_active_run_if_owned(stream_id, claim_token=claim_token)
        except Exception:
            logger.debug("gateway reattach: could not drop the active run for %s", stream_id, exc_info=True)
        try:
            release_stream_owned_registries(stream_id, session_id=sid)
        except Exception:
            logger.debug("gateway reattach: could not release the stream-owned registries for %s", stream_id, exc_info=True)
        try:
            release_gateway_stream_state(stream_id)
        except Exception:
            logger.debug("gateway reattach: could not release the Gateway state for %s", stream_id, exc_info=True)
        return False
    return True

def _restore_sealed_commentary(session, stream_id) -> None:
    """Restore this turn's sealed interim commentary from the run journal.

    The one place every Gateway turn outcome (completed, failed, cancelled)
    rebuilds the commentary segments the browser saw: the journal is the
    source, rows are display-only (process prose never enters model context),
    and dedup is limited to the current turn (after its last user row). The
    open segment after the last interim boundary is not restored here; each
    outcome writes it itself (final answer, partial or error suffix).
    """
    from api.models import _append_journaled_partial_output

    turn_start = next((
        i + 1 for i in range(len(session.messages) - 1, -1, -1)
        if session.messages[i].get("role") == "user"
    ), len(session.messages))
    _append_journaled_partial_output(
        session, stream_id, dedupe_existing=True,
        dedupe_min_index=turn_start, append_context=False,
        display_only=True, through_last_interim=True,
    )


def _settle_gateway_terminal_error(
    session_id,
    stream_id,
    workspace,
    model,
    model_provider,
    terminal_error,
    *,
    persisted_model=None,
    persisted_model_provider=None,
):
    from api.streaming import (
        _classify_provider_error,
        _materialize_pending_user_turn_before_error,
        _provider_error_payload,
        _session_payload_with_full_messages,
        _snapshot_and_append_partial_on_error,
        _terminal_turn_duration,
    )

    with _get_session_agent_lock(session_id):
        session = get_session(session_id)
        if not _stream_writeback_is_current(session, stream_id):
            return None
        if isinstance(terminal_error, dict):
            error_payload = dict(terminal_error)
            error_classification = {"label": error_payload.get("label") or "Gateway request failed"}
        else:
            error_classification = _classify_provider_error(terminal_error)
            error_payload = _provider_error_payload(
                terminal_error,
                error_classification["type"],
                error_classification.get("hint", ""),
            )
        turn_duration = _terminal_turn_duration(session)
        _materialize_pending_user_turn_before_error(session)
        # Before the open suffix and error marker, while the pending turn
        # still owns the journal output.
        _restore_sealed_commentary(session, stream_id)
        session.active_stream_id = None
        session.gateway_run = None
        session.pending_user_message = None
        session.pending_attachments = []
        session.pending_started_at = None
        session.pending_user_source = None
        try:
            _snapshot_and_append_partial_on_error(session, stream_id)
        except Exception:
            logger.debug("Failed to snapshot gateway partials on terminal error", exc_info=True)
        error_message = {
            "role": "assistant",
            "content": (
                f"**{error_classification['label']}:** "
                f"{error_payload.get('message') or error_classification['label']}"
            ) + (f"\n\n*{error_payload['hint']}*" if error_payload.get("hint") else ""),
            "timestamp": int(time.time()),
            "_error": True,
        }
        if turn_duration is not None:
            error_message["_turnDuration"] = turn_duration
        if error_payload.get("details"):
            error_message["provider_details"] = error_payload["details"]
        if not isinstance(session.messages, list):
            session.messages = []
        session.messages.append(error_message)
        session.workspace = str(workspace)
        session.model = persisted_model if persisted_model is not None else model
        session.model_provider = (
            persisted_model_provider
            if persisted_model_provider is not None
            else model_provider
        )
        terminal_session_persisted = False
        try:
            session.save()
            terminal_session_persisted = True
        except Exception:
            logger.debug("Failed to persist gateway terminal error settlement", exc_info=True)
        from api.helpers import public_session_projection
        error_payload["session"] = public_session_projection(
            _session_payload_with_full_messages(session, tool_calls=[])
        )
        error_payload["session_id"] = session.session_id
        error_payload["terminal_session_persisted"] = terminal_session_persisted
        if terminal_session_persisted:
            error_payload["terminal_session_persisted_session_id"] = session.session_id
        return error_payload


def _settle_gateway_cancelled_turn(session_id, stream_id) -> None:
    """Keep the prompt, the streamed partial, and a cancel marker when the run ended cancelled and Stop did not already settle it."""
    from api.streaming import (
        CANCEL_INITIATOR_SERVER,
        CANCEL_INITIATOR_USER,
        _CANCEL_MARKER_PATTERNS,
        _build_partial_message,
        _materialize_pending_user_turn_before_error,
        _partial_marker_already_present,
        _persist_cancelled_turn,
        _server_settle_reason,
    )

    with _get_session_agent_lock(session_id):
        session = get_session(session_id)
        if not _stream_writeback_is_current(session, stream_id):
            # Successor-owner control: the stream/turn ownership moved on
            # (reattach rotation or a successor admission owns
            # active_stream_id now) — never write over the successor's state.
            return
        # Status-lane cancel persistence (round-3 review): the durable-status
        # lane (cancelled/interrupted resolved by the watchdog's
        # GET /v1/runs/{id} probe) reaches this settle WITHOUT the browser
        # Stop path ever running, so cancel_stream()'s snapshot+upsert never
        # happened and the already-streamed partial answer / reasoning /
        # live tool buffers would be dropped by the teardown finally. This
        # runs the SAME canonical cancellation persistence the SSE-cancel
        # path uses (_build_partial_message + deduped upsert + cancel
        # marker), but at a NEW call site: unlike the inherited SSE-cancel
        # behaviour, the worker itself is settling here — its teardown
        # finally has not popped the buffers yet — so it reads its own
        # buffers directly, without cancel_stream's under-STREAMS_LOCK
        # snapshot dance against a live worker.
        # Empty-output control: _build_partial_message returns None when all
        # three buffers are empty, so nothing is appended — no empty rows.
        partial_msg = _build_partial_message(
            LiveTurnSnapshot.partial_text(stream_id),
            LiveTurnSnapshot.reasoning_text(stream_id),
            LiveTurnSnapshot.tool_calls(stream_id),
        )
        # Row order matches cancel_stream(): the user turn is materialized
        # before the partial row is placed, and the partial is inserted
        # before any cancel marker that is already present. Clearing the
        # NOTE: the pending identity fields are cleared AFTER the reconcile
        # materialize idempotent (no duplicate user row).
        _materialize_pending_user_turn_before_error(session)
        # Reconcile BEFORE any downstream row-shuffling — including the
        # display partial insert below (maintainer re-gate 2026-10-12 09:01
        # UTC item 1). With an EMPTY context the reconcile seeds itself from
        # the transcript projection, so a display partial already sitting in
        # session.messages was projected into the seeded context BEFORE the
        # recovered prompt: the seeded `_recovered` row then sat before an
        # assistant tail, the partial-append guard saw no owning-prompt tail
        # (the seeded `_recovered` flag was never cleared), and the
        # admission builder's answered-turn rule dropped the prompt while
        # its partial appeared to answer an older unanswered turn. Seeding
        # while the transcript still ends with the owning prompt lets the
        # `_recovered`-flag clear / partial append see it. This keeps the
        # round-3 ordering contract — the reconcile needs the pending turn
        # identity (pending_user_message / pending_started_at) to mirror the
        # interrupted turn (prompt + streamed partial) into the model-facing
        # context; the status-lane materialization above does NOT clear that
        # identity (only _persist_cancelled_turn below does), so this
        # ordering holds for both cancellation lanes — and extends it: the
        # reconcile must precede the DISPLAY insert too. partial_ts shares
        # the display partial's whole-second stamp (re-gate item 5) so the
        # display merge key cannot insert a second copy of the partial on
        # the next successful turn.
        _reconcile_gateway_cancelled_context(
            session,
            stream_id,
            partial_ts=partial_msg.get("timestamp") if partial_msg is not None else None,
        )
        # Seed/reconcile model context first, then restore sealed commentary
        # ahead of the open suffix without clearing turn ownership.
        _restore_sealed_commentary(session, stream_id)
        if partial_msg is not None:
            if not isinstance(session.messages, list):
                session.messages = []
            marker_idx = len(session.messages)
            for idx in range(len(session.messages) - 1, -1, -1):
                row = session.messages[idx]
                # Maintainer must-fix (round-3 re-review): never cross the
                # current user-turn boundary — a cancel marker above it
                # belongs to a PREVIOUS cancelled turn, and inserting this
                # turn's partial before it scrambles the transcript order
                # after reload.
                if isinstance(row, dict) and row.get("role") == "user":
                    break
                if not isinstance(row, dict) or row.get("role") != "assistant":
                    continue
                normalized = str(row.get("content") or "").strip().lower()
                if any(pattern in normalized for pattern in _CANCEL_MARKER_PATTERNS):
                    marker_idx = idx
                    break
            if not _partial_marker_already_present(session.messages, partial_msg, before_idx=marker_idx):
                session.messages.insert(marker_idx, partial_msg)
        # Identity cleared only NOW: the reconcile above consumed
        # pending_user_message / pending_started_at as the turn identity.
        session.pending_user_message = None
        session.pending_attachments = []
        session.pending_started_at = None
        session.pending_user_source = None
        # A server settle (#7293) that cancelled this launch before its handoff can
        # lose the session lock to this worker: write its marker, not a user one.
        _server_reason = _server_settle_reason(stream_id)
        _persist_cancelled_turn(
            session,
            message=_server_reason or "Cancelled by gateway",
            initiator=CANCEL_INITIATOR_SERVER if _server_reason is not None else CANCEL_INITIATOR_USER,
        )
        session.gateway_run = None
        session.save()


# Rows reconciled at cancel time carry the settling stream id so a replayed
# settle of the same turn can never append the same partial twice.
_GATEWAY_CANCEL_RECONCILED_STREAM_KEY = "_gateway_cancelled_stream"


def _gateway_stream_partial_text(stream_id) -> str:
    """Best-effort read of the partial assistant text streamed for a run."""
    try:
        return LiveTurnSnapshot.partial_text(stream_id)
    except Exception:
        return ""


def _project_transcript_context_rows(messages) -> list:
    """Project the canonical display transcript into model-facing context rows.

    The same row filters the runs-API history builder applies: error markers,
    empty partials, reasoning-only assistant rows, and non user/assistant
    roles never seed the model context. ``_recovered`` flags are KEPT so the
    builder's answered-turn rule still decides visibility at send time,
    timestamps are preserved verbatim so transcript merges recognize the rows
    as the same ones, and each user row's ``_active_turn_token`` is KEPT so
    same-second interrupts of one prompt stay distinguishable after the seed
    (a token-less seeded row collapses the next interrupt into the content +
    int-second fallback). Used to seed an EMPTY context_messages from
    session.messages (imported CLI sessions, first gateway turns cancelled
    gateway-side) before the interrupted turn is appended.
    """
    from api.streaming import (
        _is_non_replayable_history_row,
        _is_reasoning_only_assistant_message,
        _strip_oob_blocks,
    )

    rows = []
    for entry in messages or []:
        if not isinstance(entry, dict):
            continue
        if _is_non_replayable_history_row(entry) or _is_reasoning_only_assistant_message(entry):
            continue
        # Journal-recovered rows without a proven context owner are display-only:
        # projecting them would hand the model an answer it never produced in
        # this context (same exclusion master's cancel path applies).
        if entry.get("_recovered_display_only") is True:
            continue
        role = str(entry.get("role") or "").strip().lower()
        if role not in {"user", "assistant"}:
            continue
        content = entry.get("content")
        if content is None:
            continue
        row = {"role": role, "content": _strip_oob_blocks(content)}
        ts = entry.get("timestamp")
        if isinstance(ts, (int, float)) and not isinstance(ts, bool) and ts > 0:
            row["timestamp"] = ts
        if entry.get("_recovered"):
            row["_recovered"] = True
        row_token = entry.get("_active_turn_token")
        if row_token:
            # Turn identity survives the projection (maintainer re-gate
            # 2026-10-12 09:01 UTC item 2): a second interrupt of the same
            # prompt inside one wall second is distinguishable from the
            # first only by its stream-bound token, and a token-less seeded
            # row makes the mirror dedup fall back to content + int-second
            # and swallow the second turn. Tokens are stream-scoped
            # (stream_id:started_at), never session-scoped, so a preserved
            # token can only ever match the exact stream that created it —
            # there is no fork boundary where stripping it is required.
            row["_active_turn_token"] = str(row_token)
        rows.append(row)
    return rows


def _drop_active_turn_rows_for_writeback(rows, active_turn_identity) -> list:
    """Remove the active turn's retained rows from a model-context snapshot.

    The regeneration start restamps the retained interrupted-turn user row
    with the NEW stream's active-turn token while the same prompt is also
    sent as ``input``. Keeping the row in the snapshot duplicates the prompt:
    once in the outbound conversation_history, once more when the success
    writeback appends the freshly materialized user row (re-gate 2026-10-05
    item 3). The reconciled partial of the SAME interrupted turn — stamped
    with the cancel-reconciliation key — is dropped with it: the turn is
    being re-run wholesale, so a partial of the answer it is about to
    regenerate must not ride along.
    """
    from api.streaming import _active_turn_token_matches

    kept = []
    drop_reconciled_partial = False
    for row in rows:
        if _active_turn_token_matches(row, active_turn_identity):
            drop_reconciled_partial = True
            continue
        if (
            drop_reconciled_partial
            and isinstance(row, dict)
            and row.get("role") == "assistant"
            and row.get(_GATEWAY_CANCEL_RECONCILED_STREAM_KEY)
        ):
            continue
        drop_reconciled_partial = False
        kept.append(row)
    return kept


def _reconcile_gateway_cancelled_context(session, stream_id, *, partial_text=None, partial_ts=None) -> bool:
    """Mirror an interrupted gateway turn into the model-facing context.

    Called when a gateway run settles cancelled/interrupted so WebUI's
    context_messages includes everything the gateway persisted at the moment
    of interruption. With that snapshot complete, the next turn's always-send
    conversation_history is correct for every flow: fork (snapshot includes
    the parent prefix), truncate/undo/retry (snapshot is the shortened one),
    prefill (snapshot + prefill compose), and a normal user Stop (cancel_stream
    invokes this while the stream id is still current). Session state is saved
    by the caller.

    Order: the canonical transcript projection when the context is EMPTY (an
    imported CLI session or a first gateway turn cancelled gateway-side still
    has the whole conversation in session.messages — never fabricate a
    two-row world that would replace the gateway's stored transcript, re-gate
    2026-10-05 item 1), then the interrupted turn's pending user row, then the
    partial assistant text the browser saw (STREAM_PARTIAL_TEXT for the
    stream; skipped when empty).

    Idempotent per turn/stream identity: the partial row is stamped with the
    settling stream id and user-row mirrors are deduped by content and
    timestamp, so double settlement appends nothing twice. A replayed settle
    can still make the pending-user materializer re-append the prompt AFTER
    the already-reconciled partial (its exact-checkpoint guard only inspects
    the context tail); such trailing duplicates are collapsed here.

    ``partial_ts`` optionally shares ONE whole-second stamp between the
    model-facing partial row and the display partial cancel_stream appends
    (re-gate item 5): two independent int(time.time()) calls can straddle a
    second boundary and the display merge key would then insert a second
    copy of the partial on the next successful turn.

    Returns True when the context changed.
    """
    from api.process_event_utils import build_active_turn_token
    from api.streaming import (
        _build_partial_message,
        _normalize_user_text,
        stamp_message_source,
    )

    if session is None or not stream_id:
        return False
    context = getattr(session, "context_messages", None)
    if not isinstance(context, list):
        context = []
        session.context_messages = context

    changed = False

    # (0) An empty context is not an empty world: seed the canonical
    # transcript projection (the same row filters the runs-API history
    # builder applies) BEFORE appending the interrupted turn, so the next
    # turn's conversation_history carries the prior conversation + the
    # interrupted turn instead of replacing the gateway's stored transcript
    # with a two-row world.
    if not context:
        seeded = _project_transcript_context_rows(getattr(session, "messages", None))
        if seeded:
            context.extend(seeded)
            changed = True

    pending_text = str(getattr(session, "pending_user_message", None) or "")
    pending_started_at = getattr(session, "pending_started_at", None)
    pending_token = build_active_turn_token(stream_id, pending_started_at)
    prompt_appended_this_call = False

    # (a) The interrupted turn's user prompt. cancel_stream and
    # _persist_cancelled_turn materialize it into session.messages; mirror the
    # same row (content + pending_started_at timestamp) into the context
    # unless an identical row is already there.
    if pending_text:
        try:
            recovered_ts = int(time.time())
        except AttributeError:
            # Watchdog harness patches gc.time with a monotonic-only stub.
            recovered_ts = int(partial_ts) if partial_ts is not None else 0
        if isinstance(pending_started_at, (int, float)) and pending_started_at > 0:
            recovered_ts = int(pending_started_at)
        normalized_pending = _normalize_user_text(pending_text)
        already_mirrored = False
        for row in reversed(context[-8:]):
            if not (isinstance(row, dict) and row.get("role") == "user"):
                continue
            if _normalize_user_text(str(row.get("content") or "")) != normalized_pending:
                break
            if (
                pending_token
                and row.get("_active_turn_token")
                and row.get("_active_turn_token") != pending_token
            ):
                # Text + integer-second timestamp is not ownership (re-gate
                # 2026-10-06 item 3): a DIFFERENT stream's identical prompt
                # interrupted inside the same second carries its own token,
                # and this turn must still be mirrored. Rows without a token
                # (legacy data, lossless-unto-them projections) keep the
                # established timestamp check.
                break
            try:
                row_ts = int(row.get("timestamp") or 0)
            except (TypeError, ValueError):
                row_ts = None
            if row_ts is None or row_ts == recovered_ts:
                already_mirrored = True
            break
        if not already_mirrored:
            user_row = {
                "role": "user",
                "content": pending_text,
                "timestamp": recovered_ts,
                "_recovered": True,
            }
            if pending_token:
                # Turn identity for the token-based exclusion on the next
                # turn: the regenerate path must drop THIS row from outbound
                # history and from the writeback snapshot instead of
                # duplicating the prompt (re-gate item 3).
                user_row["_active_turn_token"] = pending_token
            stamp_message_source(
                user_row, getattr(session, "pending_user_source", None) or "webui",
            )
            pending_attachments = list(getattr(session, "pending_attachments", None) or [])
            if pending_attachments:
                user_row["attachments"] = pending_attachments
            context.append(user_row)
            prompt_appended_this_call = True
            changed = True

    def _user_row_identity(row):
        """Comparable identity for a user context row (or None).

        The active-turn token is part of the identity (re-gate 2026-10-06
        item 3): two interrupts of the same prompt inside one second share
        content and integer-second timestamp but never a token, so a replay
        collapse for one turn must never swallow the other turn's row.
        """
        if not isinstance(row, dict) or row.get("role") != "user":
            return None
        try:
            row_ts = int(row.get("timestamp") or 0)
        except (TypeError, ValueError):
            row_ts = -1
        return (
            _normalize_user_text(str(row.get("content") or "")),
            row_ts,
            row.get("_active_turn_token"),
        )

    # Locate a partial this stream already reconciled, and collapse any
    # replayed user rows the materializer appended after it.
    reconciled_idx = None
    for idx in range(len(context) - 1, -1, -1):
        row = context[idx]
        if (
            isinstance(row, dict)
            and row.get(_GATEWAY_CANCEL_RECONCILED_STREAM_KEY) == stream_id
        ):
            reconciled_idx = idx
            break
    if reconciled_idx is not None:
        seen_user_identities = {
            _user_row_identity(row) for row in context[:reconciled_idx]
        }
        seen_user_identities.discard(None)
        kept = []
        dropped_replay = False
        for row in context[reconciled_idx + 1:]:
            identity = _user_row_identity(row)
            if (
                identity is not None
                and identity in seen_user_identities
                and isinstance(row, dict)
                and row.get("_recovered")
            ):
                dropped_replay = True
                continue
            if identity is not None:
                seen_user_identities.add(identity)
            kept.append(row)
        if dropped_replay:
            context[reconciled_idx + 1:] = kept
            changed = True

    def _interrupted_turn_owns_tail_row(row) -> bool:
        """True when the trailing user context row IS the interrupted prompt.

        Ownership is the turn identity, never "any trailing user row" (re-gate
        item 2): the row this reconcile just materialized, a row carrying this
        stream's active-turn token (a replayed settle after the pending state
        was cleared), or an identical content + pending_started_at timestamp
        row (the transcript-recovery materializer's shape).
        """
        if prompt_appended_this_call:
            return True
        if not isinstance(row, dict):
            return False
        if pending_token and row.get("_active_turn_token") == pending_token:
            return True
        if not pending_text:
            return False
        if _normalize_user_text(str(row.get("content") or "")) != _normalize_user_text(pending_text):
            return False
        if not (isinstance(pending_started_at, (int, float)) and pending_started_at > 0):
            return True
        try:
            row_ts = int(row.get("timestamp") or 0)
        except (TypeError, ValueError):
            return False
        return row_ts == int(pending_started_at)

    # (b) The partial assistant text the browser saw (what the gateway's
    # persisted incomplete snapshot holds). Only appended while the context
    # tail is the interrupted prompt's OWN row — identity/token matched, not
    # merely a trailing user row, which can be an older unanswered prompt
    # (re-gate 2026-10-05 item 2).
    if reconciled_idx is None:
        raw_partial = (
            partial_text
            if partial_text is not None
            else _gateway_stream_partial_text(stream_id)
        )
        partial_row = _build_partial_message(raw_partial, "", []) if raw_partial else None
        if partial_row is not None:
            if context and _interrupted_turn_owns_tail_row(context[-1]):
                # The partial is about to ride directly under its verified
                # owning user row: that pair IS the interrupted turn, so the
                # row is no longer a possibly-unanswered recovered prompt.
                # Clearing the flag keeps the runs-API history builder from
                # dropping the question and pairing the partial with the
                # wrong one (re-gate 2026-10-06 item 1).
                context[-1].pop("_recovered", None)
                if partial_ts is not None:
                    partial_row["timestamp"] = partial_ts
                partial_row[_GATEWAY_CANCEL_RECONCILED_STREAM_KEY] = stream_id
                context.append(partial_row)
                changed = True
            else:
                logger.debug(
                    "Skipping gateway cancel partial mirror for stream %s: "
                    "interrupted prompt is not the context tail",
                    stream_id,
                )
    return changed


def _stream_writeback_is_current(session: Any, stream_id: str) -> bool:
    return bool(stream_id and getattr(session, "active_stream_id", None) == stream_id)


def _clear_gateway_pending_state(session: Any, stream_id: str) -> None:
    if not _stream_writeback_is_current(session, stream_id):
        # Cancel clears active_stream_id eagerly; still drop this stream's run record.
        if session is not None and (session.gateway_run or {}).get("stream_id") == stream_id:
            session.gateway_run = None
            session.save(touch_updated_at=False)
        return
    session.active_stream_id = None
    session.gateway_run = None
    session.pending_user_message = None
    session.pending_attachments = None
    session.pending_started_at = None
    session.pending_user_source = None
    session.save()


def _cleanup_gateway_pending_mirror(session_id: str) -> None:
    try:
        from api.route_approvals import (
            retire_gateway_pending_mirror,
        )

        retire_gateway_pending_mirror(session_id)
    except Exception:
        logger.debug("Failed to reconcile gateway pending mirror during teardown", exc_info=True)


def settle_gateway_stream_approvals(session_id: str, stream_id: str, *, reason: str) -> None:
    """Fail-close every approval card of the Gateway run ``stream_id`` maps to.

    Shared by the stream worker's teardown and ``cancel_stream`` (user Stop and
    the #7293 server settle). Stop must retire the run's cards itself: the
    worker can stay blocked on the Gateway socket long after the turn was saved
    as interrupted, and until its teardown a click on the card was still relayed
    and resumed a run WebUI had already recorded as stopped.

    Deliberately lock-free beyond the approval registry's own ``_lock``: Stop and
    the server settle must never wait on an approval relay (whose POST is bounded
    only by the runner client's timeout). Once this returns, a click finds the
    card gone and gets 409 without reaching the Gateway. A relay that had
    already validated its card before this ran still sends its one POST; that is
    a click that effectively happened just before Stop.
    """
    from api.route_approvals import settle_gateway_pending_run

    run_id = str(_STREAM_RUN_IDS.get(stream_id) or "").strip() if stream_id else ""
    if not session_id or not run_id:
        return
    settle_gateway_pending_run(session_id, run_id, reason=reason)


def _submit_stream_approval_mirror(session_id: str, approval_data: dict, cancel_event) -> tuple[dict | None, int]:
    """Publish a default-stream approval card unless its stream is being stopped.

    Stop sets the cancel flag and then retires the run's cards under the approval
    registry lock. The insertion re-checks the flag under that same lock
    (``submit_gateway_pending_mirror(..., cancel_event=...)``), so either the
    card is inserted before Stop's retirement (which then removes it) or the
    insertion sees the flag and publishes nothing. No card can appear after
    Stop has returned, not even before the recheck below. The recheck after
    publishing stays as a second line of defence. (A click that validated the
    card in the instant before it was retired still relays once; that is a
    click that happened before Stop, not after it.)
    """
    from api.route_approvals import settle_gateway_pending_run, submit_gateway_pending_mirror

    if cancel_event.is_set():
        return None, 0
    head, total = submit_gateway_pending_mirror(session_id, approval_data, cancel_event=cancel_event)
    run_id = str(approval_data.get("run_id") or "").strip()
    if run_id and cancel_event.is_set():
        settle_gateway_pending_run(
            session_id, run_id, reason="Gateway run was cancelled before approval resolution",
        )
        return None, 0
    return head, total


def _run_gateway_chat_streaming(
    session_id,
    msg_text,
    model,
    workspace,
    stream_id,
    attachments=None,
    *,
    model_provider=None,
    persisted_model=None,
    persisted_model_provider=None,
    goal_related=False,
    regeneration=False,
    reattach_run=None,
    reattach_endpoint=None,
    reattach_claim_token=None,
):
    """Bridge a WebUI chat turn through Hermes Gateway's API server.

    This default-off path keeps the browser contract unchanged: /api/chat/start
    still returns a local stream_id and /api/chat/stream still receives WebUI SSE
    event names. The worker translates OpenAI-compatible streaming chunks from
    the configured Gateway API server into those local events and persists the
    final user/assistant turn back into the WebUI session.
    """
    cancel_event = threading.Event()
    q = peek_stream(stream_id)
    if q is not None:
        # A snapshot lookup is not admission. A concurrent chat/start can classify
        # this stream as an orphan (registered, no live worker, no pending turn in
        # the registration window) and clear it between peek_stream() and the
        # registration below; publishing ourselves active afterwards would run a
        # turn with no transport/owner state, allowing overlapping turns and
        # duplicate provider/tool effects. Claim the stream, its retained cancel
        # signal and the ACTIVE_RUNS registration on ONE STREAMS_LOCK ->
        # ACTIVE_RUNS_LOCK edge -- the order Stop/Steer use, and the same edge the
        # in-process worker uses (api/streaming.py) -- revalidating stream
        # membership and cancellation in-lock and failing closed when ownership is
        # already gone.
        with STREAMS_LOCK:
            cancel_event = CANCEL_FLAGS.get(stream_id, cancel_event)
            if stream_id not in STREAMS or cancel_event.is_set():
                q = None
            else:
                CANCEL_FLAGS[stream_id] = cancel_event
                LiveTurnSnapshot.register(stream_id)
                register_active_run(
                    stream_id,
                    session_id=session_id,
                    started_at=time.time(),
                    phase="gateway-starting",
                    workspace=str(workspace),
                    model=model,
                    provider=model_provider,
                    backend="gateway",
                )
                # Worker admission ends the launch phase (#7302 finding 5): retire
                # the claim published at registration. Ownership lives in
                # ACTIVE_RUNS from here on.
                from api.config import retire_pre_admission_claim_if_owned

                retire_pre_admission_claim_if_owned(stream_id, streams_lock_held=True)
    if q is None:
        _finish_gateway_run_starting(stream_id, result="fallback")
        _clear_gateway_run_starting(stream_id)
        # Cancelled or orphan-cleared before the worker was admitted: no teardown
        # finally runs on this early-return path, so release the COMPLETE set of
        # stream-owned registries the route layer registered (stream rows,
        # owners, writeback, goal classification) and then the Gateway-owned
        # lifecycle rows. release_gateway_stream_state() is no-op-safe and must
        # run OUTSIDE STREAMS_LOCK -- it owns the lifecycle/waiter protocol and
        # also drops the endpoint mapping the old owner-only release leaked.
        # Stop can detach this stream between the pre-admission claim and this
        # admission (review 2026-10-01, third finding): cancel_stream() marks the
        # row cancelling and pops STREAMS/CANCEL_FLAGS while this worker is still
        # unwinding, and this early return never reaches the worker's `finally`.
        # Retire exactly the claim we published, by identity -- a row a successor
        # registered for the same stream id must survive. Without this the ghost
        # row reports false liveness/busy in _run_lifecycle_health() and delays
        # background wakeups (LAST_RUN_FINISHED_AT never advances).
        if reattach_claim_token:
            try:
                if unregister_active_run_if_owned(stream_id, claim_token=reattach_claim_token):
                    logger.info(
                        "gateway reattach: retired the pre-admission claim for stream %s "
                        "after cancellation", stream_id,
                    )
            except Exception:
                logger.debug(
                    "gateway reattach: could not retire the pre-admission claim for %s",
                    stream_id, exc_info=True,
                )
        release_stream_owned_registries(stream_id, session_id=session_id)
        release_gateway_stream_state(stream_id)
        return
    try:
        run_journal = RunJournalWriter(session_id, stream_id)
    except Exception:
        run_journal = None
        logger.debug("Failed to initialize gateway run journal for stream %s", stream_id, exc_info=True)

    success_writeback_committed = False
    runs_api_pending_marked = True
    preserve_gateway_pending = False
    canonical_snapshot = None

    def put_gateway_event(event, data):
        if cancel_event.is_set() and not success_writeback_committed and event not in ("cancel", "error", "apperror"):
            return
        if event == "apperror" and isinstance(data, dict):
            data = data.copy()
            data.setdefault("session_id", session_id)
        event_id = None
        if run_journal is not None:
            try:
                journaled = run_journal.append_sse_event(event, data)
                event_id = (journaled or {}).get("event_id") if isinstance(journaled, dict) else None
                if event_id:
                    STREAM_LAST_EVENT_ID[stream_id] = event_id
            except Exception:
                logger.debug("Failed to append gateway event %s for stream %s", event, stream_id, exc_info=True)
        if event_id and hasattr(q, "note_last_event_id"):
            try:
                q.note_last_event_id(event_id)
            except Exception:
                logger.debug("Failed to note gateway event_id %s for stream %s", event_id, stream_id, exc_info=True)
        try:
            queue_item = (event, data, event_id) if hasattr(q, "subscribe_with_snapshot") else (event, data)
            q.put_nowait(queue_item)
        except Exception:
            logger.debug("Failed to put gateway event to queue")

    s = None
    api_key = ""
    final_text = ""
    terminal_error = ""
    usage = {"input_tokens": 0, "output_tokens": 0, "estimated_cost": 0}
    try:
        s = get_session(session_id)
        # Profile-bound like the endpoint below: the worker thread has no request profile TLS.
        from api.gateway_sessions import config_for_session
        cfg = config_for_session(vars(s))
        # A legacy session (no stored effort) inherits agent.reasoning_effort from its
        # own profile's isolated config, as the composer chip and the local worker do:
        session_effort = getattr(s, "reasoning_effort", None)
        try:
            from api.config import effective_session_reasoning_effort
            from api.profiles import get_hermes_home_for_profile

            session_effort = effective_session_reasoning_effort(
                session_effort, get_hermes_home_for_profile(getattr(s, "profile", None)),
            )
        except Exception:
            logger.warning("isolated reasoning config read failed; using the session config", exc_info=True)
        reasoning_effort = _gateway_reasoning_effort_for_request(
            cfg,
            model=model,
            model_provider=model_provider,
            session_effort=session_effort,
        )
        # The session's own profile, not the process-active one: this worker thread has no
        # thread-local profile, so get_config()/os.environ describe whichever profile the
        # process loaded. Same resolver the restart-reattach path already uses.
        base_url, api_key = reattach_endpoint or _gateway_endpoint_for_profile(getattr(s, "profile", None))
        with _STREAM_RUN_STARTING_CONDITION:
            _STREAM_ENDPOINTS[stream_id] = (base_url, api_key)
        try:
            from api.config import _main_model_request_overrides
            _gw_overrides = _main_model_request_overrides(
                cfg,
                effective_model=model,
                effective_provider=model_provider,
            )
        except Exception:
            _gw_overrides = {}
        from api.gateway_transport import select_transport

        transport = select_transport(s, cfg, base_url=base_url, api_key=api_key, reattach_run=reattach_run)
        if not transport.durable and runs_api_pending_marked:
            # No run id will ever be published: release Stop/approval waiters now.
            _finish_gateway_run_starting(stream_id, result="fallback")
            runs_api_pending_marked = False
        try:
            from api.streaming import (
                _load_webui_prefill_context,
                _prefill_messages_with_webui_context,
                _normalize_prefill_messages_before_user_turn,
                _public_prefill_context_status,
                _webui_ephemeral_system_prompt,
            )

            prefill_context = _load_webui_prefill_context(cfg)
            # #3324: the WebUI session/delivery context (connected platforms,
            # home channels, delivery hints, session framing) is now carried in
            # the ephemeral system prompt rather than a prefill `user` message.
            # The gateway-backed path must build the SAME system prompt so that
            # context is not silently dropped on Gateway-routed WebUI chats.
            _gateway_system_prompt = _webui_ephemeral_system_prompt(
                None,
                surface_context={
                    "source": "webui",
                    "session_id": session_id,
                    "profile": getattr(s, "profile", None),
                    "workspace": s.workspace if s is not None else str(workspace),
                },
                config_data=cfg,
            )
            prefill_messages = _prefill_messages_with_webui_context(prefill_context, cfg)
            prefill_messages = _normalize_prefill_messages_before_user_turn(prefill_messages)
            prefill_messages = [
                {"role": "system", "content": _gateway_system_prompt},
                *prefill_messages,
            ]
            put_gateway_event("context_status", {
                "session_id": session_id,
                "prefill": _public_prefill_context_status(prefill_context),
            })
        except Exception:
            logger.debug("Failed to load WebUI gateway prefill context", exc_info=True)
            prefill_messages = []
        request_options = {}
        if model_provider:
            request_options["provider"] = model_provider
        request_options.update(_gateway_reasoning_body_fields(reasoning_effort))
        if _gw_overrides.get("service_tier"):
            request_options["service_tier"] = _gw_overrides["service_tier"]

        def record_run(run_id, request=None, **extra):
            _record_gateway_run(
                session_id, stream_id, run_id, request=request,
                regeneration=bool(regeneration), goal_related=bool(goal_related), **extra,
            )
            if run_id:  # the Gateway accepted the run: its delegation lease is delivered
                _settle_delegation_reservation(session_id, stream_id, accepted=True)

        from api.gateway_transport import GatewayTurn, TurnOutcomeUnknown

        turn = GatewayTurn(
            session=s, session_id=session_id, stream_id=stream_id, msg_text=msg_text,
            model=model, model_provider=model_provider, workspace=workspace,
            attachments=attachments, cfg=cfg, base_url=base_url, api_key=api_key,
            prefill_messages=prefill_messages, request_options=request_options,
            emit=put_gateway_event, cancel_event=cancel_event,
            record_run=record_run, reattach_run=reattach_run,
        )
        # ---- the only transport-specific step of the turn ----
        try:
            result = transport.run(turn)
        except TurnOutcomeUnknown as exc:
            preserve_gateway_pending = True
            put_gateway_event("apperror", {"label": "Gateway connection lost",
                "type": "gateway_session_disconnected", "message": str(exc),
                "hint": "Reopen the conversation to recover the existing turn. Do not resend it."})
            return
        except Exception as exc:
            logger.debug("Gateway %s transport failed for stream %s", transport.name, stream_id, exc_info=True)
            result = TurnResult(
                text="",
                error=(
                    _gateway_http_error_event(exc, api_key_configured=bool(api_key))
                    if isinstance(exc, urllib.error.HTTPError) else str(exc)
                ),
            )
        usage.update({k: v for k, v in result.usage.items() if v})
        if result.text is None:
            # Cancelled: the transport announced it; persist the partial turn.
            _settle_gateway_cancelled_turn(session_id, stream_id)
            return
        final_text, terminal_error, canonical_snapshot = result.text, result.error, result.canonical_snapshot
        assistant_text = final_text.strip()
        if terminal_error:
            error_payload = _settle_gateway_terminal_error(
                session_id,
                stream_id,
                workspace,
                model,
                model_provider,
                terminal_error,
                persisted_model=persisted_model,
                persisted_model_provider=persisted_model_provider,
            )
            if error_payload is None:
                return
            put_gateway_event("apperror", error_payload)
            return
        if not assistant_text:
            put_gateway_event("apperror", {
                "label": "Gateway returned no response",
                "type": "gateway_empty_response",
                "message": "Gateway returned no assistant message for this turn.",
                "hint": "Check that Hermes Gateway API server is running and reachable.",
            })
            return
        with _get_session_agent_lock(session_id):
            s = get_session(session_id)
            if not _stream_writeback_is_current(s, stream_id):
                return
            # A late Stop can land after Gateway has yielded a full answer but
            # before success writeback. Treat it as cancellation so any
            # credential-exhausted process-wakeup pause stays in place.
            if cancel_event.is_set():
                put_gateway_event("cancel", {"message": "Cancelled by user"})
                return
            now = time.time()
            # Preserve subsecond ordering for gateway-backed turns. Using an
            # integer seconds timestamp gives the user and assistant rows the
            # same sort key; later transcript merges can then fall back to
            # role/content ordering instead of turn order.
            assistant_ts = now + 0.000001
            pending_source = getattr(s, "pending_user_source", None) or "webui"
            from api.streaming import (
                _active_turn_authority,
                _active_turn_token_matches,
                _materialize_active_turn_user,
            )

            active_turn_identity = _active_turn_authority(s, stream_id, msg_text)
            user_msg = _materialize_active_turn_user(
                active_turn_identity,
                str(msg_text or ""),
                pending_source,
            )
            user_msg["timestamp"] = float(
                active_turn_identity.get("timestamp") or now
            )
            assistant_msg = {"role": "assistant", "content": assistant_text, "timestamp": assistant_ts}
            saved_reasoning = LiveTurnSnapshot.reasoning_text(stream_id)
            if saved_reasoning:
                assistant_msg["reasoning"] = saved_reasoning
            previous_messages = list(getattr(s, "messages", None) or [])
            stored_context = getattr(s, "context_messages", None)
            previous_context = list(
                stored_context
                if isinstance(stored_context, list) and (regeneration or stored_context)
                else getattr(s, "messages", None) or []
            )
            # The active turn's retained user row (restamped with this
            # stream's token at regeneration start) is re-materialized below
            # as user_msg; dropping it from the snapshot keeps the saved model
            # context at exactly one copy of the prompt (re-gate 2026-10-05
            # item 3), and the interrupted turn's reconciled partial goes with
            # it for the same reason.
            previous_context = _drop_active_turn_rows_for_writeback(
                previous_context, active_turn_identity,
            )
            previous_process_wakeup_pause = dict(getattr(s, "process_wakeup_pause", {}) or {})
            # Stamp stable ids on the two new rows (shared with the display merge
            # below) so display and model-context copies share an id for the
            # fork/truncate aligner (#context-message-stable-id).
            try:
                from api.streaming import _assign_stable_message_ids

                _assign_stable_message_ids(
                    [user_msg, assistant_msg],
                    previous_context,
                    list(getattr(s, "messages", None) or []),
                )
            except Exception:
                logger.debug("Failed to stamp stable ids on gateway turn rows", exc_info=True)
            s.context_messages = previous_context + [user_msg, assistant_msg]
            try:
                from api.streaming import _is_context_compression_marker

                display_context = [
                    msg
                    for msg in previous_context
                    if not _is_context_compression_marker(msg)
                ]
            except Exception:
                logger.debug("Failed to filter gateway display context markers", exc_info=True)
                display_context = previous_context
            display = merge_session_messages_append_only(
                previous_messages,
                display_context,
            )
            try:
                from api.streaming import _merge_display_messages_after_agent_result

                s.messages = _merge_display_messages_after_agent_result(
                    display,
                    previous_context,
                    s.context_messages,
                    str(msg_text or ""),
                    source=pending_source,
                    verification_nudge_provenance={
                        "active_turn_identity": active_turn_identity,
                    },
                )
            except Exception:
                logger.debug("Failed to merge gateway display transcript", exc_info=True)
                # Avoid duplicating the eager-save checkpointed user message.
                if display:
                    latest = display[-1]
                    if isinstance(latest, dict) and latest.get("role") == "user":
                        latest_text = " ".join(str(latest.get("content") or "").split())
                        msg_norm = " ".join(str(msg_text or "").split())
                        if latest_text == msg_norm:
                            display = display[:-1]
                s.messages = display + [user_msg, assistant_msg]
            if s.messages and s.messages[-1].get("content") == assistant_text:
                # Sealed commentary goes before the final row.
                final_display = s.messages.pop()
                try:
                    _restore_sealed_commentary(s, stream_id)
                finally:
                    s.messages.append(final_display)
            if active_turn_identity.get("token"):
                current_display_rows = [
                    message
                    for message in s.messages
                    if _active_turn_token_matches(message, active_turn_identity)
                ]
                if len(current_display_rows) == 1:
                    current_display_rows[0]["timestamp"] = user_msg["timestamp"]
            if canonical_snapshot is not None:
                # Gateway history remains authoritative across surfaces. In particular,
                # a messaging sidecar may start empty; never replace its older turns
                # with just this WebUI user/assistant pair, or append a recovered pair twice.
                canonical_messages = canonical_snapshot.get("messages") or []
                s.context_messages = list(canonical_messages)
                from api.gateway_sessions import display_history
                s.messages = display_history(canonical_snapshot)
                info = canonical_snapshot.get("info") or {}
                persisted_model = info.get("model") or s.model
                persisted_model_provider = info.get("provider") or s.model_provider
            s.active_stream_id = None
            s.gateway_run = None
            s.pending_user_message = None
            s.pending_attachments = None
            s.pending_started_at = None
            s.pending_user_source = None
            s.workspace = str(workspace)
            s.model = persisted_model if persisted_model is not None else model
            s.model_provider = (
                persisted_model_provider
                if persisted_model_provider is not None
                else model_provider
            )

            def _restore_cancelled_success_writeback():
                if pending_source in ("process_wakeup", "delegation_wakeup"):
                    s.context_messages = previous_context
                    s.messages = previous_messages
                    s.process_wakeup_pause = dict(previous_process_wakeup_pause)
                elif previous_process_wakeup_pause:
                    s.process_wakeup_pause = dict(previous_process_wakeup_pause)
                else:
                    clear_process_wakeup_pause(s, reason="run_completed")
                s.save()
                put_gateway_event("cancel", {"message": "Cancelled by user"})

            # Recheck immediately before clearing the pause; Stop can arrive
            # while the success transcript is being assembled.
            if cancel_event.is_set():
                _restore_cancelled_success_writeback()
                return
            clear_process_wakeup_pause(s, reason="run_completed")
            if cancel_event.is_set():
                _restore_cancelled_success_writeback()
                return
            s.save()
            if cancel_event.is_set():
                _restore_cancelled_success_writeback()
                return
            # #6366 re-gate: record the durable same-stream completion
            # event in the crash-safe turn journal. The run journal's
            # terminal state is only reached on its own ``stream_end``
            # write path, so a Gateway run whose terminal write is lost
            # would otherwise leave no completion evidence at all and
            # stale-cancel recovery would re-append a duplicate
            # recovered row after the valid final answer.
            try:
                append_turn_journal_event_for_stream(
                    session_id,
                    stream_id,
                    {"event": "completed", "created_at": time.time()},
                )
            except Exception:
                logger.debug("Failed to append completed turn journal event", exc_info=True)
            success_writeback_committed = True
        try:
            from api.goals import (
                evaluate_goal_after_turn,
                has_active_goal,
                register_pending_goal_continuation,
            )
            from api.profiles import get_hermes_home_for_profile

            profile_home = get_hermes_home_for_profile(getattr(s, "profile", None))
            if goal_related and has_active_goal(session_id, profile_home=profile_home):
                put_gateway_event("goal", {
                    "session_id": session_id,
                    "state": "evaluating",
                    "message": "Evaluating goal progress…",
                    "message_key": "goal_evaluating_progress",
                })
                decision = evaluate_goal_after_turn(
                    session_id,
                    assistant_text,
                    user_initiated=True,
                    profile_home=profile_home,
                ) or {}
                goal_message = str(decision.get("message") or "").strip()
                if goal_message:
                    put_gateway_event("goal", {
                        "session_id": session_id,
                        "state": "continuing" if decision.get("should_continue") else "idle",
                        "message": goal_message,
                        "message_key": decision.get("message_key") or (
                            "goal_continuing" if goal_message else ""
                        ),
                        "message_args": decision.get("message_args") or [],
                        "decision": decision,
                    })
                if decision.get("should_continue"):
                    continuation_prompt = str(decision.get("continuation_prompt") or "").strip()
                    continuation_id = None
                    if continuation_prompt:
                        # #1932 + #6885 + #7855: one record (marker + prompt +
                        # continuation ID). The SSE event fires only when the
                        # record landed so the frontend queue and the server
                        # record cannot disagree. The ID — not the prompt text —
                        # is what admits the turn later, so the browser can
                        # edit or combine the queued entry freely.
                        continuation_id = register_pending_goal_continuation(session_id, continuation_prompt)
                        if not continuation_id:
                            continuation_prompt = ""
                    if continuation_prompt:
                        put_gateway_event("goal_continue", {
                            "session_id": session_id,
                            "continuation_prompt": continuation_prompt,
                            "continuation_id": continuation_id,
                            "text": continuation_prompt,
                            "message": goal_message,
                            "message_key": decision.get("message_key") or "goal_continuing",
                            "message_args": decision.get("message_args") or [],
                            "decision": decision,
                        })
        except Exception as goal_exc:
            logger.debug(
                "Gateway goal continuation hook failed for session %s: %s",
                session_id,
                goal_exc,
            )
        from api.streaming import _session_payload_with_full_messages
        gateway_session_payload = _session_payload_with_full_messages(s, tool_calls=[])
        from api.helpers import public_session_projection
        put_gateway_event("done", {"session": public_session_projection(gateway_session_payload), "usage": usage})
        put_gateway_event("stream_end", {"session_id": session_id})
    except urllib.error.HTTPError as exc:
        error_payload = _settle_gateway_terminal_error(
            session_id, stream_id, workspace, model, model_provider,
            _gateway_http_error_event(exc, api_key_configured=bool(api_key)),
            persisted_model=persisted_model,
            persisted_model_provider=persisted_model_provider,
        )
        if error_payload is not None:
            put_gateway_event("apperror", error_payload)
    except Exception as exc:
        safe = _redact_text(str(exc))[:500]
        put_gateway_event("apperror", {
            "label": "Gateway request failed",
            "type": "gateway_error",
            "message": safe or "Gateway request failed.",
            "hint": "Check HERMES_WEBUI_GATEWAY_BASE_URL and Gateway API server health.",
        })
    finally:
        try:
            settle_gateway_stream_approvals(
                session_id,
                stream_id,
                reason="Gateway run ended during teardown before approval resolution",
            )
        except Exception:
            logger.debug("Failed to settle gateway pending approvals during teardown", exc_info=True)
        if s is not None and not preserve_gateway_pending:
            try:
                with _get_session_agent_lock(session_id):
                    _clear_gateway_pending_state(get_session(session_id), stream_id)
            except Exception:
                logger.debug("Failed to clear gateway stream state", exc_info=True)
            _cleanup_gateway_pending_mirror(session_id)
        with STREAMS_LOCK:
            AGENT_INSTANCES.pop(stream_id, None)
            CANCEL_FLAGS.pop(stream_id, None)
            STREAM_GOAL_RELATED.pop(stream_id, None)
            LiveTurnSnapshot.release(stream_id)
            STREAM_LAST_EVENT_ID.pop(stream_id, None)
            STREAMS.pop(stream_id, None)
        # Shared, no-op-safe release (#7302 re-gate): the chat/start orphan
        # recovery retires a dead stream's Gateway rows through this same path,
        # so both teardowns stay one implementation. ``runs_api_pending_marked``
        # keeps this call site's original finish-pending semantics.
        release_gateway_stream_state(stream_id, finish_pending=runs_api_pending_marked)
        _settle_delegation_reservation(session_id, stream_id, accepted=False)  # no-op once committed
        unregister_stream_owner(stream_id)
        unregister_active_run(stream_id)
        # Release the writeback-owner entry the route layer registered for this
        # Gateway run so SESSION_WRITEBACK_OWNERS does not grow unbounded across
        # the process lifetime (compare-and-clear: only clears if still owned by
        # this stream, mirroring the local streaming teardown).
        clear_session_writeback_owner_if_owned(session_id, stream_id)
