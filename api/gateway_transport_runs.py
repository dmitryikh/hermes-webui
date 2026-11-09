"""Runs API transport of the Hermes Gateway bridge (``/v1/runs``).

A durable transport: the turn is admitted as a Gateway run (idempotency key
``webui-<stream_id>``), its id is recorded before and after admission so a
restarted WebUI can reattach, and the run is followed over
``GET /v1/runs/{id}/events`` with one recovery procedure
(``_GatewayRunFollower``) arbitrated by the durable ``GET /v1/runs/{id}``
status. Decoding and projection are the shared layers
(``api.gateway_events`` / ``api.gateway_projection``); approvals, run-id
lifecycle and turn settlement are shared services of ``api.gateway_chat``.
"""
from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from api import gateway_chat, gateway_sse
from api.config import update_active_run
from api.gateway_events import (
    ApprovalRequest,
    ReplayTruncated,
    RunCancelled,
    RunCompleted,
    RunFailed,
    UsageReport,
    decode_runs_frame,
    runs_usage,
)
from api.gateway_projection import GatewayEventProjector, LiveTurnSnapshot, TurnCheckpoint
from api.gateway_sse import iter_lines_cancellable, iter_sse_frames
from api.gateway_transport import GatewayTransport, GatewayTurn, TurnResult

logger = logging.getLogger(__name__)


class RunsApiTransport(GatewayTransport):
    """Run the turn as a durable Gateway run, or reattach the run a previous WebUI started."""

    name = "runs"
    durable = True

    def run(self, turn: GatewayTurn) -> TurnResult:
        if turn.reattach_run:
            run_id = str(turn.reattach_run.get("run_id") or "").strip()
            if not run_id:
                # Crashed between admission and saving the id: the same key returns the original run.
                run_id = _admit_gateway_run(
                    f"{turn.base_url.rstrip('/')}/v1/runs", _gateway_run_headers(turn.session_id, turn.api_key),
                    turn.reattach_run["request"], turn.stream_id,
                )
                turn.record_run(run_id)
            gateway_chat._settle_delegation_reservation(turn.session_id, turn.stream_id, accepted=True)
            text, usage = _await_gateway_run_result(
                turn.session_id, turn.stream_id, run_id, turn.base_url, turn.api_key,
                put_gateway_event=turn.emit, cancel_event=turn.cancel_event,
            )
        else:
            text, usage = _run_gateway_runs_api_streaming(
                turn.session_id, turn.msg_text, turn.model, turn.workspace, turn.stream_id,
                turn.base_url, turn.api_key, turn.prefill_messages, turn.request_options,
                put_gateway_event=turn.emit,
                cancel_event=turn.cancel_event,
                attachments=turn.attachments,
                cfg=turn.cfg,
                session=turn.session,
                active_provider=(turn.model_provider or ""),
                on_run_id=turn.record_run,
            )
        return TurnResult(text=text, usage=usage)


def _relay_gateway_run_approval(session_id, run_id, payload, base_url, api_key, *, put_gateway_event) -> bool:
    """Auto-approve or surface one runs-API approval request as a WebUI approval card.

    Returns True when the payload was relayed (settled or surfaced as a
    card) and False when the translator rejected it — callers that register
    dedupe keys use that to avoid masking a later, well-formed replay of
    the same approval (round-4 maintainer minor)."""
    approval_data = gateway_chat._gateway_runs_approval_event(payload)
    if not approval_data:
        return False
    approval_data["run_id"] = run_id
    approval_data["_gateway_agent_identity_v1"] = gateway_chat._gateway_approval_identity_v1(approval_data)
    auto_approved, head, total = gateway_chat._settle_gateway_run_approval(
        session_id,
        approval_data,
        base_url,
        api_key,
    )
    if not auto_approved:
        put_gateway_event("approval", {**(head or approval_data), "pending_count": total})
    return True


def _open_gateway_run_events(base_url, headers, run_id, last_seq: int = -1, read_timeout_secs: float | None = None):
    headers_sse = {**headers, "Accept": "text/event-stream"}
    if last_seq >= 0:
        headers_sse["Last-Event-ID"] = str(last_seq)
    # Round-4 review (Greptile item 3): the wall-clock stall check only runs
    # when a line ARRIVES, so a socket that goes byte-SILENT blocks inside the
    # read for the full configured read timeout — 5x past the watchdog budget
    # — before the stall check or Stop handling can run. The streaming caller
    # bounds this read by the watchdog budget (+ small epsilon); the reattach
    # poller passes nothing and keeps the configured timeout.
    timeout = (
        min(gateway_sse.read_timeout_secs(), read_timeout_secs)
        if read_timeout_secs
        else gateway_sse.read_timeout_secs()
    )
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/runs/{urllib.parse.quote(run_id, safe='')}/events",
        headers=headers_sse,
        method="GET",
    )
    return urllib.request.urlopen(req, timeout=timeout)


def _relay_gateway_run_events(
    resp, session_id, stream_id, run_id, base_url, api_key,
    *, put_gateway_event, cancel_event, checkpoint=None, on_seq=None, final_text="",
    surfaced_approval_ids=None, output_is_authoritative=False, approval_is_current=None,
    watchdog_secs=None,
):
    """Relay one /v1/runs/{id}/events connection.

    Reads SSE frames (``api.gateway_sse``), decodes them into Gateway events
    (``api.gateway_events``) and dispatches each event: stream content goes to
    the ``GatewayEventProjector`` (WebUI event + reconnect snapshot), control
    events (approvals, terminal run events, replay truncation) are handled
    here because their side effects are run orchestration.

    Returns ``(text or None if cancelled, usage, outcome, accepted_frame)``,
    where ``accepted_frame`` is True when THIS connection accepted and
    relayed any real frame (delta, reasoning, tool, approval — some accepted
    frames carry no seq, so a cursor compare alone misses them); the
    streaming caller uses it to reset the clean-EOF reconnect backoff
    (round-7 review: cumulative text cannot drive the reset, because the
    relay is seeded from the turn-wide buffer).

    outcome is "ended", "eof" or "stalled" (only with ``watchdog_secs``). A
    ``replay.truncated`` frame is logged and the retained events keep relaying:
    the final answer comes from ``run.completed``/durable status, not from the
    streamed text. Relayed payloads carry ``gateway_seq`` so the WebUI run
    journal records the gateway cursor; each frame's seq is committed to
    ``checkpoint`` as soon as its events are relayed (then ``on_seq`` observes
    it), so a reconnect from the checkpoint never re-emits it.
    ``checkpoint`` (a ``TurnCheckpoint``) is the turn's recovery state and is
    updated in place, so nothing is lost when the connection raises; without
    one, a fresh checkpoint is seeded from ``final_text``.
    ``surfaced_approval_ids`` is shared with the reattach status probe so one approval surfaces once.
    ``output_is_authoritative`` lets ``run.completed.output`` replace streamed text (reattach: the
    Agent may transform its answer after streaming). ``approval_is_current`` drops replayed approvals
    the Gateway no longer has pending. ``watchdog_secs`` (streaming watchdog only) treats a
    connection that delivers nothing but comment/keepalive frames for that long as stalled and
    returns so the caller can consult the durable run status: keepalives are liveness, not
    progress (this exact case pinned a single-connection loop forever — #7978). Once
    ``run.completed`` is seen the outcome latches to "ended": the relay breaks
    out on the frame itself (trailing keepalives/[DONE] are post-terminal
    noise), and as a backstop any keepalive-only interval that slips past the
    latch finishes the turn as "ended" instead of probing (round-4 review: the
    frame is terminal; Greptile round-3: the latch must leave an exit).
    """
    usage: dict = {}
    outcome = "eof"
    # Cursor of the frame being dispatched; committed once its events are relayed.
    seq = None
    # True when this connection accepted and relayed any real frame (delta,
    # reasoning, tool, approval). Round-7 review: the streaming caller's
    # clean-EOF backoff reset must key on per-connection progress, not on
    # the cumulative turn text (which seeds every reconnect non-empty).
    accepted_frame = False
    # Keepalives are liveness, not progress: a stream that emits nothing but
    # comment frames past watchdog_secs is treated as stalled even though the
    # socket read never blocks past its timeout.
    last_progress = time.monotonic() if watchdog_secs is not None else None
    # Round-4 review (Greptile item 2): run.completed is terminal. Once the
    # frame is seen the outcome latches to "ended" for this connection — a
    # subsequent keepalive-only interval must not flip an already-terminal
    # lane back to "stalled" (which would force a status probe and could FAIL
    # an already-completed turn when the probe is unavailable).
    terminal_frame_seen = False

    def emit(event_name, data):
        if seq is not None and isinstance(data, dict):
            data = {**data, "gateway_seq": seq}
        put_gateway_event(event_name, data)

    def commit():
        nonlocal seq
        if seq is not None:
            checkpoint.commit(seq)
            if on_seq is not None:
                on_seq(seq)
        seq = None

    if checkpoint is None:
        checkpoint = TurnCheckpoint(stream_id, text=final_text)
    snapshot = checkpoint.snapshot
    projector = GatewayEventProjector(
        snapshot,
        emit,
        on_tool_activity=lambda name: update_active_run(stream_id, phase="gateway-tool", latest_tool=name),
    )

    def settle_terminal(payload, reason):
        # Same order on every terminal lane: replay steer text the Gateway
        # accepted but the run never consumed, then retire parked approvals.
        _relay_gateway_pending_steer(session_id, payload, put_gateway_event)
        from api.route_approvals import settle_gateway_pending_run
        settle_gateway_pending_run(session_id, run_id, reason=reason)

    for sse_frame in iter_sse_frames(iter_lines_cancellable(resp, cancel_event)):
        # Stop wins over stall classification: a Stop landing during a silent
        # read (surfaced by the read timeout as a blank line) cancels without
        # burning a status-probe round trip first.
        if cancel_event.is_set():
            put_gateway_event("cancel", {"message": "Cancelled by user"})
            return None, usage, "ended", False
        if not sse_frame.has_payload:
            # Blank / "event:" / "id:" / comment-keepalive frames carry no
            # payload. The wall-clock stall budget is checked HERE — after
            # classification (round-4 maintainer nit) — so protocol noise
            # still trips the watchdog (keepalives are liveness, not
            # progress) while a REAL data frame arriving after a silent gap
            # is processed as progress below instead of being discarded and
            # re-fetched via replay. Once run.completed has been seen the
            # outcome is latched (round-4 item 2): a keepalive interval past
            # the frame cannot flip the lane to "stalled" — post-terminal
            # stall means the stream refused to close after a terminal
            # frame, and the turn is already complete, so it finishes with
            # the relayed text instead of probing/failing (Greptile round-3:
            # the round-4 latch must not leave the loop without an exit).
            if watchdog_secs is not None and time.monotonic() - last_progress > watchdog_secs:
                if terminal_frame_seen:
                    outcome = "ended"
                    break
                logger.warning(
                    "Gateway events stream for run %s stalled "
                    "(no real events past %ss); probing run status",
                    run_id, watchdog_secs)
                outcome = "stalled"
                break
            continue
        if watchdog_secs is not None:
            # A real (non-comment) frame: this is stream progress.
            last_progress = time.monotonic()
        frame = decode_runs_frame(sse_frame)
        if frame is None:
            continue
        if frame.done:
            break
        seq = frame.seq
        if any(isinstance(event, ReplayTruncated) for event in frame.events):
            logger.info("Gateway replay for run %s truncated; relaying retained events", run_id)
            continue
        # Any parsed, non-replay-control frame counts as per-connection
        # progress for the clean-EOF backoff (round-7 review): delta,
        # reasoning, tool, approval and terminal frames all qualify — some
        # carry no seq, so this is deliberately not a cursor compare.
        accepted_frame = True
        for event in frame.events:
            if isinstance(event, ApprovalRequest):
                approval_key = gateway_chat._gateway_approval_key(event.payload)
                already = surfaced_approval_ids is not None and approval_key in surfaced_approval_ids
                if not already and (approval_is_current is None or approval_is_current(event.payload)):
                    if surfaced_approval_ids is not None and approval_key:
                        surfaced_approval_ids.add(approval_key)
                    _relay_gateway_run_approval(
                        session_id, run_id, event.payload, base_url, api_key,
                        put_gateway_event=emit,
                    )
            elif isinstance(event, RunCompleted):
                settle_terminal(event.payload, "Gateway run completed before approval resolution")
                if event.error:
                    raise RuntimeError(event.error)
                if event.output and (output_is_authoritative or not snapshot.text):
                    snapshot.adopt_text(event.output)
                usage.update(event.usage)
                outcome = "ended"
                terminal_frame_seen = True
            elif isinstance(event, RunFailed):
                settle_terminal(event.payload, "Gateway run failed before approval resolution")
                raise RuntimeError(event.error)
            elif isinstance(event, RunCancelled):
                settle_terminal(event.payload, "Gateway run was cancelled before approval resolution")
                put_gateway_event("cancel", {"message": "Cancelled by gateway"})
                return None, usage, "ended", False
            elif isinstance(event, UsageReport):
                usage.update(event.usage)
            else:
                projector.project(event)
        commit()
        if terminal_frame_seen:
            # A terminal frame definitionally ends the run: break instead of
            # reading on (Greptile round-3). Trailing frames on a healthy
            # gateway — including [DONE] — are post-terminal noise, and
            # `with resp:` closes the socket on the way out. Continuing here
            # stranded the completed turn inside the relay: the round-4 latch
            # disables the stall check, so endless keepalives pinned the loop
            # forever. (The watchdog branch above keeps an exit as a backstop
            # for any future path that re-enters the read loop latched.)
            break
    commit()
    return snapshot.text, usage, outcome, accepted_frame


def _admit_gateway_run(url_runs, headers, run_body, stream_id) -> str:
    """POST /v1/runs; the same stream and body return the originally admitted run id."""
    req = urllib.request.Request(
        url_runs,
        data=json.dumps(run_body).encode("utf-8"),
        # Durable run record on the gateway: GET /v1/runs/{id} survives either side restarting.
        headers={**headers, "Idempotency-Key": f"webui-{stream_id}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        run_data = json.loads(resp.read(65536))
    run_id = str(run_data.get("run_id") or "").strip()
    if not run_id:
        raise ValueError(f"Gateway runs API returned no run_id: {run_data!r}")
    return run_id


def _gateway_run_headers(session_id, api_key) -> dict:
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "X-Hermes-Session-Id": session_id,
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
        headers["X-Hermes-Session-Key"] = f"webui:{session_id}"
    return headers


def _run_gateway_runs_api_streaming(
    session_id, msg_text, model, workspace, stream_id,
    base_url, api_key, prefill_messages, body_extras,
    *, put_gateway_event, cancel_event,
    attachments=None, cfg=None, session=None,
    active_provider: str = "",
    on_run_id=None,
    on_seq=None,
):
    """Submit via POST /v1/runs and relay SSE events including approval."""
    try:
        url_runs = f"{base_url.rstrip('/')}/v1/runs"
        headers = _gateway_run_headers(session_id, api_key)
        message_content: Any = str(msg_text or "")
        if attachments:
            try:
                from api.streaming import _build_native_multimodal_message

                message_content = _build_native_multimodal_message("", str(msg_text or ""), attachments, str(workspace), cfg=cfg, active_provider=active_provider, active_model=(model or ""), requested_provider=active_provider, profile=getattr(session, "profile", None))
            except Exception:
                logger.debug("Failed to build runs-API multimodal attachment payload", exc_info=True)
                message_content = str(msg_text or "")
        from api.streaming import (
            _active_turn_authority,
            _active_turn_token_matches,
            _is_non_replayable_history_row,
            _is_reasoning_only_assistant_message,
            _recovered_user_row_is_kept,
            _strip_oob_blocks,
        )

        instructions_parts = []
        conversation_history = []
        # (role, content, recovered) for each session row that may be sent.
        history_rows = []
        # The active turn's retained user row (regeneration restamps it with
        # this stream's token at chat-start) is re-sent as `input` below;
        # sending it in conversation_history too makes the gateway see the
        # prompt twice (re-gate 2026-10-05 item 3). The reconciled partial of
        # that same interrupted turn is dropped with it.
        active_turn_identity = _active_turn_authority(session, stream_id, msg_text)
        drop_reconciled_partial = False
        # Set when the outbound history is EMPTY ON PURPOSE and the webui
        # KNOWS it: either the token match excluded the retained active-turn
        # user row (it rides as `input` alone), context_messages had rows
        # that were all filtered away, or the session was deliberately
        # emptied (the 0.0 truncate-to-empty watermark stamped; a
        # deferred-mode turn commits the new prompt only after the run,
        # while eager mode checkpoints it into messages without advancing
        # that watermark). An
        # absent (or empty) key makes the Gateway admission handler fall back
        # to its own stored transcript, which reloads turns the user just
        # deleted (re-gate 2026-10-06 item 2; #8166 Greptile 2026-10-08:
        # /clear leaves the gateway store intact, so its pre-clear transcript
        # resurrected wholesale).
        active_row_excluded = False
        for entry in getattr(session, "context_messages", None) or []:
            if not isinstance(entry, dict):
                continue
            if _active_turn_token_matches(entry, active_turn_identity):
                drop_reconciled_partial = True
                active_row_excluded = True
                continue
            if (
                drop_reconciled_partial
                and entry.get("role") == "assistant"
                and entry.get(gateway_chat._GATEWAY_CANCEL_RECONCILED_STREAM_KEY)
            ):
                continue
            drop_reconciled_partial = False
            # The same rows the legacy path drops: error markers, empty partials
            # and reasoning-only assistant rows.
            if _is_non_replayable_history_row(entry) or _is_reasoning_only_assistant_message(entry):
                continue
            role = str(entry.get("role") or "").strip().lower()
            if role not in {"user", "assistant"}:
                continue
            content = entry.get("content")
            if content is not None:
                content = _strip_oob_blocks(content)
                history_rows.append((role, content, role == "user" and bool(entry.get("_recovered"))))
        # A _recovered user row is sent only where it opens an answered turn
        # (after an assistant turn or as the first row sent), as the legacy path decides it.
        for index, (role, content, recovered) in enumerate(history_rows):
            if recovered:
                prev_role = conversation_history[-1]["role"] if conversation_history else None
                next_role = history_rows[index + 1][0] if index + 1 < len(history_rows) else None
                if not _recovered_user_row_is_kept(prev_role, next_role):
                    continue
            conversation_history.append({"role": role, "content": content})
        for entry in prefill_messages or []:
            if not isinstance(entry, dict):
                continue
            role = str(entry.get("role") or "").strip().lower()
            content = entry.get("content")
            if role == "system":
                if isinstance(content, str) and content.strip():
                    instructions_parts.append(content)
                elif content is not None:
                    instructions_parts.append(str(content))
                continue
            if role not in {"user", "assistant"}:
                continue
            if content is not None:
                content = _strip_oob_blocks(content)
            conversation_history.append({"role": role, "content": content})
        run_input = message_content
        if isinstance(run_input, list):
            run_input = [{"role": "user", "content": run_input}]
        run_body = {
            "model": gateway_chat._gateway_model_field(model) or "default",
            "input": run_input,
            **body_extras,
            "session_id": session_id,
        }
        if instructions_parts:
            run_body["instructions"] = "\n\n".join(part for part in instructions_parts if part)
        if conversation_history:
            run_body["conversation_history"] = conversation_history
        elif (
            active_row_excluded
            or bool(getattr(session, "context_messages", None))
            # 0.0 is the truncate-to-empty sentinel every deliberate
            # emptying stamps (/clear, retry, undo, truncate keep=0 via
            # session_ops._truncation_watermark_for([])); the eager-mode
            # chat-start checkpoint appends the NEW prompt to messages but
            # never advances a 0.0 watermark, so this holds in both save
            # modes. A brand-new or imported session has no watermark.
            or getattr(session, "truncation_watermark", None) == 0.0
        ):
            # The outbound history is empty ON PURPOSE and the webui KNOWS
            # it: the retained active-turn row was excluded and nothing else
            # survived the filters (regenerate), context_messages HAD rows
            # that were all filtered away, or the session was deliberately
            # emptied (0.0 truncate-to-empty watermark — /clear, retry,
            # undo, truncate keep=0; api/routes.py /api/session/clear). Keep
            # the key non-empty with a sentinel so the admission handler
            # takes this body as-is instead of falling back to its stored
            # transcript and resurrecting deleted turns (re-gate 2026-10-06
            # item 2; #8166 Greptile 2026-10-08: /clear leaves the gateway
            # store intact, so its pre-clear transcript resurrected
            # wholesale).
            # The sentinel is a USER row: the Anthropic-native converter
            # keeps the LAST system row, so a system sentinel replaced the
            # cached prompt + WebUI instructions wholesale on every
            # deliberate-empty lane (re-gate 2026-10-13 item 2).
            # Any other empty-history shape OMITS the key so the admission
            # handler falls back to its own stored transcript exactly as on
            # master: a brand-new session's first deferred-mode turn
            # (messages empty, context empty, NO watermark) and an imported
            # CLI/legacy session (messages populated, context empty).
            # api/gateway_chat.py 73955fec had fired the sentinel for EVERY
            # empty outbound history, so imported sessions ran blind
            # (re-gate 2026-10-12 14:04 UTC).
            # The prompt travels in `input` only.
            run_body["conversation_history"] = [
                {"role": "user", "content": "No earlier conversation turns."}
            ]
        update_active_run(stream_id, phase="gateway-request")
        # Persist the exact body first: a restart before the run id is saved replays this admission.
        if on_run_id is not None:
            try:
                on_run_id("", request=run_body)
            except gateway_chat.GatewayHandoffCancelled:
                # Cancelled before the handoff was durable: never submit the run (#7293).
                from api.streaming import _server_settle_reason, _worker_cancel_event_payload

                server_settled = _server_settle_reason(stream_id) is not None
                put_gateway_event("cancel", _worker_cancel_event_payload(stream_id) if server_settled else {
                    "message": "Cancelled by user",
                })
                return None, {}
        run_id = _admit_gateway_run(url_runs, headers, run_body, stream_id)
    except Exception:
        gateway_chat._finish_gateway_run_starting(stream_id)
        raise

    gateway_chat._publish_gateway_run_id(stream_id, run_id)
    if on_run_id is not None:
        on_run_id(run_id)

    return _GatewayRunFollower(
        session_id, stream_id, run_id, base_url, api_key,
        put_gateway_event=put_gateway_event, cancel_event=cancel_event,
        checkpoint=TurnCheckpoint(stream_id, text=""), on_seq=on_seq,
    ).run()


def _relay_gateway_pending_steer(session_id, payload, put_gateway_event) -> None:
    """Replay steer text the Gateway accepted but the run never consumed, like _settle_pending_steer()."""
    text = payload.get("pending_steer") if isinstance(payload, dict) else None
    if isinstance(text, str) and text.strip():
        put_gateway_event("pending_steer_leftover", {"session_id": session_id, "text": text})


_GATEWAY_RUN_TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled", "interrupted"})


GATEWAY_REATTACH_POLL_INTERVAL = 2.0


# Consecutive unreachable polls tolerated before the reattached turn is failed (~5 min at 2s).
GATEWAY_REATTACH_MAX_POLL_FAILURES = 150


# Streaming watchdog budget (#7978): a connection delivering nothing but
# keepalive/comment frames for this long is treated as stalled and the durable
# run status is consulted; keepalives are liveness, not progress. The same
# budget (plus a small epsilon) also bounds the per-read wait on the streaming
# events connection (round-4 review): the wall-clock stall check only runs
# when a line ARRIVES, so a socket that goes byte-SILENT would otherwise block
# inside the read for the full 600s read timeout — 5x past this budget —
# before the stall check or Stop handling could run. Other readers (reattach
# poller) keep the configured read timeout.
_GATEWAY_WATCHDOG_SECS = 120.0


_GATEWAY_WATCHDOG_READ_EPSILON = 2.0


# Consecutive durable-status 404s tolerated inside the streaming watchdog
# before the turn fails closed: a 404 is terminal (round-2 review); the extra
# probe only covers a status-before-registration race, and the re-probe runs
# immediately with no poll-interval sleep.
_STATUS_404_GRACE_PROBES = 2


# Consecutive clean-EOF reconnects (no terminal frame, no transport error:
# the gateway or a proxy accepts the connection and closes it immediately)
# pace themselves before reopening /events. Backoff doubles from
# _CLEAN_EOF_BACKOFF_BASE_SECS up to _CLEAN_EOF_BACKOFF_MAX_SECS; any REAL
# progress (a data frame committed a new seq, or durable status reporting the
# run no longer streaming) resets it. Stop is honoured during every wait.
_CLEAN_EOF_BACKOFF_BASE_SECS = 0.5


_CLEAN_EOF_BACKOFF_MAX_SECS = 30.0


def _get_gateway_run_status(base_url: str, api_key: str, run_id: str) -> dict:
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/runs/{urllib.parse.quote(run_id, safe='')}",
        headers=headers,
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        payload = json.loads(resp.read(1024 * 1024) or b"{}")
    return payload if isinstance(payload, dict) else {}


def _await_gateway_run_result(
    session_id, stream_id, run_id, base_url, api_key,
    *, put_gateway_event, cancel_event,
):
    """Resume a run started by a previous WebUI process; same return contract as _run_gateway_runs_api_streaming.

    The turn resumes from the checkpoint the run journal describes, through the
    same follower (and recovery procedure) a live turn uses.
    """
    gateway_chat._publish_gateway_run_id(stream_id, run_id)
    update_active_run(stream_id, phase="gateway-reattached")
    return _GatewayRunFollower(
        session_id, stream_id, run_id, base_url, api_key,
        put_gateway_event=put_gateway_event, cancel_event=cancel_event,
        checkpoint=TurnCheckpoint.restore(session_id, stream_id), resumed=True,
    ).run()


class _GatewayRunFollower:
    """Follow one admitted Gateway run to its outcome; the single recovery procedure.

    Every connection loss — read timeout, transport error, HTTP error on
    ``/events``, clean EOF, watchdog stall (#7978) and a WebUI restart —
    goes through the same ``recover()`` step:

      1. probe the durable run status, the only success arbiter;
      2. surface an approval the run is parked on (the events replay may
         have lost it or never re-send it);
      3. terminal  -> settle the turn from the status (output, else the
                      checkpoint's streamed text) and stop;
         running   -> pace, then reconnect ``/events`` from the
                      ``TurnCheckpoint`` cursor and keep projecting onto its
                      snapshot.

    The checkpoint is the only resume state: a live turn starts from a fresh
    one; a turn that outlived the WebUI process starts from
    ``TurnCheckpoint.restore`` (the run journal) and begins with
    ``recover()``, because approvals relayed before the restart sit behind
    the replay cursor. A checkpoint without a trustworthy cursor follows the
    run by status only.

    ``resumed`` marks a follower whose in-memory approval state was lost with
    the previous process: status approvals are surfaced without an exact
    identity, and replayed approvals are checked against the Gateway before
    they surface (an auto-approved one leaves no journal row).
    """

    def __init__(self, session_id, stream_id, run_id, base_url, api_key, *,
                 put_gateway_event, cancel_event, checkpoint, resumed=False, on_seq=None):
        self.session_id, self.stream_id, self.run_id = session_id, stream_id, run_id
        self.base_url, self.api_key = base_url, api_key
        self.put_gateway_event, self.cancel_event = put_gateway_event, cancel_event
        self.checkpoint, self.resumed, self.on_seq = checkpoint, resumed, on_seq
        self.headers = _gateway_run_headers(session_id, api_key)
        self.usage: dict = {}
        # One approval surfaces once across reconnects, status probes and both
        # sources (events replay and status).
        self.surfaced_approval_ids: set[str] = set()
        self.probe_failures = 0
        self.status_404_streak = 0
        self.clean_eof_reconnects = 0
        # The relay's result for an "ended" connection: the final text, or None when cancelled.
        self.ended_text = None

    def run(self):
        """Return ``(final text, usage)``, or ``(None, usage)`` when the turn was cancelled."""
        outcome = "resume" if self.resumed else None
        while True:
            if self.cancel_event.is_set():
                self.put_gateway_event("cancel", {"message": "Cancelled by user"})
                return None, self.usage
            if outcome is not None:
                settled = self.recover(outcome)
                if settled is not None:
                    return settled
                if self.cancel_event.is_set():
                    continue
            outcome = self.stream() if self.checkpoint.replayable else "poll"
            if outcome == "ended":
                return self.ended_text, self.usage

    # ---- one /events connection ----------------------------------------------

    def stream(self) -> str:
        """Relay one events connection from the checkpoint; return how it ended.

        ``"ended"`` (terminal frame relayed, or cancelled), ``"eof"``,
        ``"stalled"`` or ``"unreachable"`` (connect/read failure).
        """
        try:
            resp = _open_gateway_run_events(
                self.base_url, self.headers, self.run_id, self.checkpoint.cursor,
                # Bound each read by the watchdog budget so a byte-silent socket
                # surfaces within ~budget and Stop/stall handling can run.
                read_timeout_secs=_GATEWAY_WATCHDOG_SECS + _GATEWAY_WATCHDOG_READ_EPSILON,
            )
        except (urllib.error.URLError, OSError) as exc:
            logger.warning("Gateway events stream for run %s unavailable (%s); probing durable run status",
                           self.run_id, getattr(exc, "code", exc.__class__.__name__))
            return "unreachable"
        try:
            with resp:
                text, usage, outcome, accepted_frame = _relay_gateway_run_events(
                    resp, self.session_id, self.stream_id, self.run_id, self.base_url, self.api_key,
                    put_gateway_event=self.put_gateway_event, cancel_event=self.cancel_event,
                    checkpoint=self.checkpoint, on_seq=self.on_seq,
                    # The Agent may transform its answer after streaming.
                    output_is_authoritative=True,
                    surfaced_approval_ids=self.surfaced_approval_ids,
                    approval_is_current=self.approval_is_current if self.resumed else None,
                    watchdog_secs=_GATEWAY_WATCHDOG_SECS,
                )
        except (urllib.error.URLError, OSError):
            # Read timeout / reset mid-stream. Nothing is lost: the relay
            # updated the checkpoint frame by frame.
            return "unreachable"
        self.usage.update({k: v for k, v in usage.items() if v})
        # Per-connection progress resets the clean-EOF backoff (never the
        # cumulative turn text, which seeds every reconnect). A stall is a
        # connection that stayed open for the whole budget, not a fast close.
        if accepted_frame or outcome == "stalled":
            self.clean_eof_reconnects = 0
        self.ended_text = text
        return outcome

    def approval_is_current(self, payload) -> bool:
        try:
            status = _get_gateway_run_status(self.base_url, self.api_key, self.run_id)
        except (urllib.error.URLError, OSError, ValueError):
            return True  # fail open: a stale card 409s, a missing one stalls the run
        approval = gateway_chat._gateway_status_approval(status)
        return (
            str(status.get("status") or "").strip().lower() == "waiting_for_approval"
            and isinstance(approval, dict)
            and gateway_chat._gateway_approval_key(approval) == gateway_chat._gateway_approval_key(payload)
        )

    # ---- recovery ------------------------------------------------------------

    def recover(self, outcome: str):
        """Arbitrate a lost connection by durable status.

        Returns the turn result when the run is terminal, else ``None`` after
        pacing (the caller reconnects from the checkpoint).
        """
        status = self.probe()
        if status is None:
            return None  # probe failed and was paced; reconnect and probe again
        self.surface_status_approval(status)
        state = str(status.get("status") or "").strip().lower()
        if state in _GATEWAY_RUN_TERMINAL_STATUSES:
            return self.settle(status, state)
        self.pace(outcome)
        return None

    def probe(self) -> dict | None:
        """Read the durable run status.

        Raises when the run cannot be followed: rejected credentials, a
        repeated 404 (the Gateway has no record of the run; one immediate
        re-probe covers a status-before-registration race) or an exhausted
        transport-failure budget. Returns None after pacing a transient
        failure.
        """
        for _attempt in range(_STATUS_404_GRACE_PROBES):
            try:
                status = _get_gateway_run_status(self.base_url, self.api_key, self.run_id)
            except urllib.error.HTTPError as exc:
                if exc.code in (401, 403):
                    raise RuntimeError(
                        f"Gateway rejected the WebUI credentials (HTTP {exc.code}) while following run {self.run_id}"
                    ) from exc
                if exc.code == 404:
                    self.status_404_streak += 1
                    if self.status_404_streak >= _STATUS_404_GRACE_PROBES:
                        raise RuntimeError(
                            "Gateway no longer has the run; its result could not be recovered"
                        ) from exc
                    if self.cancel_event.is_set():
                        return None
                    continue
                self._probe_failed(exc)
                return None
            except (urllib.error.URLError, OSError, ValueError) as exc:
                self._probe_failed(exc)
                return None
            self.probe_failures = 0
            self.status_404_streak = 0
            return status
        return None

    def _probe_failed(self, exc) -> None:
        # A probe failure is not proof the run is gone: retry within budget.
        self.probe_failures += 1
        self.status_404_streak = 0
        if self.probe_failures >= GATEWAY_REATTACH_MAX_POLL_FAILURES:
            raise RuntimeError("Gateway became unreachable while waiting for the run to finish") from exc
        self.cancel_event.wait(GATEWAY_REATTACH_POLL_INTERVAL)

    def surface_status_approval(self, status: dict) -> None:
        """Surface the approval a ``waiting_for_approval`` status is parked on, once.

        The status carries only one pending approval. While the events replay
        can still deliver the queue in order, surfacing it ahead of the replay
        would invert FIFO, so a live follower does so only for a prompt fenced
        by its execution generation. A resumed follower always does: the previous process's
        cards are gone and the replay may never re-send them.
        """
        approval = gateway_chat._gateway_status_approval(status)
        if str(status.get("status") or "").strip().lower() != "waiting_for_approval" or not isinstance(approval, dict):
            return
        if not self.resumed and type(approval.get("execution_generation")) is not int:
            return
        approval_key = gateway_chat._gateway_approval_key(approval)
        if approval_key and approval_key not in self.surfaced_approval_ids:
            relayed = _relay_gateway_run_approval(
                self.session_id, self.run_id, approval, self.base_url, self.api_key,
                put_gateway_event=self.put_gateway_event,
            )
            # Register only after a successful relay: a rejected payload must
            # not suppress a later, well-formed replay of the same approval.
            if relayed:
                self.surfaced_approval_ids.add(approval_key)

    def settle(self, status: dict, state: str):
        """Settle the turn from a terminal durable status."""
        from api.route_approvals import settle_gateway_pending_run

        # Same order as the events relay: replay steer text the Gateway accepted
        # but the run never consumed, then retire parked approvals.
        _relay_gateway_pending_steer(self.session_id, status, self.put_gateway_event)
        settle_gateway_pending_run(
            self.session_id, self.run_id, reason=f"Gateway run {state} before approval resolution",
        )
        if state in ("cancelled", "interrupted"):
            # Persist before emitting: the browser refetches the session on this event.
            gateway_chat._settle_gateway_cancelled_turn(self.session_id, self.stream_id)
            self.put_gateway_event("cancel", {"message": "Cancelled by gateway"})
            return None, self.usage
        if state != "completed":
            raise RuntimeError(str(status.get("error") or f"Gateway run {state}"))
        # Final answer: the durable output when non-empty, else what streamed.
        output = str(status.get("output") or "") or self.checkpoint.text or LiveTurnSnapshot.partial_text(self.stream_id)
        if output:
            self.checkpoint.snapshot.adopt_text(output)
        self.usage.update(runs_usage(status))
        return self.checkpoint.text, self.usage

    def pace(self, outcome: str) -> None:
        """Wait before reconnecting after a non-terminal arbitration (Stop interrupts every wait)."""
        if outcome == "resume":
            return  # first recovery of a resumed run: stream right away
        if outcome in ("unreachable", "poll"):
            self.cancel_event.wait(GATEWAY_REATTACH_POLL_INTERVAL)
            return
        # A clean EOF (or stall) reconnects with an exponential backoff, so a
        # gateway or proxy that closes every connection immediately cannot
        # hot-loop this thread; progress on a connection resets it. The
        # exponent is bounded before multiplying (no float overflow).
        self.clean_eof_reconnects += 1
        self.cancel_event.wait(min(
            _CLEAN_EOF_BACKOFF_BASE_SECS * (2 ** min(self.clean_eof_reconnects - 1, 6)),
            _CLEAN_EOF_BACKOFF_MAX_SECS,
        ))
