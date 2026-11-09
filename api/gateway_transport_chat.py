"""Chat Completions transport of the Hermes Gateway bridge (``/v1/chat/completions``).

A stateless OpenAI-compatible stream: one POST whose SSE response carries the
turn (``chat.completion.chunk`` objects plus the named Hermes events
``hermes.tool.progress`` and ``approval.request``). The Gateway publishes no
run identity for it, so the turn cannot be reattached after a WebUI restart,
stopped on the Gateway or steered; approvals are answered through the
completion id the approval payload carries.
"""
from __future__ import annotations

import json
import logging
import urllib.request
from typing import Any

from api import gateway_chat, gateway_sse
from api.config import gateway_approval_unavailable_reason, update_active_run
from api.gateway_events import ApprovalRequest, ChunkError, UsageReport, decode_chat_completion_frame
from api.gateway_projection import GatewayEventProjector, LiveTurnSnapshot
from api.gateway_sse import iter_lines_cancellable, iter_sse_frames
from api.gateway_transport import GatewayTransport, GatewayTurn, TurnResult

logger = logging.getLogger(__name__)


class ChatCompletionsTransport(GatewayTransport):
    """Stream the turn from ``POST /v1/chat/completions``."""

    name = "chat_completions"
    durable = False

    def run(self, turn: GatewayTurn) -> TurnResult:
        _notify_approval_unavailable(turn)
        req = urllib.request.Request(
            f"{turn.base_url}/v1/chat/completions",
            data=json.dumps(_request_body(turn)).encode("utf-8"),
            headers=_request_headers(turn.session_id, turn.api_key),
            method="POST",
        )
        usage: dict = {}
        error = None
        update_active_run(turn.stream_id, phase="gateway-request")
        with urllib.request.urlopen(req, timeout=gateway_sse.read_timeout_secs()) as resp:
            gateway_chat._settle_delegation_reservation(turn.session_id, turn.stream_id, accepted=True)
            projector = GatewayEventProjector(
                LiveTurnSnapshot(turn.stream_id, text=""),
                turn.emit,
                on_tool_activity=lambda name: update_active_run(
                    turn.stream_id, phase="gateway-tool", latest_tool=name,
                ),
            )
            for sse_frame in iter_sse_frames(iter_lines_cancellable(resp, turn.cancel_event)):
                if turn.cancel_event.is_set():
                    turn.emit("cancel", {"message": "Cancelled by user"})
                    return TurnResult(text=None, usage=usage)
                frame = decode_chat_completion_frame(sse_frame)
                if frame is None:
                    continue
                if frame.done:
                    break
                for event in frame.events:
                    if isinstance(event, ApprovalRequest):
                        _relay_chat_completion_approval(
                            turn.session_id, turn.stream_id, event.payload, turn.base_url, turn.api_key,
                            cancel_event=turn.cancel_event, put_gateway_event=turn.emit,
                        )
                    elif isinstance(event, ChunkError):
                        error = event.message
                    elif isinstance(event, UsageReport):
                        usage.update(event.usage)
                    else:
                        projector.project(event)
        return TurnResult(text=projector.text, usage=usage, error=error)


def _request_headers(session_id: str, api_key: str) -> dict:
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "X-Hermes-Session-Id": session_id,
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
        # Scope Gateway long-term continuity to this WebUI conversation
        # without exposing the browser's auth cookie or CSRF material.
        headers["X-Hermes-Session-Key"] = f"webui:{session_id}"
    return headers


def _request_body(turn: GatewayTurn) -> dict:
    message_content: Any = str(turn.msg_text or "")
    if turn.attachments:
        try:
            from api.streaming import _build_native_multimodal_message

            message_content = _build_native_multimodal_message(
                "", str(turn.msg_text or ""), turn.attachments, str(turn.workspace), cfg=turn.cfg,
                active_provider=(turn.model_provider or ""), active_model=(turn.model or ""),
                requested_provider=(turn.model_provider or ""), profile=getattr(turn.session, "profile", None),
            )
        except Exception:
            logger.debug("Failed to build gateway multimodal attachment payload", exc_info=True)
            message_content = str(turn.msg_text or "")
    return {
        "model": gateway_chat._gateway_model_field(turn.model) or "default",
        "stream": True,
        "messages": [*turn.prefill_messages, {"role": "user", "content": message_content}],
        **turn.request_options,
    }


def _notify_approval_unavailable(turn: GatewayTurn) -> None:
    """Warn once per session when this Gateway cannot relay approvals to WebUI."""
    s = turn.session
    approval_reason = gateway_approval_unavailable_reason(turn.base_url, turn.api_key)
    if approval_reason is None:
        return
    if not hasattr(s, "_approval_notice_emitted"):
        s._approval_notice_emitted = False
    if s._approval_notice_emitted:
        return
    approval_message = "Approvals require a newer gateway. Upgrade the connected Hermes gateway to enable this."
    approval_type = "approval_gateway_unsupported"
    if approval_reason == "unreachable":
        approval_type = "approval_gateway_offline"
        approval_message = "Gateway connection failed. Check that the connected Hermes gateway is running and reachable."
    turn.emit("warning", {"type": approval_type, "message": approval_message})
    s._approval_notice_emitted = True


def _relay_chat_completion_approval(
    session_id, stream_id, payload, base_url, api_key, *, cancel_event, put_gateway_event,
) -> bool:
    """Surface one ``/v1/chat/completions`` approval request as a mirrored WebUI approval card.

    Returns False (and surfaces nothing) for a payload the translator rejects.
    """
    approval_data = gateway_chat._gateway_runs_approval_event(payload)
    if not approval_data:
        logger.debug("Ignoring malformed gateway approval payload")
        return False
    # Record the gateway run_id so /api/approval/respond can relay the choice
    # back and resume the parked run (this path never creates a local run;
    # without this the card renders but approve/deny returns ok:false).
    # No-op when the payload omits run_id.
    approval_run_id = str(approval_data.get("run_id") or "").strip()
    if approval_run_id:
        gateway_chat._STREAM_RUN_IDS[stream_id] = approval_run_id
    # Same creation-time identity decision the Runs API relay makes (#8579):
    # the mirror must carry the marker /api/approval/respond reads, or a
    # generation-fenced gateway rejects the relayed answer with 400
    # invalid_params and the card never settles. Skipping it here silently
    # downgraded every default-stream card to the legacy body.
    approval_data["_gateway_agent_identity_v1"] = gateway_chat._gateway_approval_identity_v1(approval_data)
    try:
        head, total = gateway_chat._submit_stream_approval_mirror(session_id, approval_data, cancel_event)
        approval_data = {**(head or approval_data), "pending_count": total}
    except Exception:
        logger.debug("submit_gateway_pending_mirror failed", exc_info=True)
    put_gateway_event("approval", approval_data)
    return True
