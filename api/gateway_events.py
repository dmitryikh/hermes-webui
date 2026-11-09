"""Layer 2 of the Hermes Gateway bridge: Hermes event decoding.

Turns the payload of one ``SseFrame`` (layer 1, ``api.gateway_sse``) into a
``GatewayFrame``: the Gateway cursor (``seq``) plus a tuple of typed
``GatewayEvent`` values. Decoding is pure — no stream registries, journals,
sessions, network or WebUI event names. What an event *means* for the WebUI
is decided by ``api.gateway_projection`` (layer 3) and by the run
orchestration in ``api.gateway_chat``.

The Gateway speaks three wire dialects, one decoder each:

* ``decode_runs_frame`` — ``GET /v1/runs/{id}/events`` (Runs API): every
  payload is a ``_run_event`` envelope named by its ``event`` field.
* ``decode_chat_completion_frame`` — ``POST /v1/chat/completions`` streaming:
  ``chat.completion.chunk`` objects (text, ``reasoning_content``, finish
  usage/error) plus the named Hermes events ``hermes.tool.progress`` and
  ``approval.request``.
* ``decode_session_event`` — the canonical session protocol
  (``api.gateway_sessions``) whose frames arrive already parsed over JSON-RPC.

Only the shapes the current Hermes Agent Gateway emits are accepted; older
field names and event aliases are not decoded.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from api.gateway_sse import SseFrame
from api.run_journal import bound_run_journal_snapshot_args


# ---------------------------------------------------------------------------
# Intermediate representation
# ---------------------------------------------------------------------------

class GatewayEvent:
    """Base class of one decoded Hermes Gateway event."""

    __slots__ = ()


@dataclass(frozen=True)
class TextDelta(GatewayEvent):
    """A chunk of the open assistant answer segment."""

    text: str


@dataclass(frozen=True)
class ReasoningDelta(GatewayEvent):
    """A chunk of model reasoning for the open answer segment."""

    text: str


@dataclass(frozen=True)
class ToolProgress(GatewayEvent):
    """A tool call started (``completed=False``) or finished (``completed=True``)."""

    completed: bool
    name: str
    preview: Any = None
    args: dict = field(default_factory=dict)
    is_error: bool = False
    tid: str | None = None


@dataclass(frozen=True)
class InterimMessage(GatewayEvent):
    """Commentary the Agent sealed before continuing the turn.

    It closes the open answer segment: the text (and its reasoning) belongs to
    the sealed commentary row, and the final answer starts afresh.
    """

    text: str
    already_streamed: bool = False


@dataclass(frozen=True)
class ApprovalRequest(GatewayEvent):
    """A parked tool approval; ``payload`` is the raw Gateway approval payload."""

    payload: dict


@dataclass(frozen=True)
class UsageReport(GatewayEvent):
    """Token/cost accounting; only the non-zero fields are carried."""

    usage: dict


@dataclass(frozen=True)
class ChunkError(GatewayEvent):
    """An OpenAI-compatible chunk that carries a terminal ``error`` string."""

    message: str


@dataclass(frozen=True)
class RunCompleted(GatewayEvent):
    output: str
    usage: dict
    error: str
    payload: dict


@dataclass(frozen=True)
class RunFailed(GatewayEvent):
    error: str
    payload: dict


@dataclass(frozen=True)
class RunCancelled(GatewayEvent):
    payload: dict


@dataclass(frozen=True)
class ReplayTruncated(GatewayEvent):
    """The Gateway dropped retained events after the requested cursor."""


@dataclass(frozen=True)
class GatewayFrame:
    """One decoded wire frame: its Gateway cursor and the events it carries.

    A frame may carry no events (e.g. a tool-progress frame for an internal
    tool) and still advance the cursor. ``done`` marks the OpenAI-compatible
    ``[DONE]`` end-of-stream sentinel.
    """

    seq: int | None = None
    events: tuple[GatewayEvent, ...] = ()
    done: bool = False


DONE_FRAME = GatewayFrame(done=True)

TERMINAL_RUN_EVENTS = (RunCompleted, RunFailed, RunCancelled)


# ---------------------------------------------------------------------------
# Payload field extraction
#
# Each dialect is decoded exactly as Hermes Agent's API server emits it
# (gateway/platforms/api_server_runs.py, api_server_openai_routes.py,
# gateway/run_turn_progress.py); shapes no current Gateway sends are not
# accepted.
# ---------------------------------------------------------------------------

def _usage(payload: dict, input_key: str, output_key: str) -> dict:
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return {}
    values = {"input_tokens": usage.get(input_key), "output_tokens": usage.get(output_key)}
    return {key: int(value) for key, value in values.items() if isinstance(value, int) and value}


def runs_usage(payload: dict) -> dict:
    """Non-zero usage of a Runs API terminal event or run status (``input_tokens``/``output_tokens``)."""
    return _usage(payload, "input_tokens", "output_tokens")


def _chat_usage(payload: dict) -> dict:
    """Non-zero usage of a Chat Completions finish chunk (``prompt_tokens``/``completion_tokens``)."""
    return _usage(payload, "prompt_tokens", "completion_tokens")


def _public_tool(name) -> str:
    """The tool name, or "" for a missing or internal (``_``-prefixed) tool."""
    name = name.strip() if isinstance(name, str) else ""
    return "" if name.startswith("_") else name


def _json_object(data: str) -> dict | None:
    try:
        payload = json.loads(data)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


# ---------------------------------------------------------------------------
# Dialect decoders
# ---------------------------------------------------------------------------

def _decode_runs_event(name: str, payload: dict) -> GatewayEvent | None:
    if name == "message.delta":
        delta = payload.get("delta")
        return TextDelta(delta) if isinstance(delta, str) and delta else None
    if name == "message.interim":
        text = payload.get("text")
        if isinstance(text, str) and text.strip():
            return InterimMessage(text.strip(), payload.get("already_streamed") is True)
        return None
    if name == "reasoning.available":
        text = payload.get("text")
        return ReasoningDelta(text) if isinstance(text, str) and text.strip() else None
    if name in ("tool.started", "tool.completed"):
        tool = _public_tool(payload.get("tool"))
        if not tool:
            return None
        tid = payload.get("tool_call_id")
        return ToolProgress(
            completed=name == "tool.completed",
            name=tool,
            preview=payload.get("preview"),
            is_error=payload.get("error") is True,
            tid=tid if isinstance(tid, str) and tid else None,
        )
    if name == "approval.request":
        return ApprovalRequest(payload)
    if name == "run.completed":
        return RunCompleted(
            output=str(payload.get("output") or ""),
            usage=runs_usage(payload),
            error=str(payload["error"]) if payload.get("error") else "",
            payload=payload,
        )
    if name == "run.failed":
        return RunFailed(str(payload.get("error") or "Gateway run failed"), payload)
    if name in ("run.cancelled", "run.interrupted"):
        return RunCancelled(payload)
    if name == "replay.truncated":
        return ReplayTruncated()
    return None  # lifecycle/subagent/control events the WebUI does not render


def decode_runs_frame(frame: SseFrame) -> GatewayFrame | None:
    """Decode one ``/v1/runs/{id}/events`` frame; ``None`` when it carries no JSON object.

    Every Runs API payload is a ``_run_event`` envelope: the event name in
    ``event`` and the replay cursor in ``seq``.
    """
    if not frame.has_payload:
        return None
    if frame.is_done:
        return DONE_FRAME
    payload = _json_object(frame.data)
    if payload is None:
        return None
    seq = payload.get("seq") if isinstance(payload.get("seq"), int) else None
    event = _decode_runs_event(str(payload.get("event") or ""), payload)
    return GatewayFrame(seq=seq, events=(event,) if event is not None else ())


def _decode_chat_chunk(payload: dict) -> list[GatewayEvent]:
    """Reasoning, text, usage and failure of one ``chat.completion.chunk``, in relay order."""
    events: list[GatewayEvent] = []
    choices = payload.get("choices")
    choice = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
    delta = choice.get("delta") if isinstance(choice.get("delta"), dict) else {}
    reasoning = delta.get("reasoning_content")
    if isinstance(reasoning, str) and reasoning.strip():
        events.append(ReasoningDelta(reasoning))
    content = delta.get("content")
    if isinstance(content, str) and content:
        events.append(TextDelta(content))
    usage = _chat_usage(payload)
    if usage:
        events.append(UsageReport(usage))
    if choice.get("finish_reason") == "error":
        error = payload.get("error")
        message = error.get("message") if isinstance(error, dict) else None
        events.append(ChunkError(str(message or "Gateway run failed")))
    return events


def decode_chat_completion_frame(frame: SseFrame) -> GatewayFrame | None:
    """Decode one ``/v1/chat/completions`` streaming frame; ``None`` when it carries no JSON object.

    Unnamed frames are ``chat.completion.chunk`` objects; Hermes extensions are
    named SSE events: ``hermes.tool.progress`` and ``approval.request``.
    """
    if not frame.has_payload:
        return None
    if frame.is_done:
        return DONE_FRAME
    payload = _json_object(frame.data)
    if payload is None:
        return None
    if frame.event == "message":
        return GatewayFrame(events=tuple(_decode_chat_chunk(payload)))
    if frame.event == "approval.request":
        return GatewayFrame(events=(ApprovalRequest(payload),))
    if frame.event == "hermes.tool.progress":
        tool = _public_tool(payload.get("tool"))
        tid = payload.get("toolCallId")
        if tool:
            return GatewayFrame(events=(ToolProgress(
                completed=payload.get("status") == "completed",
                name=tool,
                preview=payload.get("label"),
                tid=tid if isinstance(tid, str) and tid else None,
            ),))
    return GatewayFrame()  # other named events (e.g. ``hermes.status``) are not rendered


def decode_session_event(kind: str, payload: dict) -> GatewayEvent | None:
    """Decode the stream-content events of the canonical session protocol.

    Control events (approvals, clarifications, cancellation) stay with the
    session transport, which owns their request/response lifecycle.
    """
    payload = payload if isinstance(payload, dict) else {}
    if kind in ("message.delta", "reasoning.delta", "thinking.delta"):
        text = payload.get("text")  # ``None`` is a viewer-only segment boundary
        if not isinstance(text, str) or not text:
            return None
        return TextDelta(text) if kind == "message.delta" else ReasoningDelta(text)
    if kind in ("tool.start", "tool.complete"):
        tid = payload.get("tool_call_id")
        args = payload.get("args")
        return ToolProgress(
            completed=kind == "tool.complete",
            name=str(payload.get("name") or ""),
            args=bound_run_journal_snapshot_args(args) if isinstance(args, dict) else {},
            is_error=payload.get("is_error") is True,
            tid=tid if isinstance(tid, str) and tid else None,
        )
    return None
