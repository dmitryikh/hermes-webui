"""Layer 2 of the Hermes Gateway bridge: Hermes event decoding.

Turns the payload of one ``SseFrame`` (layer 1, ``api.gateway_sse``) into a
``GatewayFrame``: the Gateway cursor (``seq``) plus a tuple of typed
``GatewayEvent`` values. Decoding is pure — no stream registries, journals,
sessions, network or WebUI event names. What an event *means* for the WebUI
is decided by ``api.gateway_projection`` (layer 3) and by the run
orchestration in ``api.gateway_chat``.

The Gateway speaks three wire dialects, one decoder each:

* ``decode_runs_frame`` — ``GET /v1/runs/{id}/events`` (Runs API). Hermes
  event names travel in the JSON ``event``/``type`` field; untyped frames are
  OpenAI-compatible chunks.
* ``decode_chat_completion_frame`` — ``POST /v1/chat/completions`` streaming.
  OpenAI chunks, with Hermes extensions announced by the SSE ``event:`` field
  (``hermes.tool.progress``, ``reasoning.available``) and approval requests.
* ``decode_session_event`` — the canonical session protocol
  (``api.gateway_sessions``) whose frames arrive already parsed over JSON-RPC.
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
# ---------------------------------------------------------------------------

def chunk_text_delta(payload: dict) -> str:
    """Extract assistant text from an OpenAI-compatible streaming chunk."""
    try:
        choices = payload.get("choices") or []
        if not choices:
            return ""
        choice = choices[0] or {}
        delta = choice.get("delta") or {}
        content = delta.get("content")
        if isinstance(content, str):
            return content
        message = choice.get("message") or {}
        content = message.get("content")
        return content if isinstance(content, str) else ""
    except Exception:
        return ""


def chunk_reasoning_delta(payload: dict) -> str:
    """Extract reasoning text from OpenAI-compatible streaming chunks."""
    try:
        choices = payload.get("choices") or []
        if not choices:
            return ""
        choice = choices[0] or {}
        delta = choice.get("delta") or {}
        reasoning = delta.get("reasoning_content")
        if isinstance(reasoning, str) and reasoning.strip():
            return reasoning
        message = choice.get("message") or {}
        reasoning = message.get("reasoning_content")
        return reasoning if isinstance(reasoning, str) and reasoning.strip() else ""
    except Exception:
        return ""


def payload_usage(payload: dict) -> dict:
    """Normalize OpenAI/Hermes usage names to WebUI usage keys."""
    usage = payload.get("usage") if isinstance(payload, dict) else None
    if not isinstance(usage, dict):
        return {}
    return {
        "input_tokens": int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0),
        "output_tokens": int(usage.get("completion_tokens") or usage.get("output_tokens") or 0),
        "estimated_cost": usage.get("estimated_cost") or usage.get("estimated_cost_usd") or 0,
    }


def nonzero_usage(payload: dict) -> dict:
    """``payload_usage`` without zero/empty fields (safe to ``dict.update`` into a total)."""
    return {k: v for k, v in payload_usage(payload).items() if v}


def reasoning_text(payload: dict) -> str:
    """Reasoning text of a ``reasoning.available`` / ``_thinking`` payload."""
    if not isinstance(payload, dict):
        return ""
    for key in ("text", "preview", "delta", "content"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def decode_tool_progress(payload: dict) -> ToolProgress | ReasoningDelta | None:
    """Decode a Hermes tool-progress / reasoning payload.

    ``_thinking`` progress and ``reasoning.available`` are reasoning; other
    underscore-prefixed tools are internal and dropped.
    """
    if not isinstance(payload, dict):
        return None
    event_type = str(payload.get("event") or "").strip().lower()
    if event_type == "reasoning.available":
        text = reasoning_text(payload)
        return ReasoningDelta(text) if text else None
    name = str(payload.get("tool") or payload.get("name") or payload.get("function_name") or "").strip()
    if not name:
        return None
    if name == "_thinking":
        text = reasoning_text(payload)
        return ReasoningDelta(text) if text else None
    if name.startswith("_"):
        return None
    status = str(payload.get("status") or "running").strip().lower()
    tid = payload.get("toolCallId") or payload.get("tool_call_id") or payload.get("id")
    return ToolProgress(
        completed=event_type == "tool.completed" or status in {"completed", "complete", "success", "error", "failed"},
        name=name,
        preview=payload.get("label") or payload.get("preview"),
        args=bound_run_journal_snapshot_args(payload.get("args")) if isinstance(payload.get("args"), dict) else {},
        is_error=bool(payload.get("error")) or status in {"error", "failed"},
        tid=str(tid) if tid else None,
    )


def _chunk_events(payload: dict) -> list[GatewayEvent]:
    """Reasoning, text and usage carried by one OpenAI-compatible chunk, in relay order."""
    events: list[GatewayEvent] = []
    reasoning = chunk_reasoning_delta(payload)
    if reasoning:
        events.append(ReasoningDelta(reasoning))
    text = chunk_text_delta(payload)
    if text:
        events.append(TextDelta(text))
    usage = nonzero_usage(payload)
    if usage:
        events.append(UsageReport(usage))
    return events


def _json_object(data: str) -> dict | None:
    try:
        payload = json.loads(data)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def payload_event_name(payload: dict, sse_event: str = "message") -> str:
    """The Hermes event name: JSON ``event``/``type`` field, else the SSE event name."""
    return str(payload.get("event") or payload.get("type") or sse_event).strip() or "message"


# ---------------------------------------------------------------------------
# Dialect decoders
# ---------------------------------------------------------------------------

_RUNS_TOOL_EVENTS = frozenset({"tool.started", "tool.completed", "reasoning.available"})
APPROVAL_EVENT_NAMES = frozenset({"hermes.approval.request", "approval.request"})


def decode_runs_frame(frame: SseFrame) -> GatewayFrame | None:
    """Decode one ``/v1/runs/{id}/events`` frame; ``None`` when it carries no JSON object."""
    if not frame.has_payload:
        return None
    if frame.is_done:
        return DONE_FRAME
    payload = _json_object(frame.data)
    if payload is None:
        return None
    seq = payload.get("seq") if isinstance(payload.get("seq"), int) else None
    name = payload_event_name(payload, frame.event)
    events: list[GatewayEvent] = []
    if name == "replay.truncated":
        events.append(ReplayTruncated())
    elif name == "approval.request":
        events.append(ApprovalRequest(payload))
    elif name in _RUNS_TOOL_EVENTS:
        decoded = decode_tool_progress(payload)
        if decoded is not None:
            events.append(decoded)
    elif name == "message.interim":
        text = payload.get("text")
        if isinstance(text, str) and text.strip():
            events.append(InterimMessage(text.strip(), payload.get("already_streamed") is True))
    elif name == "message.delta":
        delta = str(payload.get("delta") or "")
        if delta:
            events.append(TextDelta(delta))
    elif name == "run.completed":
        events.append(RunCompleted(
            output=str(payload.get("output") or ""),
            usage=nonzero_usage(payload),
            error=str(payload["error"]) if payload.get("error") else "",
            payload=payload,
        ))
    elif name == "run.failed":
        events.append(RunFailed(str(payload.get("error") or "Gateway run failed"), payload))
    elif name == "run.cancelled":
        events.append(RunCancelled(payload))
    else:
        events.extend(_chunk_events(payload))
    return GatewayFrame(seq=seq, events=tuple(events))


def decode_chat_completion_frame(frame: SseFrame) -> GatewayFrame | None:
    """Decode one ``/v1/chat/completions`` streaming frame; ``None`` when it carries no JSON object."""
    if not frame.has_payload:
        return None
    if frame.is_done:
        return DONE_FRAME
    payload = _json_object(frame.data)
    if payload is None:
        return None
    events: list[GatewayEvent] = []
    if payload_event_name(payload, frame.event) in APPROVAL_EVENT_NAMES:
        events.append(ApprovalRequest(payload))
    elif frame.event == "hermes.tool.progress":
        decoded = decode_tool_progress(payload)
        if decoded is not None:
            events.append(decoded)
    elif frame.event == "reasoning.available":
        text = reasoning_text(payload)
        if text:
            events.append(ReasoningDelta(text))
    else:
        if payload.get("error"):
            events.append(ChunkError(str(payload["error"])))
        events.extend(_chunk_events(payload))
    return GatewayFrame(events=tuple(events))


def decode_session_event(kind: str, payload: dict) -> GatewayEvent | None:
    """Decode the stream-content events of the canonical session protocol.

    Control events (approvals, clarifications, cancellation) stay with the
    session transport, which owns their request/response lifecycle.
    """
    payload = payload if isinstance(payload, dict) else {}
    if kind == "message.delta":
        text = payload.get("text") or payload.get("delta") or ""
        return TextDelta(text) if isinstance(text, str) and text else None
    if kind in {"reasoning.delta", "thinking.delta"}:
        text = payload.get("text") or payload.get("delta") or ""
        return ReasoningDelta(text) if isinstance(text, str) and text else None
    if kind in {"tool.start", "tool.complete"}:
        tid = payload.get("tool_call_id") or payload.get("id")
        args = payload.get("args")
        return ToolProgress(
            completed=kind == "tool.complete",
            name=str(payload.get("name") or payload.get("tool") or ""),
            args=bound_run_journal_snapshot_args(args) if isinstance(args, dict) else {},
            is_error=bool(payload.get("is_error", payload.get("error", False))),
            tid=str(tid) if tid else None,
        )
    return None
