"""Layer 3 of the Hermes Gateway bridge: projecting Gateway events onto WebUI state.

Input: typed ``GatewayEvent`` values from ``api.gateway_events`` (layer 2).
Output, for every stream-content event:

* one WebUI SSE event (``token``, ``reasoning``, ``tool``, ``tool_complete``,
  ``interim_assistant``) handed to the caller's ``emit``. In the Gateway
  worker that callable journals the event (``RunJournalWriter``) and puts it
  on the browser queue, so the journal is exactly the sequence of projected
  events;
* the matching update of the stream's reconnect snapshot
  (``LiveTurnSnapshot``).

Because both outputs come from the same projection, the reconnect snapshot is
always the fold of the journaled WebUI events. ``TurnCheckpoint.restore``
relies on that: after a WebUI restart it rebuilds the snapshot by folding the
journal with the very same ``apply`` the live path uses.

Control-plane events (approvals, terminal run events, usage) are not
projected here: their side effects (approval cards, pending steer, turn
settlement) are run orchestration and live in ``api.gateway_chat``.
"""
from __future__ import annotations

import logging
from typing import Callable

from api.config import STREAM_LIVE_TOOL_CALLS, STREAM_PARTIAL_TEXT, STREAM_REASONING_TEXT
from api.gateway_events import GatewayEvent, InterimMessage, ReasoningDelta, TextDelta, ToolProgress

logger = logging.getLogger(__name__)

# WebUI events that change the reconnect snapshot. Every journaled row of
# these kinds must carry ``gateway_seq`` for the journal to be a replay cursor.
SNAPSHOT_EVENTS = frozenset({"token", "interim_assistant", "reasoning", "tool", "tool_complete"})


class LiveTurnSnapshot:
    """The reconnect snapshot of one Gateway-backed stream.

    Owns, for ``stream_id``, the shared rows a reconnecting browser renders
    from: the open answer segment (``STREAM_PARTIAL_TEXT``), its reasoning
    (``STREAM_REASONING_TEXT``) and the live tool calls
    (``STREAM_LIVE_TOOL_CALLS``). The shared rows exist only while the stream
    is registered; every update is a no-op for a row that is absent, so a
    late event can never resurrect a released stream.

    ``text`` is the relay's local carrier of the open answer segment. It moves
    in lockstep with ``STREAM_PARTIAL_TEXT`` (seed / append / adopt) but
    survives when the shared row is absent, so a relay always returns exactly
    what it streamed.
    """

    __slots__ = ("stream_id", "text")

    def __init__(self, stream_id: str, text: str | None = None):
        self.stream_id = stream_id
        # Seed: resume from exactly what the browser has seen.
        self.text = STREAM_PARTIAL_TEXT.get(stream_id, "") if text is None else text

    # ---- registry lifecycle -------------------------------------------------

    @staticmethod
    def register(stream_id: str) -> None:
        """Create empty shared rows for a newly admitted stream."""
        STREAM_PARTIAL_TEXT[stream_id] = ""
        STREAM_REASONING_TEXT[stream_id] = ""
        STREAM_LIVE_TOOL_CALLS[stream_id] = []

    @staticmethod
    def reset(stream_id: str) -> None:
        """Empty the shared rows that exist (never creates rows for a released stream)."""
        if stream_id in STREAM_PARTIAL_TEXT:
            STREAM_PARTIAL_TEXT[stream_id] = ""
        if stream_id in STREAM_REASONING_TEXT:
            STREAM_REASONING_TEXT[stream_id] = ""
        if stream_id in STREAM_LIVE_TOOL_CALLS:
            STREAM_LIVE_TOOL_CALLS[stream_id] = []

    @staticmethod
    def release(stream_id: str) -> None:
        STREAM_PARTIAL_TEXT.pop(stream_id, None)
        STREAM_REASONING_TEXT.pop(stream_id, None)
        STREAM_LIVE_TOOL_CALLS.pop(stream_id, None)

    # ---- reads --------------------------------------------------------------

    @staticmethod
    def partial_text(stream_id: str) -> str:
        return str(STREAM_PARTIAL_TEXT.get(stream_id) or "")

    @staticmethod
    def reasoning_text(stream_id: str) -> str:
        return STREAM_REASONING_TEXT.get(stream_id, "")

    @staticmethod
    def tool_calls(stream_id: str) -> list:
        return list(STREAM_LIVE_TOOL_CALLS.get(stream_id, []) or [])

    # ---- updates ------------------------------------------------------------

    def append_text(self, delta: str) -> str:
        self.text += delta
        if self.stream_id in STREAM_PARTIAL_TEXT:
            STREAM_PARTIAL_TEXT[self.stream_id] += delta
        return self.text

    def adopt_text(self, text: str) -> str:
        """Replace the open segment with an authoritative answer (``run.completed``/status output)."""
        self.text = text
        if self.stream_id in STREAM_PARTIAL_TEXT:
            STREAM_PARTIAL_TEXT[self.stream_id] = text
        return text

    def append_reasoning(self, text: str) -> None:
        if text and self.stream_id in STREAM_REASONING_TEXT:
            STREAM_REASONING_TEXT[self.stream_id] += text

    def seal_segment(self) -> None:
        """Close the open segment at an interim boundary.

        The journal owns the sealed segment, including its reasoning; the
        snapshot keeps only the still-open segment.
        """
        self.adopt_text("")
        if self.stream_id in STREAM_REASONING_TEXT:
            STREAM_REASONING_TEXT[self.stream_id] = ""

    def note_tool(self, event_name: str, payload: dict) -> None:
        calls = STREAM_LIVE_TOOL_CALLS.get(self.stream_id)
        if calls is None:
            return
        if event_name == "tool":
            calls.append({
                "name": payload.get("name"),
                "args": payload.get("args") or {},
                "done": False,
                **({"tid": payload.get("tid")} if payload.get("tid") else {}),
            })
        elif event_name == "tool_complete":
            for call in reversed(calls):
                if call.get("done"):
                    continue
                if (payload.get("tid") and call.get("tid") == payload.get("tid")) or call.get("name") == payload.get("name"):
                    call["done"] = True
                    call["is_error"] = bool(payload.get("is_error"))
                    break

    # ---- fold ---------------------------------------------------------------

    def apply(self, event_name: str, payload: dict) -> None:
        """Fold one projected WebUI event (live or read back from the journal)."""
        if event_name == "token":
            self.append_text(str(payload.get("text") or ""))
        elif event_name == "interim_assistant":
            if isinstance(payload.get("text"), str) and payload["text"].strip():
                self.seal_segment()
        elif event_name == "reasoning":
            self.append_reasoning(payload.get("text"))
        elif event_name in {"tool", "tool_complete"}:
            self.note_tool(event_name, payload)


class TurnCheckpoint:
    """The single recovery state of a Gateway-backed turn.

    A checkpoint is what the WebUI has relayed so far: the Gateway cursor of
    the last relayed frame and the reconnect snapshot that relaying produced.
    Every way a turn can lose its connection to the Gateway — read timeout,
    HTTP error, reset, clean EOF, watchdog stall, WebUI restart — resumes from
    a checkpoint the same way: ask the Gateway for events after ``cursor``
    and keep projecting onto ``snapshot``.

    In a live worker the checkpoint is maintained in lockstep with the run
    journal (the relay commits each frame's cursor after its WebUI events are
    journaled). After a WebUI restart ``restore`` rebuilds the identical
    checkpoint by folding the journal with ``LiveTurnSnapshot.apply``.

    ``replayable`` is False when no trustworthy cursor exists (the journal is
    unreadable, or has snapshot rows without a Gateway cursor): replaying
    from any cursor could duplicate output, so the turn is followed by
    durable run status only.
    """

    __slots__ = ("snapshot", "cursor", "replayable")

    def __init__(self, stream_id: str, *, text: str | None = None, cursor: int = -1, replayable: bool = True):
        self.snapshot = LiveTurnSnapshot(stream_id, text=text)
        self.cursor = cursor
        self.replayable = replayable

    @property
    def stream_id(self) -> str:
        return self.snapshot.stream_id

    @property
    def text(self) -> str:
        return self.snapshot.text

    def commit(self, seq: int) -> None:
        """Advance the cursor past a frame whose events have been relayed."""
        self.cursor = seq

    @classmethod
    def restore(cls, session_id: str, stream_id: str) -> "TurnCheckpoint":
        """Rebuild the checkpoint (and the stream's snapshot rows) from the run journal.

        Idempotent: the snapshot rows are reset and re-folded from the journal.
        Every journaled row (approvals included) advances the cursor.
        """
        from api.run_journal import read_run_events

        try:
            events = read_run_events(session_id, stream_id).get("events") or []
        except Exception:
            logger.debug("Failed to read run journal for stream %s", stream_id, exc_info=True)
            return cls(stream_id, text="", replayable=False)
        stateful, cursor = [], -1
        for row in events:
            payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
            seq = payload.get("gateway_seq")
            if row.get("event") in SNAPSHOT_EVENTS:
                if not isinstance(seq, int):
                    return cls(stream_id, text="", replayable=False)
                stateful.append((row["event"], payload))
            if isinstance(seq, int):
                cursor = max(cursor, seq)
        LiveTurnSnapshot.reset(stream_id)
        checkpoint = cls(stream_id, text="", cursor=cursor)
        for name, payload in stateful:
            checkpoint.snapshot.apply(name, payload)
        return checkpoint


class GatewayEventProjector:
    """Projects stream-content Gateway events to WebUI events and the snapshot.

    ``emit(event_name, payload)`` receives the WebUI event (the caller stamps
    the Gateway cursor and journals it). ``on_tool_activity(name)`` is told
    about tool progress so the caller can update its run-phase registry.
    """

    __slots__ = ("snapshot", "emit", "on_tool_activity")

    def __init__(
        self,
        snapshot: LiveTurnSnapshot,
        emit: Callable[[str, dict], None],
        *,
        on_tool_activity: Callable[[str | None], None] | None = None,
    ):
        self.snapshot = snapshot
        self.emit = emit
        self.on_tool_activity = on_tool_activity

    @property
    def text(self) -> str:
        return self.snapshot.text

    def project(self, event: GatewayEvent) -> bool:
        """Project one event; ``False`` when it is not a stream-content event."""
        if isinstance(event, TextDelta):
            self.snapshot.append_text(event.text)
            self.emit("token", {"text": event.text})
        elif isinstance(event, ReasoningDelta):
            self.snapshot.append_reasoning(event.text)
            self.emit("reasoning", {"text": event.text})
        elif isinstance(event, ToolProgress):
            name, payload = tool_event(event)
            self.snapshot.note_tool(name, payload)
            self.emit(name, payload)
            if self.on_tool_activity is not None:
                self.on_tool_activity(event.name)
        elif isinstance(event, InterimMessage):
            self.emit("interim_assistant", {"text": event.text, "already_streamed": event.already_streamed})
            self.snapshot.seal_segment()
        else:
            return False
        return True


def tool_event(event: ToolProgress) -> tuple[str, dict]:
    """The WebUI ``tool``/``tool_complete`` event for one tool-progress event."""
    payload = {
        "event_type": "tool.completed" if event.completed else "tool.started",
        "name": event.name,
        "preview": event.preview,
        "args": event.args,
        "is_error": event.is_error,
    }
    if event.tid:
        payload["tid"] = event.tid
    return ("tool_complete" if event.completed else "tool"), payload
