"""Unit tests for the layered Hermes Gateway bridge.

layer 1  api.gateway_sse         SSE lines -> SseFrame
layer 2  api.gateway_events      SseFrame / session frame -> GatewayFrame(seq, events)
layer 3  api.gateway_projection  GatewayEvent -> WebUI event + LiveTurnSnapshot

The cross-layer invariant pinned at the end: the reconnect snapshot built live
equals the snapshot restored by folding the journaled WebUI events.
"""
from __future__ import annotations

import json

import pytest

from api import config, run_journal
from api.gateway_events import (
    ApprovalRequest,
    ChunkError,
    InterimMessage,
    ReasoningDelta,
    ReplayTruncated,
    RunCancelled,
    RunCompleted,
    RunFailed,
    TextDelta,
    ToolProgress,
    UsageReport,
    decode_chat_completion_frame,
    decode_runs_frame,
    decode_session_event,
)
from api.gateway_projection import GatewayEventProjector, LiveTurnSnapshot, TurnCheckpoint
from api.gateway_sse import SseFrame, iter_sse_frames
from api import gateway_transport_runs as gateway_runs

STREAM = "layers-stream"


@pytest.fixture
def registered_stream():
    LiveTurnSnapshot.register(STREAM)
    try:
        yield STREAM
    finally:
        LiveTurnSnapshot.release(STREAM)


def data(payload) -> SseFrame:
    return SseFrame("message", json.dumps(payload))


# ---------------------------------------------------------------- layer 1 ---

def test_sse_framer_classifies_lines_and_scopes_event_names():
    frames = list(iter_sse_frames([
        b": keepalive",
        b"event: hermes.tool.progress",
        b'data: {"a": 1}',
        b'data: {"b": 2}',
        "event: reasoning.available",
        b"",
        b"data: [DONE]",
    ]))
    payload_frames = [f for f in frames if f.has_payload]
    assert [(f.event, f.data) for f in payload_frames] == [
        ("hermes.tool.progress", '{"a": 1}'),
        # The event name names only the next data line.
        ("message", '{"b": 2}'),
        # A blank separator resets a pending event name.
        ("message", "[DONE]"),
    ]
    assert payload_frames[-1].is_done
    # Comments, fields and separators are surfaced as liveness-only frames.
    assert [f.has_payload for f in frames] == [False, False, True, True, False, False, True]


# ---------------------------------------------------------------- layer 2 ---

def test_runs_decoder_types_every_hermes_event():
    assert decode_runs_frame(SseFrame()) is None
    assert decode_runs_frame(SseFrame("message", "not json")) is None
    assert decode_runs_frame(SseFrame("message", "[1, 2]")) is None
    assert decode_runs_frame(SseFrame("message", "[DONE]")).done

    def events(payload):
        return decode_runs_frame(data(payload)).events

    assert decode_runs_frame(data({"event": "message.delta", "seq": 7, "delta": "hi"})).seq == 7
    assert events({"event": "message.delta", "delta": "hi"}) == (TextDelta("hi"),)
    assert events({"event": "message.delta", "delta": ""}) == ()
    assert events({"event": "message.interim", "text": "  Look.  ", "already_streamed": True}) == (
        InterimMessage("Look.", True),
    )
    assert events({"event": "message.interim", "text": "   "}) == ()
    assert events({"event": "reasoning.available", "text": "hmm"}) == (ReasoningDelta("hmm"),)
    (tool,) = events({"event": "tool.completed", "tool": "terminal", "tool_call_id": "t1", "error": True})
    assert tool == ToolProgress(completed=True, name="terminal", args={}, is_error=True, tid="t1")
    assert events({"event": "tool.started", "tool": "_internal"}) == ()
    assert events({"event": "replay.truncated"}) == (ReplayTruncated(),)
    approval = {"event": "approval.request", "command": "ls"}
    assert events(approval) == (ApprovalRequest(approval),)
    completed = {"event": "run.completed", "output": "done", "usage": {"input_tokens": 3, "output_tokens": 0}}
    assert events(completed) == (RunCompleted("done", {"input_tokens": 3}, "", completed),)
    failed = {"event": "run.failed"}
    assert events(failed) == (RunFailed("Gateway run failed", failed),)
    for name in ("run.cancelled", "run.interrupted"):
        cancelled = {"event": name}
        assert events(cancelled) == (RunCancelled(cancelled),)
    # Lifecycle/subagent events and untyped payloads are not rendered.
    assert events({"event": "run.started"}) == ()
    assert events({"choices": [{"delta": {"content": "c"}}]}) == ()


def test_chat_completion_decoder_routes_hermes_extensions_by_sse_event():
    tool = {"tool": "terminal", "status": "running"}
    assert decode_chat_completion_frame(SseFrame("hermes.tool.progress", json.dumps(tool))).events == (
        ToolProgress(completed=False, name="terminal"),
    )
    # Reasoning rides on chunks as ``delta.reasoning_content``.
    assert decode_chat_completion_frame(
        data({"choices": [{"delta": {"reasoning_content": "thinking"}}]})
    ).events == (ReasoningDelta("thinking"),)
    # Without the SSE event name the same payload is just an (empty) chunk.
    assert decode_chat_completion_frame(data(tool)).events == ()
    # A failed turn ends with an ``error`` finish chunk; the message is optional.
    failed = {"choices": [{"delta": {}, "finish_reason": "error"}], "error": {"message": "boom", "type": "agent_error"}}
    assert decode_chat_completion_frame(data(failed)).events == (ChunkError("boom"),)
    crashed = {"choices": [{"delta": {}, "finish_reason": "error"}]}
    assert decode_chat_completion_frame(data(crashed)).events == (ChunkError("Gateway run failed"),)
    assert decode_chat_completion_frame(SseFrame("hermes.status", json.dumps({"x": 1}))).events == ()


def test_session_decoder_maps_canonical_content_events():
    assert decode_session_event("message.delta", {"text": "a"}) == TextDelta("a")
    assert decode_session_event("thinking.delta", {"text": "b"}) == ReasoningDelta("b")
    assert decode_session_event("message.delta", {"text": None}) is None  # segment boundary
    assert decode_session_event("reasoning.delta", {"text": ""}) is None
    assert decode_session_event("tool.complete", {"name": "x", "tool_call_id": "t", "is_error": True}) == (
        ToolProgress(completed=True, name="x", is_error=True, tid="t")
    )
    assert decode_session_event("approval.request", {}) is None


# ---------------------------------------------------------------- layer 3 ---

def test_projector_emits_webui_events_and_updates_the_snapshot(registered_stream):
    emitted, tools = [], []
    projector = GatewayEventProjector(
        LiveTurnSnapshot(registered_stream), lambda e, d: emitted.append((e, d)), on_tool_activity=tools.append,
    )
    for event in (
        ReasoningDelta("plan "),
        TextDelta("Checking."),
        ToolProgress(completed=False, name="terminal", args={"command": "pwd"}, tid="t1"),
        InterimMessage("Checking.", already_streamed=True),
        ToolProgress(completed=True, name="terminal", tid="t1"),
        TextDelta("Done."),
    ):
        assert projector.project(event)
    assert not projector.project(UsageReport({"input_tokens": 1}))
    assert [e for e, _ in emitted] == ["reasoning", "token", "tool", "interim_assistant", "tool_complete", "token"]
    assert emitted[2][1] == {
        "event_type": "tool.started", "name": "terminal", "preview": None,
        "args": {"command": "pwd"}, "is_error": False, "tid": "t1",
    }
    assert tools == ["terminal", "terminal"]
    # The interim boundary sealed the first segment and its reasoning.
    assert projector.text == config.STREAM_PARTIAL_TEXT[registered_stream] == "Done."
    assert config.STREAM_REASONING_TEXT[registered_stream] == ""
    assert config.STREAM_LIVE_TOOL_CALLS[registered_stream] == [
        {"name": "terminal", "args": {"command": "pwd"}, "done": True, "tid": "t1", "is_error": False},
    ]


def test_snapshot_never_resurrects_a_released_stream():
    snapshot = LiveTurnSnapshot("released-stream", text="")
    snapshot.append_text("late")
    snapshot.append_reasoning("late")
    snapshot.note_tool("tool", {"name": "x"})
    snapshot.seal_segment()
    assert snapshot.text == ""
    assert "released-stream" not in config.STREAM_PARTIAL_TEXT
    assert "released-stream" not in config.STREAM_REASONING_TEXT
    assert "released-stream" not in config.STREAM_LIVE_TOOL_CALLS


def test_restored_snapshot_equals_the_live_snapshot(monkeypatch, registered_stream):
    """The journal is the fold source: restoring it rebuilds exactly the live snapshot."""
    journal = []

    def emit(name, payload):
        journal.append({"event": name, "payload": {**payload, "gateway_seq": len(journal)}})

    projector = GatewayEventProjector(LiveTurnSnapshot(registered_stream), emit)
    lines = [
        {"event": "reasoning.available", "text": "sealed thought"},
        {"event": "message.delta", "delta": "Checking."},
        {"event": "tool.started", "tool": "terminal", "tool_call_id": "t1"},
        {"event": "message.interim", "text": "Checking.", "already_streamed": True},
        {"event": "tool.completed", "tool": "terminal", "tool_call_id": "t1"},
        {"event": "reasoning.available", "text": "open thought"},
        {"event": "message.delta", "delta": "Ans"},
    ]
    for payload in lines:
        for event in decode_runs_frame(data(payload)).events:
            projector.project(event)
    live = (
        LiveTurnSnapshot.partial_text(registered_stream),
        LiveTurnSnapshot.reasoning_text(registered_stream),
        LiveTurnSnapshot.tool_calls(registered_stream),
    )
    assert live[0] == "Ans" and live[1] == "open thought"

    LiveTurnSnapshot.register(registered_stream)
    monkeypatch.setattr(run_journal, "read_run_events", lambda *a, **k: {"events": journal})
    checkpoint = TurnCheckpoint.restore("sid", registered_stream)
    assert (checkpoint.cursor, checkpoint.replayable, checkpoint.text) == (len(journal) - 1, True, "Ans")
    restored = (
        LiveTurnSnapshot.partial_text(registered_stream),
        LiveTurnSnapshot.reasoning_text(registered_stream),
        LiveTurnSnapshot.tool_calls(registered_stream),
    )
    assert restored == live


def test_checkpoint_restore_is_idempotent_and_fails_closed(monkeypatch, registered_stream):
    journal = [
        {"event": "token", "payload": {"text": "Hi", "gateway_seq": 0}},
        {"event": "approval", "payload": {"approval_id": "a", "gateway_seq": 1}},
    ]
    monkeypatch.setattr(run_journal, "read_run_events", lambda *a, **k: {"events": journal})
    config.STREAM_PARTIAL_TEXT[registered_stream] = "stale"
    for _ in range(2):  # recovering twice rebuilds the same state
        checkpoint = TurnCheckpoint.restore("sid", registered_stream)
        assert (checkpoint.cursor, checkpoint.replayable, checkpoint.text) == (1, True, "Hi")
        assert config.STREAM_PARTIAL_TEXT[registered_stream] == "Hi"

    # A snapshot row without a Gateway cursor: replaying from anywhere could duplicate.
    journal.append({"event": "token", "payload": {"text": "?"}})
    assert TurnCheckpoint.restore("sid", registered_stream).replayable is False

    def unreadable(*a, **k):
        raise OSError("journal unreadable")

    monkeypatch.setattr(run_journal, "read_run_events", unreadable)
    assert TurnCheckpoint.restore("sid", registered_stream).replayable is False


def test_relay_updates_the_checkpoint_in_place_when_the_connection_dies(registered_stream):
    """Nothing is resynced after a reset: the checkpoint already holds every relayed frame."""
    import threading

    from api import gateway_chat

    def dying():
        yield ('data: ' + json.dumps({"event": "message.delta", "delta": "A", "seq": 5})).encode()
        raise ConnectionResetError("reset by peer")

    checkpoint = TurnCheckpoint(registered_stream, text="")
    with pytest.raises(ConnectionResetError):
        gateway_runs._relay_gateway_run_events(
            dying(), "sid", registered_stream, "run", "http://x", "",
            put_gateway_event=lambda e, d: None, cancel_event=threading.Event(), checkpoint=checkpoint,
        )
    assert (checkpoint.cursor, checkpoint.text) == (5, "A")
    assert config.STREAM_PARTIAL_TEXT[registered_stream] == "A"
