"""The Gateway transport contract: one transport-independent turn lifecycle.

``_run_gateway_chat_streaming`` owns everything except ``transport.run``; these
tests drive the real worker with a fake transport and pin how each kind of
``TurnResult`` is settled, then pin which real transport a turn selects.
"""
from __future__ import annotations

import urllib.error
from collections import OrderedDict
from types import SimpleNamespace

import pytest

import api.gateway_chat as gateway_chat
import api.models as models
from api import gateway_transport
from api.config import STREAMS, create_stream_channel
from api.gateway_transport import GatewayTransport, TurnOutcomeUnknown, TurnResult
from api.models import new_session


class FakeTransport(GatewayTransport):
    name = "fake"

    def __init__(self, behaviour, *, durable=False):
        self.behaviour = behaviour
        self.durable = durable
        self.turns = []

    def run(self, turn):
        self.turns.append(turn)
        return self.behaviour(turn)


@pytest.fixture
def worker(tmp_path, monkeypatch):
    session_dir = tmp_path / "sessions"
    session_dir.mkdir()
    monkeypatch.setattr(models, "SESSION_DIR", session_dir)
    monkeypatch.setattr(models, "SESSION_INDEX_FILE", session_dir / "_index.json")
    monkeypatch.setattr(models, "SESSIONS", OrderedDict())
    monkeypatch.setenv("HERMES_WEBUI_GATEWAY_BASE_URL", "http://gateway.local")

    def run(transport):
        monkeypatch.setattr(gateway_transport, "select_transport", lambda *a, **k: transport)
        s = new_session()
        stream_id = f"stream-transport-{s.session_id}"
        s.active_stream_id = stream_id
        s.pending_user_message = "Say hello"
        s.pending_attachments = []
        s.pending_started_at = 123
        s.save()
        channel = create_stream_channel()
        subscriber = channel.subscribe()
        STREAMS[stream_id] = channel
        gateway_chat._run_gateway_chat_streaming(s.session_id, "Say hello", "test-model", str(tmp_path), stream_id, [])
        events = []
        while not subscriber.empty():
            events.append(subscriber.get_nowait())
        return models.get_session(s.session_id), [(item[0], item[1]) for item in events]

    return run


def test_the_turn_reaches_the_transport_with_everything_resolved(worker):
    transport = FakeTransport(lambda turn: TurnResult(text="hello", usage={"input_tokens": 3}))
    saved, events = worker(transport)

    (turn,) = transport.turns
    assert (turn.msg_text, turn.model, turn.base_url) == ("Say hello", "test-model", "http://gateway.local")
    assert turn.prefill_messages and turn.prefill_messages[0]["role"] == "system"
    assert callable(turn.emit) and callable(turn.record_run) and turn.reattach_run is None
    assert [m["content"] for m in saved.messages[-2:]] == ["Say hello", "hello"]
    assert saved.active_stream_id is None and saved.pending_user_message is None
    done = [data for name, data in events if name == "done"]
    assert done and done[0]["usage"]["input_tokens"] == 3


def test_a_transport_error_result_settles_the_turn_as_failed(worker):
    saved, events = worker(FakeTransport(lambda turn: TurnResult(text="part", error="model exploded")))
    assert saved.messages[-1].get("_error") is True
    assert [name for name, _ in events].count("apperror") == 1
    assert "done" not in [name for name, _ in events]


def test_any_transport_exception_settles_the_turn_as_failed(worker):
    def fail(turn):
        raise urllib.error.URLError("connection refused")

    saved, events = worker(FakeTransport(fail))
    assert saved.messages[-1].get("_error") is True
    assert saved.active_stream_id is None
    assert [name for name, _ in events].count("apperror") == 1


def test_a_cancelled_result_persists_the_cancelled_turn(worker, monkeypatch):
    settled = []
    monkeypatch.setattr(gateway_chat, "_settle_gateway_cancelled_turn", lambda *a: settled.append(a))

    def cancel(turn):
        turn.emit("cancel", {"message": "Cancelled by user"})
        return TurnResult(text=None)

    _saved, events = worker(FakeTransport(cancel))
    assert len(settled) == 1
    assert "done" not in [name for name, _ in events]


def test_an_unknown_outcome_preserves_the_pending_turn(worker):
    def lost(turn):
        raise TurnOutcomeUnknown("connection lost after admission")

    saved, events = worker(FakeTransport(lost, durable=True))
    assert saved.pending_user_message == "Say hello"
    assert [data["type"] for name, data in events if name == "apperror"] == ["gateway_session_disconnected"]


def test_an_empty_answer_is_reported_not_saved(worker):
    saved, events = worker(FakeTransport(lambda turn: TurnResult(text="   ")))
    assert [data["type"] for name, data in events if name == "apperror"] == ["gateway_empty_response"]
    assert all(m.get("content") != "   " for m in saved.messages)


def test_transport_selection(monkeypatch):
    from api.gateway_sessions import SessionProtocolTransport
    from api.gateway_transport_chat import ChatCompletionsTransport
    from api.gateway_transport_runs import RunsApiTransport

    session = SimpleNamespace(session_source="webui", raw_source="webui")
    monkeypatch.setattr(gateway_transport, "gateway_supports_approval", lambda *a: True)

    def pick(cfg=None, reattach_run=None):
        return type(gateway_transport.select_transport(
            session, cfg or {}, base_url="http://gateway.local", api_key="", reattach_run=reattach_run,
        ))

    assert pick() is ChatCompletionsTransport
    assert pick({"webui_gateway_use_runs_api": "true"}) is RunsApiTransport
    assert pick(reattach_run={"run_id": "run-1"}) is RunsApiTransport
    assert pick(reattach_run={"run_id": "session:x", "transport": "session"}) is SessionProtocolTransport
    monkeypatch.setattr(gateway_transport, "gateway_supports_approval", lambda *a: False)
    assert pick({"webui_gateway_use_runs_api": "true"}) is ChatCompletionsTransport
    # Only the Runs API and the session protocol publish a run identity.
    assert (ChatCompletionsTransport.durable, RunsApiTransport.durable, SessionProtocolTransport.durable) == (
        False, True, True,
    )
