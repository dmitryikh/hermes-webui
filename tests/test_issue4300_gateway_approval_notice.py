"""Regression coverage for #4300 legacy gateway approval unsupported notice."""

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MESSAGES_JS = (REPO / "static" / "messages.js").read_text(encoding="utf-8")
I18N_JS = (REPO / "static" / "i18n.js").read_text(encoding="utf-8")


def _notices(reason, *, session=None, sends=2):
    """Run the Chat Completions transport's approval notice ``sends`` times for one session."""
    from types import SimpleNamespace
    from unittest.mock import patch

    from api import gateway_transport_chat
    from api.gateway_transport import GatewayTurn

    session = session if session is not None else SimpleNamespace()
    events = []
    turn = GatewayTurn(
        session=session, session_id="sid-4300", stream_id="stream-4300", msg_text="hi",
        model=None, model_provider=None, workspace="/tmp", attachments=None, cfg={},
        base_url="http://gateway.test", api_key="", emit=lambda e, d: events.append((e, d)),
    )
    with patch.object(gateway_transport_chat, "gateway_approval_unavailable_reason", return_value=reason):
        for _ in range(sends):
            gateway_transport_chat._notify_approval_unavailable(turn)
    return events, session


def test_gateway_chat_warns_once_per_session_when_approvals_are_unsupported():
    events, session = _notices("unsupported")
    assert events == [("warning", {
        "type": "approval_gateway_unsupported",
        "message": "Approvals require a newer gateway. Upgrade the connected Hermes gateway to enable this.",
    })]
    assert session._approval_notice_emitted is True


def test_gateway_chat_warns_offline_when_capabilities_are_unreachable():
    events, _session = _notices("unreachable")
    assert [data["type"] for _event, data in events] == ["approval_gateway_offline"]


def test_gateway_chat_stays_silent_when_the_gateway_relays_approvals():
    events, session = _notices(None)
    assert events == []
    assert not hasattr(session, "_approval_notice_emitted")


def test_messages_js_handles_approval_gateway_unsupported_event():
    """Verify client-side warning handler recognizes the event type."""
    assert "d.type==='approval_gateway_unsupported'" in MESSAGES_JS


def test_messages_js_references_i18n_key_for_approval_gateway_unsupported():
    """Verify the i18n key is referenced in messages.js."""
    # The key should be used in the message handling logic
    assert "approval_gateway_unsupported" in MESSAGES_JS


def test_i18n_js_has_approval_gateway_unsupported_key():
    """Verify the i18n key exists in at least the English locale (first occurrence)."""
    assert "approval_gateway_unsupported: 'Approvals require a newer gateway" in I18N_JS
    lines = I18N_JS.split("\n")
    en_end = next(i for i, l in enumerate(lines) if l.strip().startswith("zh:"))
    found_in_english = any(
        "approval_gateway_unsupported:" in l
        for l in lines[:en_end]
    )
    assert found_in_english, "approval_gateway_unsupported key not found in English i18n locale"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
