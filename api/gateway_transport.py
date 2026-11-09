"""Transport abstraction of the Hermes Gateway bridge.

A Gateway-backed chat turn has one transport-independent lifecycle, owned by
``api.gateway_chat._run_gateway_chat_streaming``:

    admit the stream -> resolve session, profile config, endpoint, prefill and
    request options -> transport.run(turn) -> settle the outcome (cancelled /
    failed / empty / success writeback, goal hook, done) -> teardown.

Only ``transport.run`` differs between the ways WebUI can reach a Gateway.
Each transport turns a ``GatewayTurn`` into a ``TurnResult``, emitting the
turn's WebUI events through ``turn.emit`` on the way (it is the journaling
browser channel), and uses the shared layers below it:

    api.gateway_sse          SSE framing          (HTTP transports)
    api.gateway_events       Gateway event decoding (one decoder per dialect)
    api.gateway_projection   WebUI events + reconnect snapshot + checkpoint

Implementations:

* ``ChatCompletionsTransport`` (``api.gateway_transport_chat``) — stateless
  ``POST /v1/chat/completions`` stream. No run identity: it cannot be
  reattached, stopped remotely or steered.
* ``RunsApiTransport`` (``api.gateway_transport_runs``) — durable
  ``/v1/runs``: admission by idempotency key, ``/events`` replay from a
  checkpoint, durable status, reattach after a WebUI restart.
* ``SessionProtocolTransport`` (``api.gateway_sessions``) — the Agent's
  canonical session protocol over the profile's private socket, used to
  continue sessions that another client (CLI, TUI, Desktop, messaging) owns.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Callable

from api.config import gateway_supports_approval

Emit = Callable[[str, dict], None]


class TurnOutcomeUnknown(RuntimeError):
    """The transport lost the turn and cannot tell whether it ran.

    The pending turn is preserved so reopening the conversation recovers the
    original admission; the message must not be resent.
    """


@dataclass
class GatewayTurn:
    """Everything a transport needs to run one turn, resolved by the common worker."""

    session: Any
    session_id: str
    stream_id: str
    msg_text: str
    model: str | None
    model_provider: str | None
    workspace: Any
    attachments: list | None
    cfg: dict
    base_url: str
    api_key: str
    # System prompt + WebUI context rows sent ahead of the user turn.
    prefill_messages: list = field(default_factory=list)
    # Request fields shared by the HTTP transports: provider, reasoning, service tier.
    request_options: dict = field(default_factory=dict)
    # The journaling browser channel: every WebUI event of the turn goes here.
    emit: Emit = lambda _event, _data: None
    cancel_event: Any = None
    # Durable handoff record for transports that survive a WebUI restart:
    # ``record_run(run_id, request=None, **extra)``.
    record_run: Callable[..., None] | None = None
    # The saved ``session.gateway_run`` when this turn is reattached after a restart.
    reattach_run: dict | None = None


@dataclass
class TurnResult:
    """How a transport's turn ended, for the common settlement.

    ``text is None`` means the turn was cancelled; the transport already emitted
    the browser-facing cancel event. ``error`` (a message, or an apperror
    payload dict) fails the turn. ``canonical_snapshot`` is the Gateway's own
    transcript of the session, authoritative over the WebUI-built rows.
    """

    text: str | None
    usage: dict = field(default_factory=dict)
    error: str | dict | None = None
    canonical_snapshot: dict | None = None


class GatewayTransport(abc.ABC):
    """One way of running a WebUI turn on a Hermes Gateway."""

    #: Short name for logs and the durable run record.
    name: str = ""
    #: True when the transport publishes a run id that Stop, approvals and a
    #: restarted WebUI can address; False ends the stream's run-id wait at once.
    durable: bool = False

    @abc.abstractmethod
    def run(self, turn: GatewayTurn) -> TurnResult:
        """Run the turn to its outcome.

        Raise ``TurnOutcomeUnknown`` when the turn may have run but its outcome
        cannot be known; any other exception fails the turn with its message.
        """


def select_transport(session, cfg, *, base_url: str, api_key: str, reattach_run=None) -> GatewayTransport:
    """Pick the transport for one turn.

    A reattached run keeps the transport that admitted it. Otherwise an
    external session opted into the canonical session protocol uses it; the
    Runs API is used when enabled and the Gateway can relay approvals; every
    other turn uses Chat Completions.
    """
    from api import gateway_chat
    from api.gateway_sessions import SessionProtocolTransport, configured

    if (reattach_run or {}).get("transport") == "session" or configured(vars(session), cfg):
        return SessionProtocolTransport()
    if reattach_run or (
        gateway_chat._gateway_use_runs_api_enabled(cfg)
        and gateway_supports_approval(base_url, api_key)
    ):
        from api.gateway_transport_runs import RunsApiTransport

        return RunsApiTransport()
    from api.gateway_transport_chat import ChatCompletionsTransport

    return ChatCompletionsTransport()
