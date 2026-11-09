"""Layer 1 of the Hermes Gateway bridge: Server-Sent Events wire framing.

This module turns the bytes of a Gateway HTTP response into ``SseFrame``
objects. It knows the SSE line grammar (``event:``, ``data:``, comments, blank
separators), the Gateway read-timeout budget and how a read timeout interacts
with the user's Stop. It knows nothing about Hermes event names, JSON payloads,
WebUI events or stream state; those belong to ``api.gateway_events`` (layer 2)
and ``api.gateway_projection`` (layer 3).

Layering of the Gateway bridge::

    HTTP response bytes
      -> gateway_sse.iter_sse_frames()          SseFrame(event, data)
      -> gateway_events.decode_*_frame()        GatewayFrame(seq, events)
      -> gateway_projection.GatewayEventProjector
                                                WebUI events + LiveTurnSnapshot
      -> gateway_chat (run orchestration: reconnect, watchdog, status,
                       approvals, terminal settlement and persistence)
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, Iterator

READ_TIMEOUT_ENV = "HERMES_WEBUI_GATEWAY_READ_TIMEOUT"
READ_TIMEOUT_DEFAULT = 600.0

DONE_SENTINEL = "[DONE]"


def read_timeout_secs() -> float:
    """Total byte-silence budget for Gateway SSE reads (default 600s, env-tunable)."""
    raw = os.environ.get(READ_TIMEOUT_ENV)
    if raw:
        try:
            val = float(raw)
            if val > 0:
                return val
        except (TypeError, ValueError):
            pass
    return READ_TIMEOUT_DEFAULT


def iter_lines_cancellable(resp, cancel_event) -> Iterator[bytes]:
    """Yield raw SSE lines from ``resp``, unblocking cleanly on a read timeout.

    ``resp``'s socket carries a read timeout (``urlopen(timeout=...)``). A read
    that blocks past it raises ``socket.timeout``, and that timeout is TERMINAL:
    CPython's ``socket.makefile`` latches ``_timeout_occurred`` on the first
    timeout, so every subsequent read raises a bare ``OSError`` ("cannot read
    from timed out object") — there is no multi-read grace to reclaim. So on a
    read timeout (or the poisoned-socket ``OSError``/any read error) this either
    surfaces the user's Stop or tears the stalled turn down:

      - cancel set -> yield ``b""`` (the caller's ``if cancel_event.is_set()``
        branch emits its cancel event), then stop. Stop is honored within one
        timeout window instead of waiting for a stalled gateway to error.
      - otherwise -> re-raise, so the caller's error handling reports the stall.

    A stream that keeps emitting bytes within the timeout window never trips it,
    so a genuinely alive (if slow) token stream is untouched. Emitting ``b""`` is
    safe: the framer decodes it to a blank (payload-less) frame.

    Iterates ``resp`` via the iterator protocol so a real ``HTTPResponse`` and the
    test fakes (which implement ``__iter__``) behave identically.
    """
    resp_iter = iter(resp)
    while True:
        try:
            raw_line = next(resp_iter)
        except StopIteration:
            return  # EOF
        except OSError:
            # socket.timeout / TimeoutError are OSError subclasses, as is the
            # post-timeout poisoned-socket "cannot read" error. All are terminal
            # for this connection.
            if cancel_event.is_set():
                yield b""  # let the caller emit its cancel event
                return
            raise
        yield raw_line


@dataclass(frozen=True)
class SseFrame:
    """One SSE line, classified.

    ``data`` is the text of a ``data:`` line, or ``None`` for lines that carry
    no payload (blank separators, ``event:``/``id:`` fields and ``:`` comment
    keepalives). Payload-less frames are still yielded because they are
    liveness signals: the Runs API watchdog treats a connection that delivers
    only those for too long as stalled.

    ``event`` is the SSE event name in effect for this line (``"message"``
    unless an ``event:`` field set it).
    """

    event: str = "message"
    data: str | None = None

    @property
    def has_payload(self) -> bool:
        return self.data is not None

    @property
    def is_done(self) -> bool:
        """OpenAI-compatible end-of-stream marker (``data: [DONE]``)."""
        return self.data == DONE_SENTINEL


def iter_sse_frames(lines: Iterable[bytes | str]) -> Iterator[SseFrame]:
    """Frame raw SSE lines.

    Every ``data:`` line is dispatched as its own frame (Gateway payloads are
    single-line JSON, and a stream that omits blank separators must still
    relay each line). An ``event:`` field names the data line(s) that follow;
    the name resets to ``"message"`` after a data line is dispatched and on a
    blank separator, so one named event never leaks onto the next payload.
    """
    event = "message"
    for raw in lines:
        if isinstance(raw, bytes):
            line = raw.decode("utf-8", errors="replace").strip()
        else:
            line = str(raw).strip()
        if line.startswith("data:"):
            yield SseFrame(event, line[5:].strip())
            event = "message"
            continue
        if not line:
            event = "message"
        elif line.startswith("event:"):
            event = line[6:].strip() or "message"
        yield SseFrame(event, None)
