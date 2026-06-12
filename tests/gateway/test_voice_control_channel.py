"""Tests for the voice control channel: SSE subscribe auth, event dispatch,
keepalive/malformed-line tolerance, reconnect-on-error, clean stop.

httpx never touches the network — the loaded module's `httpx` attribute is
replaced with a scripted fake.
"""

from __future__ import annotations

import asyncio
import sys

import pytest

from tests.gateway._voice_module_loader import load_voice_module

control_channel = load_voice_module("control_channel")


class FakeStreamResponse:
    def __init__(self, status_code, chunks, hang_after=True):
        self.status_code = status_code
        self._chunks = chunks
        self._hang_after = hang_after

    async def aiter_text(self):
        for chunk in self._chunks:
            yield chunk
        if self._hang_after:
            # Keep the connection "open" like a real idle SSE stream.
            await asyncio.Event().wait()


class _FakeStreamCM:
    def __init__(self, response):
        self._response = response

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, *exc):
        return False


class FakeAsyncClient:
    """Scripted httpx.AsyncClient stand-in. Each stream() call consumes the
    next scripted response; when the script runs dry it serves an idle
    (hanging) empty stream."""

    script: list = []
    instances: list = []

    def __init__(self, timeout=None):
        self.timeout = timeout
        self.closed = False
        self.stream_calls = []
        FakeAsyncClient.instances.append(self)

    def stream(self, method, url, headers=None, timeout=None):
        self.stream_calls.append(
            {"method": method, "url": url, "headers": headers})
        if FakeAsyncClient.script:
            response = FakeAsyncClient.script.pop(0)
        else:
            response = FakeStreamResponse(200, [])
        return _FakeStreamCM(response)

    async def aclose(self):
        self.closed = True


@pytest.fixture()
def fake_httpx(monkeypatch):
    FakeAsyncClient.script = []
    FakeAsyncClient.instances = []
    fake_module = type(sys)("httpx")
    fake_module.AsyncClient = FakeAsyncClient
    monkeypatch.setattr(control_channel, "httpx", fake_module)
    # Fast retries so reconnect paths are testable.
    monkeypatch.setattr(control_channel, "RETRY_INITIAL", 0.01)
    monkeypatch.setattr(control_channel, "RETRY_MAX", 0.05)
    return FakeAsyncClient


def _channel(events):
    async def on_event(event):
        events.append(event)

    return control_channel.ControlChannel(
        "http://control.test/", "sb_key_1", on_event)


async def _eventually(cond, timeout=2.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while not cond():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("condition not met within %.1fs" % timeout)
        await asyncio.sleep(0.01)


def test_requires_url_and_key():
    async def on_event(_e):
        return None

    with pytest.raises(ValueError):
        control_channel.ControlChannel("", "sb_key", on_event)
    with pytest.raises(ValueError):
        control_channel.ControlChannel("http://control.test", "", on_event)


@pytest.mark.asyncio
async def test_subscribes_with_bearer_and_dispatches_join_room(fake_httpx):
    fake_httpx.script = [FakeStreamResponse(200, [
        ": connected\n\n",
        'data: {"action":"join_room","roomUrl":"https://x.daily.co/r",'
        '"token":"t1"}\n\n',
    ])]
    events = []
    channel = _channel(events)
    channel.start()
    try:
        await _eventually(lambda: events)
        assert events == [{
            "action": "join_room",
            "roomUrl": "https://x.daily.co/r",
            "token": "t1",
        }]
        call = fake_httpx.instances[0].stream_calls[0]
        assert call["url"] == "http://control.test/api/agents/events"
        assert call["headers"]["Authorization"] == "Bearer sb_key_1"
        assert call["headers"]["Accept"] == "text/event-stream"
    finally:
        await channel.stop()


@pytest.mark.asyncio
async def test_tolerates_keepalives_split_chunks_and_bad_json(fake_httpx):
    fake_httpx.script = [FakeStreamResponse(200, [
        ": keepalive\n\n",
        "data: this-is-not-json\n\n",
        'data: {"no_action_key": 1}\n\n',
        # one event split across two chunks exercises line buffering
        'data: {"action":"leave',
        '_room"}\n\n',
    ])]
    events = []
    channel = _channel(events)
    channel.start()
    try:
        await _eventually(lambda: events)
        assert events == [{"action": "leave_room"}]
    finally:
        await channel.stop()


@pytest.mark.asyncio
async def test_http_error_status_retries_until_success(fake_httpx):
    fake_httpx.script = [
        FakeStreamResponse(401, [], hang_after=False),
        FakeStreamResponse(200, ['data: {"action":"leave_room"}\n\n']),
    ]
    events = []
    channel = _channel(events)
    channel.start()
    try:
        await _eventually(lambda: events)
        assert events == [{"action": "leave_room"}]
        assert len(fake_httpx.instances[0].stream_calls) >= 2
    finally:
        await channel.stop()


@pytest.mark.asyncio
async def test_connection_exception_retries(fake_httpx):
    class ExplodingClient(fake_httpx):
        def stream(self, method, url, headers=None, timeout=None):
            self.stream_calls.append(
                {"method": method, "url": url, "headers": headers})
            if len(self.stream_calls) == 1:
                raise ConnectionError("network down")
            return super().stream(method, url, headers=headers, timeout=timeout)

    fake_httpx.script = [
        FakeStreamResponse(200, ['data: {"action":"leave_room"}\n\n']),
    ]
    events = []

    async def on_event(event):
        events.append(event)

    channel = control_channel.ControlChannel(
        "http://control.test", "sb_key_1", on_event)
    fake_module = type(sys)("httpx")
    fake_module.AsyncClient = ExplodingClient
    control_channel.httpx = fake_module
    channel.start()
    try:
        await _eventually(lambda: events)
        assert events == [{"action": "leave_room"}]
        assert len(fake_httpx.instances[0].stream_calls) >= 2
    finally:
        await channel.stop()


@pytest.mark.asyncio
async def test_handler_exception_does_not_kill_stream(fake_httpx):
    fake_httpx.script = [FakeStreamResponse(200, [
        'data: {"action":"join_room","roomUrl":"u","token":"t"}\n\n',
        'data: {"action":"leave_room"}\n\n',
    ])]
    events = []

    async def on_event(event):
        events.append(event)
        if event["action"] == "join_room":
            raise RuntimeError("handler blew up")

    channel = control_channel.ControlChannel(
        "http://control.test", "sb_key_1", on_event)
    channel.start()
    try:
        await _eventually(lambda: len(events) == 2)
        assert events[1] == {"action": "leave_room"}
    finally:
        await channel.stop()


@pytest.mark.asyncio
async def test_stop_cancels_idle_stream_and_closes_client(fake_httpx):
    fake_httpx.script = [FakeStreamResponse(200, [": connected\n\n"])]
    events = []
    channel = _channel(events)
    channel.start()
    await asyncio.sleep(0.05)          # let it connect and idle
    await channel.stop()
    assert fake_httpx.instances[0].closed is True
    assert channel._task.done()
