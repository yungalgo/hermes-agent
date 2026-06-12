"""Tests for the Gradium per-turn TTS client: setup shape, text/filler
sends, audio delivery, end-of-turn drain, barge-in abort."""

from __future__ import annotations

import asyncio
import base64
import importlib.util
import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load(name: str):
    mod_name = f"voice_plugin_{name}"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    path = _REPO_ROOT / "plugins" / "platforms" / "voice" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


gradium_tts = _load("gradium_tts")


class FakeWS:
    def __init__(self):
        self.sent = []
        self.inbox = asyncio.Queue()
        self.closed = False

    async def send(self, data):
        self.sent.append(json.loads(data))

    async def close(self):
        self.closed = True

    def __aiter__(self):
        return self

    async def __anext__(self):
        item = await self.inbox.get()
        if item is None:
            raise StopAsyncIteration
        return json.dumps(item)


@pytest.fixture()
def fake_ws(monkeypatch):
    ws = FakeWS()

    async def fake_connect(url, additional_headers=None):
        ws.url = url
        ws.headers = additional_headers
        return ws

    fake_module = type(sys)("websockets")
    fake_module.connect = fake_connect
    monkeypatch.setitem(sys.modules, "websockets", fake_module)
    return ws


async def _noop_audio(_pcm: bytes) -> None:
    return None


@pytest.mark.asyncio
async def test_setup_without_voice_id(fake_ws):
    turn = gradium_tts.GradiumTTSTurn("g-key", "", _noop_audio)
    await turn.open()
    assert fake_ws.headers == {"x-api-key": "g-key"}
    assert fake_ws.sent[0] == {
        "type": "setup", "model_name": "default", "output_format": "pcm"}
    assert "voice_id" not in fake_ws.sent[0]
    await turn.abort()


@pytest.mark.asyncio
async def test_setup_includes_voice_id_when_set(fake_ws):
    turn = gradium_tts.GradiumTTSTurn("g-key", "voice-123", _noop_audio)
    await turn.open()
    assert fake_ws.sent[0]["voice_id"] == "voice-123"
    assert fake_ws.sent[0]["output_format"] == "pcm"
    await turn.abort()


@pytest.mark.asyncio
async def test_send_text_shape_and_char_count(fake_ws):
    turn = gradium_tts.GradiumTTSTurn("g-key", "", _noop_audio)
    await turn.open()
    await turn.send_text("Hello there.")
    assert fake_ws.sent[1] == {"type": "text", "text": "Hello there."}
    assert turn.chars_sent == len("Hello there.")
    await turn.abort()


@pytest.mark.asyncio
async def test_send_filler_appends_flush_tag(fake_ws):
    turn = gradium_tts.GradiumTTSTurn("g-key", "", _noop_audio)
    await turn.open()
    await turn.send_filler("One moment.")
    assert fake_ws.sent[1] == {"type": "text", "text": "One moment. <flush>"}
    await turn.abort()


@pytest.mark.asyncio
async def test_audio_messages_invoke_on_audio_decoded(fake_ws):
    received = []

    async def on_audio(pcm: bytes) -> None:
        received.append(pcm)

    turn = gradium_tts.GradiumTTSTurn("g-key", "", on_audio)
    await turn.open()
    pcm = b"\x00\x01\x02\x03"
    await fake_ws.inbox.put(
        {"type": "audio", "audio": base64.b64encode(pcm).decode("ascii")})
    await fake_ws.inbox.put({"type": "end_of_stream"})
    await asyncio.wait_for(turn.end(), timeout=2)
    assert received == [pcm]


@pytest.mark.asyncio
async def test_end_returns_after_server_end_of_stream(fake_ws):
    turn = gradium_tts.GradiumTTSTurn("g-key", "", _noop_audio)
    await turn.open()
    await turn.send_text("Bye.")
    await fake_ws.inbox.put({"type": "end_of_stream"})
    await asyncio.wait_for(turn.end(), timeout=2)
    # end() sent the client-side end_of_stream before waiting
    assert {"type": "end_of_stream"} in fake_ws.sent
    assert fake_ws.closed is True


@pytest.mark.asyncio
async def test_abort_closes_socket_and_gates_send_text(fake_ws):
    turn = gradium_tts.GradiumTTSTurn("g-key", "", _noop_audio)
    await turn.open()
    await turn.abort()
    assert fake_ws.closed is True
    sent_before = list(fake_ws.sent)
    await turn.send_text("should be dropped")
    assert fake_ws.sent == sent_before
    # end() after abort is a no-op and must not hang
    await asyncio.wait_for(turn.end(), timeout=2)
