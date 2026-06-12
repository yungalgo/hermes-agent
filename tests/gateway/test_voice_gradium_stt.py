"""Tests for the Gradium STT client: setup handshake, audio encoding,
flush ids, rotation gating."""

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


gradium_stt = _load("gradium_stt")

# Live-verified (2026-06-12): the FIRST server message on every ASR socket is
# a "ready" frame with this shape. Fixtures inject it on every connect so the
# client's tolerance of it is always under test.
READY_MSG = {
    "type": "ready",
    "sample_rate": 24000,
    "frame_size": 1920,
    "delay_in_frames": 10,
}

# Live-verified: step.vad carries FOUR horizons (0.5/1.0/2.0/3.0). Fixtures
# order them with 0.5 LAST so any index-0 lookup reads the wrong horizon and
# fails the assertions below (lookups must be by horizon_s value).


def _vad(p05: float, p10: float = 0.5, p20: float = 0.5, p30: float = 0.5):
    return [
        {"horizon_s": 1.0, "inactivity_prob": p10},
        {"horizon_s": 2.0, "inactivity_prob": p20},
        {"horizon_s": 3.0, "inactivity_prob": p30},
        {"horizon_s": 0.5, "inactivity_prob": p05},
    ]


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
        ws.inbox.put_nowait(dict(READY_MSG))   # server greets every socket
        return ws

    fake_module = type(sys)("websockets")
    fake_module.connect = fake_connect
    monkeypatch.setitem(sys.modules, "websockets", fake_module)
    return ws


@pytest.mark.asyncio
async def test_setup_and_audio_b64(fake_ws):
    stt = gradium_stt.GradiumSTT("g-key")
    await stt.start()
    assert fake_ws.headers == {"x-api-key": "g-key"}
    assert fake_ws.sent[0] == {
        "type": "setup", "model_name": "default", "input_format": "pcm"}
    await stt.send_audio(b"\x01\x02")
    assert fake_ws.sent[1]["type"] == "audio"
    assert base64.b64decode(fake_ws.sent[1]["audio"]) == b"\x01\x02"
    await stt.stop()


@pytest.mark.asyncio
async def test_flush_ids_increment(fake_ws):
    stt = gradium_stt.GradiumSTT("g-key")
    await stt.start()
    assert await stt.flush() == 1
    assert await stt.flush() == 2
    await stt.stop()


@pytest.mark.asyncio
async def test_no_rotation_before_threshold_or_during_speech(fake_ws):
    stt = gradium_stt.GradiumSTT("g-key")
    await stt.start()
    stt._session.total_duration_s = 100.0
    await stt.maybe_rotate({"vad": _vad(0.99, 0.99, 0.99, 0.99)})
    first = stt._session
    stt._session.total_duration_s = 250.0
    # 0.5s horizon says SPEAKING (0.1) while every other horizon is over the
    # rotate threshold — an index-0 lookup would wrongly rotate here.
    await stt.maybe_rotate({"vad": _vad(0.1, 0.95, 0.9, 0.9)})
    assert stt._session is first     # speaking: no rotate
    await stt.stop()


@pytest.mark.asyncio
async def test_rotation_during_silence_past_threshold(fake_ws):
    stt = gradium_stt.GradiumSTT("g-key")
    await stt.start()
    first = stt._session
    first.total_duration_s = 250.0
    # 0.5s horizon says SILENT (0.95) while vad[0] (1.0s horizon) is under the
    # threshold — an index-0 lookup would wrongly skip this rotation.
    await stt.maybe_rotate({"vad": _vad(0.95, 0.3, 0.4, 0.5)})
    assert stt._session is not first
    assert first.closed is True
    await stt.stop()


@pytest.mark.asyncio
async def test_events_yields_server_messages(fake_ws):
    stt = gradium_stt.GradiumSTT("g-key")
    await stt.start()
    await fake_ws.inbox.put(
        {"type": "step", "vad": _vad(0.2), "total_duration_s": 0.08})
    await fake_ws.inbox.put({"type": "text", "text": "hi", "start_s": 0.0})
    events = stt.events()
    # The leading server "ready" frame passes through without breaking the
    # client; downstream consumers ignore unknown types.
    first = await asyncio.wait_for(events.__anext__(), timeout=2)
    second = await asyncio.wait_for(events.__anext__(), timeout=2)
    third = await asyncio.wait_for(events.__anext__(), timeout=2)
    assert first == READY_MSG
    assert second["type"] == "step"
    assert stt._session.total_duration_s == 0.08
    assert third == {"type": "text", "text": "hi", "start_s": 0.0}
    await stt.stop()


@pytest.mark.asyncio
async def test_flush_noop_without_live_session(fake_ws):
    stt = gradium_stt.GradiumSTT("g-key")
    assert await stt.flush() is None        # never started
    await stt.start()
    assert await stt.flush() == 1
    await stt.stop()
    assert await stt.flush() is None        # closed
