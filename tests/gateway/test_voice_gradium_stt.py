"""Tests for the Gradium STT client: setup handshake, audio encoding,
flush ids, rotation gating."""

from __future__ import annotations

import asyncio
import base64
import json
import sys

import pytest

from tests.gateway._voice_module_loader import load_voice_module

gradium_stt = load_voice_module("gradium_stt")

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
async def test_flush_ids_monotonic_across_rotation(fake_ws):
    """Flush ids are CALL-scoped, not socket-scoped: after a session
    rotation, new flush ids keep counting up, so a stale ack from the
    pre-rotation socket can never collide with a post-rotation wait
    (both sessions share one events queue)."""
    stt = gradium_stt.GradiumSTT("g-key")
    await stt.start()
    pre_rotation_ids = [await stt.flush(), await stt.flush()]
    assert pre_rotation_ids == [1, 2]
    first = stt._session
    first.total_duration_s = 250.0
    await stt.maybe_rotate({"vad": _vad(0.95)})
    assert stt._session is not first
    new_id = await stt.flush()
    assert new_id > max(pre_rotation_ids)
    # A stale pre-rotation ack can never satisfy a wait keyed on new_id.
    assert new_id not in pre_rotation_ids
    # The wire frames carried the monotonic ids — no per-socket restart.
    flush_frames = [m for m in fake_ws.sent if m.get("type") == "flush"]
    assert [f["flush_id"] for f in flush_frames] == [1, 2, 3]
    await stt.stop()


@pytest.mark.asyncio
async def test_watchdog_reconnects_dead_session(fake_ws, monkeypatch):
    """A session whose socket dies (e.g. the server's 300s wall-clock kill,
    observed live 2026-06-12) must be replaced automatically — without it
    the call goes permanently deaf."""
    monkeypatch.setattr(gradium_stt, "WATCHDOG_INTERVAL_S", 0.05)
    stt = gradium_stt.GradiumSTT("g-key")
    await stt.start()
    first = stt._session
    fake_ws.inbox.put_nowait(None)      # server closes: recv loop exits
    for _ in range(100):
        await asyncio.sleep(0.02)
        if stt._session is not first:
            break
    assert stt._session is not first
    assert first.closed is True
    await stt.stop()


@pytest.mark.asyncio
async def test_watchdog_rotates_on_wall_clock_age_without_steps(fake_ws, monkeypatch):
    """The server kills sessions on WALL CLOCK; maybe_rotate only sees step
    events (audio inflow). With no audio at all, the watchdog must still
    rotate before the kill."""
    monkeypatch.setattr(gradium_stt, "WATCHDOG_INTERVAL_S", 0.05)
    monkeypatch.setattr(gradium_stt, "HARD_ROTATE_AFTER_S", 0.1)
    stt = gradium_stt.GradiumSTT("g-key")
    await stt.start()
    first = stt._session
    assert first.total_duration_s == 0.0    # no audio was ever processed
    for _ in range(100):
        await asyncio.sleep(0.02)
        if stt._session is not first:
            break
    assert stt._session is not first
    assert first.closed is True
    await stt.stop()


@pytest.mark.asyncio
async def test_flush_noop_without_live_session(fake_ws):
    stt = gradium_stt.GradiumSTT("g-key")
    assert await stt.flush() is None        # never started
    await stt.start()
    assert await stt.flush() == 1
    await stt.stop()
    assert await stt.flush() is None        # closed
