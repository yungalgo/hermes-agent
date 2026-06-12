"""Tests for the Daily transport: device geometry, join settings, outbound
audio ordering, barge-in queue clear, inbound frame forwarding.

The real daily-python SDK never loads here — a fake ``daily`` module is
injected into sys.modules (unit tests cannot open real WebRTC calls).
"""

from __future__ import annotations

import asyncio
import queue
import sys

import pytest

from tests.gateway._voice_module_loader import load_voice_module

daily_transport = load_voice_module("daily_transport")


class FakeMic:
    def __init__(self, name, sample_rate, channels, non_blocking):
        self.name = name
        self.sample_rate = sample_rate
        self.channels = channels
        self.non_blocking = non_blocking
        self.written = []

    def write_frames(self, frames, completion=None):
        self.written.append(frames)
        return len(frames) // 2


class FakeSpeaker:
    def __init__(self, name, sample_rate, channels, non_blocking):
        self.name = name
        self.sample_rate = sample_rate
        self.channels = channels
        self.non_blocking = non_blocking
        self.frames_q: "queue.Queue[bytes]" = queue.Queue()

    def read_frames(self, num_frames, completion=None):
        try:
            return self.frames_q.get(timeout=0.05)
        except queue.Empty:
            return b""


class FakeCallClient:
    # "ok" -> join succeeds; "error" -> completion gets an error;
    # "hang" -> completion never fires (join timeout path).
    join_behavior = "ok"

    def __init__(self, event_handler=None):
        self.event_handler = event_handler
        self.join_args = None
        self.subscription_profiles = None
        self.left = False
        self.released = False

    def update_subscription_profiles(self, profile_settings, completion=None):
        self.subscription_profiles = profile_settings
        if completion:
            completion(None)

    def join(self, meeting_url, meeting_token=None, client_settings=None,
             completion=None):
        self.join_args = {
            "meeting_url": meeting_url,
            "meeting_token": meeting_token,
            "client_settings": client_settings,
        }
        if self.join_behavior == "ok":
            completion({"participants": {}}, None)
        elif self.join_behavior == "error":
            completion(None, "boom")
        # "hang": never call completion

    def leave(self, completion=None):
        self.left = True
        if completion:
            completion(None)

    def release(self):
        self.released = True


@pytest.fixture()
def fake_daily(monkeypatch):
    """Fresh fake `daily` module + reset transport process-level globals."""

    class FakeDaily:
        init_calls = []
        mics = []
        speakers = []
        selected = []

        @staticmethod
        def init(*args, **kwargs):
            FakeDaily.init_calls.append((args, kwargs))

        @staticmethod
        def create_microphone_device(device_name, sample_rate=16000,
                                     channels=1, non_blocking=False):
            mic = FakeMic(device_name, sample_rate, channels, non_blocking)
            FakeDaily.mics.append(mic)
            return mic

        @staticmethod
        def create_speaker_device(device_name, sample_rate=16000,
                                  channels=1, non_blocking=False):
            spk = FakeSpeaker(device_name, sample_rate, channels, non_blocking)
            FakeDaily.speakers.append(spk)
            return spk

        @staticmethod
        def select_speaker_device(device_name):
            FakeDaily.selected.append(device_name)

    fake_module = type(sys)("daily")
    fake_module.Daily = FakeDaily
    fake_module.CallClient = FakeCallClient
    monkeypatch.setitem(sys.modules, "daily", fake_module)
    monkeypatch.setattr(FakeCallClient, "join_behavior", "ok")
    monkeypatch.setattr(daily_transport, "_initialized", False)
    monkeypatch.setattr(daily_transport, "_mic", None)
    monkeypatch.setattr(daily_transport, "_speaker", None)
    return FakeDaily


async def _eventually(cond, timeout=2.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while not cond():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("condition not met within %.1fs" % timeout)
        await asyncio.sleep(0.02)


async def _noop_audio(_pcm: bytes) -> None:
    return None


def _make_transport(on_audio_in=_noop_audio):
    loop = asyncio.get_running_loop()
    return daily_transport.DailyTransport(loop, on_audio_in)


def test_ensure_daily_idempotent_and_device_geometry(fake_daily):
    daily_transport._ensure_daily()
    daily_transport._ensure_daily()
    assert len(fake_daily.init_calls) == 1
    assert len(fake_daily.mics) == 1
    assert len(fake_daily.speakers) == 1
    mic, spk = fake_daily.mics[0], fake_daily.speakers[0]
    # 48 kHz mic = Gradium TTS passthrough; 24 kHz speaker = Gradium ASR feed.
    assert (mic.name, mic.sample_rate, mic.channels) == (
        daily_transport.MIC_DEVICE, 48000, 1)
    assert (spk.name, spk.sample_rate, spk.channels) == (
        daily_transport.SPEAKER_DEVICE, 24000, 1)
    assert fake_daily.selected == [daily_transport.SPEAKER_DEVICE]


@pytest.mark.asyncio
async def test_join_passes_token_mic_settings_and_subscribes_audio(fake_daily):
    transport = _make_transport()
    await transport.join("https://x.daily.co/room", "tok-1")
    try:
        client = transport._client
        assert client.join_args["meeting_url"] == "https://x.daily.co/room"
        assert client.join_args["meeting_token"] == "tok-1"
        inputs = client.join_args["client_settings"]["inputs"]
        assert inputs["camera"] is False
        assert inputs["microphone"]["isEnabled"] is True
        assert (inputs["microphone"]["settings"]["deviceId"]
                == daily_transport.MIC_DEVICE)
        # Caller audio must be subscribed (camera never).
        assert client.subscription_profiles == {
            "base": {"camera": "unsubscribed", "microphone": "subscribed"}}
    finally:
        await transport.leave()


@pytest.mark.asyncio
async def test_join_raises_on_completion_error(fake_daily, monkeypatch):
    monkeypatch.setattr(FakeCallClient, "join_behavior", "error")
    transport = _make_transport()
    with pytest.raises(RuntimeError, match="Daily join failed: boom"):
        await transport.join("https://x.daily.co/room", "tok-1")


@pytest.mark.asyncio
async def test_join_raises_on_timeout(fake_daily, monkeypatch):
    monkeypatch.setattr(FakeCallClient, "join_behavior", "hang")
    transport = _make_transport()
    with pytest.raises(RuntimeError, match="timed out"):
        await transport.join("https://x.daily.co/room", "tok-1", timeout=0.2)


@pytest.mark.asyncio
async def test_send_audio_reaches_mic_in_order(fake_daily):
    transport = _make_transport()
    await transport.join("https://x.daily.co/room", "tok-1")
    try:
        await transport.send_audio(b"\x01" * 10)
        await transport.send_audio(b"\x02" * 10)
        mic = fake_daily.mics[0]
        await _eventually(lambda: len(mic.written) == 2)
        assert mic.written == [b"\x01" * 10, b"\x02" * 10]
    finally:
        await transport.leave()


@pytest.mark.asyncio
async def test_clear_output_drops_queued_chunks(fake_daily):
    # No join -> no writer thread; the queue is inspected directly.
    transport = _make_transport()
    await transport.send_audio(b"\x01" * 10)
    await transport.send_audio(b"\x02" * 10)
    transport.clear_output()
    assert transport._out_q.empty()


@pytest.mark.asyncio
async def test_read_loop_forwards_frames_to_on_audio_in(fake_daily):
    received = []

    async def on_audio_in(pcm: bytes) -> None:
        received.append(pcm)

    transport = _make_transport(on_audio_in)
    await transport.join("https://x.daily.co/room", "tok-1")
    try:
        spk = fake_daily.speakers[0]
        spk.frames_q.put(b"\x0a" * 16)
        spk.frames_q.put(b"\x0b" * 16)
        await _eventually(lambda: received == [b"\x0a" * 16, b"\x0b" * 16])
    finally:
        await transport.leave()


@pytest.mark.asyncio
async def test_leave_tears_down_client_and_threads(fake_daily):
    transport = _make_transport()
    await transport.join("https://x.daily.co/room", "tok-1")
    client = transport._client
    reader, writer = transport._reader, transport._writer
    await transport.leave()
    assert client.left is True
    assert client.released is True
    assert transport._client is None
    await _eventually(
        lambda: not reader.is_alive() and not writer.is_alive(), timeout=3.0)
