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


class FakeEventHandler:
    """Stands in for daily.EventHandler (the SDK base class the transport
    subclasses for participant/call-state events)."""

    def __init__(self):
        pass


class FakeCallClient:
    # "ok" -> join succeeds; "error" -> completion gets an error;
    # "hang" -> completion never fires (join timeout path).
    join_behavior = "ok"
    # participants() roster returned after join (overridden per-test to
    # simulate a human already waiting in the room).
    initial_participants = {
        "local": {"id": "agent-id", "info": {"isLocal": True}}}

    def __init__(self, event_handler=None):
        self.event_handler = event_handler
        self.join_args = None
        self.subscription_profiles = None
        self.left = False
        self.released = False

    def participants(self):
        return dict(self.initial_participants)

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
    fake_module.EventHandler = FakeEventHandler
    monkeypatch.setitem(sys.modules, "daily", fake_module)
    monkeypatch.setattr(FakeCallClient, "join_behavior", "ok")
    monkeypatch.setattr(FakeCallClient, "initial_participants",
                        FakeCallClient.initial_participants)
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
        # Real frames arrive in order; DTX keep-alive silence (all-zero
        # chunks) may interleave when the queue runs dry.
        await _eventually(
            lambda: [c for c in received if any(c)] == [b"\x0a" * 16,
                                                        b"\x0b" * 16])
    finally:
        await transport.leave()


@pytest.mark.asyncio
async def test_read_loop_feeds_silence_during_dtx_gaps(fake_daily):
    """A silent caller (WebRTC DTX) must not starve the ASR stream: the
    reader synthesizes 80ms silence chunks at cadence while read_frames
    returns nothing."""
    received = []

    async def on_audio_in(pcm: bytes) -> None:
        received.append(pcm)

    transport = _make_transport(on_audio_in)
    await transport.join("https://x.daily.co/room", "tok-1")
    try:
        # No frames pushed at all — only keep-alive silence should flow.
        await _eventually(lambda: len(received) >= 3, timeout=3.0)
        assert all(not any(c) for c in received)
        assert all(len(c) == daily_transport.IN_CHUNK_FRAMES * 2
                   for c in received)
    finally:
        await transport.leave()


@pytest.mark.asyncio
async def test_leave_tears_down_client_and_threads(fake_daily):
    transport = _make_transport()
    await transport.join("https://x.daily.co/room", "tok-1")
    client = transport._client
    reader, writer = transport._reader, transport._writer
    keepalive = transport._keepalive
    await transport.leave()
    assert client.left is True
    assert client.released is True
    assert transport._client is None
    # The keep-alive thread feeds BILLABLE ASR silence — it must never
    # outlive the call (ENG-555 cost leak).
    await _eventually(
        lambda: (not reader.is_alive() and not writer.is_alive()
                 and not keepalive.is_alive()), timeout=3.0)


# ---------------------------------------------------------------------------
# Presence + call-state events (ENG-555 abandoned-call watchdog inputs)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_join_wires_event_handler_into_call_client(fake_daily):
    transport = _make_transport()
    await transport.join("https://x.daily.co/room", "tok-1")
    try:
        handler = transport._client.event_handler
        assert handler is not None
        # The handler is a daily.EventHandler subclass (SDK requirement).
        assert isinstance(handler, FakeEventHandler)
    finally:
        await transport.leave()


@pytest.mark.asyncio
async def test_remote_participant_count_tracks_join_and_leave(fake_daily):
    transport = _make_transport()
    await transport.join("https://x.daily.co/room", "tok-1")
    try:
        handler = transport._client.event_handler
        assert transport.remote_participant_count == 0
        handler.on_participant_joined(
            {"id": "human-1", "info": {"isLocal": False, "userName": "yung"}})
        assert transport.remote_participant_count == 1
        # The agent's own (local) entry never counts as a human.
        handler.on_participant_joined(
            {"id": "agent-id", "info": {"isLocal": True}})
        assert transport.remote_participant_count == 1
        # Duplicate join events do not double-count.
        handler.on_participant_joined(
            {"id": "human-1", "info": {"isLocal": False}})
        assert transport.remote_participant_count == 1
        handler.on_participant_left(
            {"id": "human-1", "info": {"isLocal": False}}, "leftCall")
        assert transport.remote_participant_count == 0
        # A leave for an unknown id is harmless.
        handler.on_participant_left({"id": "ghost"}, "leftCall")
        assert transport.remote_participant_count == 0
    finally:
        await transport.leave()


@pytest.mark.asyncio
async def test_presence_seeded_from_post_join_roster(fake_daily, monkeypatch):
    """A human already waiting in the room when the agent joins (their
    joined event fired before our handler existed) must be counted."""
    monkeypatch.setattr(FakeCallClient, "initial_participants", {
        "local": {"id": "agent-id", "info": {"isLocal": True}},
        "human-7": {"id": "human-7", "info": {"isLocal": False}},
    })
    transport = _make_transport()
    await transport.join("https://x.daily.co/room", "tok-1")
    try:
        assert transport.remote_participant_count == 1
    finally:
        await transport.leave()


@pytest.mark.asyncio
async def test_remote_left_state_marks_abnormal_end(fake_daily):
    """Room expiry / ejection: the call state flips to "left" WITHOUT us
    calling leave(). The transport must flag it for the adapter watchdog."""
    transport = _make_transport()
    await transport.join("https://x.daily.co/room", "tok-1")
    try:
        handler = transport._client.event_handler
        assert transport.abnormal_end is None
        handler.on_call_state_updated("joined")
        assert transport.abnormal_end is None
        handler.on_call_state_updated("left")
        assert transport.abnormal_end == "left"
    finally:
        await transport.leave()


@pytest.mark.asyncio
async def test_local_leave_left_state_is_not_abnormal(fake_daily):
    transport = _make_transport()
    await transport.join("https://x.daily.co/room", "tok-1")
    handler = transport._client.event_handler
    await transport.leave()
    # The "left" state caused by OUR leave() must not look like an ejection.
    handler.on_call_state_updated("left")
    assert transport.abnormal_end is None


@pytest.mark.asyncio
async def test_client_error_marks_abnormal_end(fake_daily):
    transport = _make_transport()
    await transport.join("https://x.daily.co/room", "tok-1")
    try:
        transport._client.event_handler.on_error("connection lost")
        assert transport.abnormal_end == "error: connection lost"
    finally:
        await transport.leave()


# ---------------------------------------------------------------------------
# Keep-alive teardown ordering (ENG-555: silence into the ASR is billable)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_begin_teardown_stops_keepalive_silence_immediately(fake_daily):
    received = []

    async def on_audio_in(pcm: bytes) -> None:
        received.append(pcm)

    transport = _make_transport(on_audio_in)
    await transport.join("https://x.daily.co/room", "tok-1")
    try:
        # Keep-alive silence is flowing (DTX gap, no real frames).
        await _eventually(lambda: len(received) >= 3, timeout=3.0)
        keepalive = transport._keepalive
        transport.begin_teardown()
        # The keep-alive thread exits promptly — it must not keep feeding
        # billable silence while the (slow) teardown awaits run.
        await _eventually(lambda: not keepalive.is_alive(), timeout=2.0)
        fed_after_teardown = len(received)
        await asyncio.sleep(0.3)
        assert len(received) == fed_after_teardown
    finally:
        await transport.leave()


@pytest.mark.asyncio
async def test_begin_teardown_is_idempotent_and_leave_still_works(fake_daily):
    transport = _make_transport()
    await transport.join("https://x.daily.co/room", "tok-1")
    transport.begin_teardown()
    transport.begin_teardown()
    await transport.leave()
    assert transport._client is None
