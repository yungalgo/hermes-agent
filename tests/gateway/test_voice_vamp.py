"""Tests for the vamp layer (perceived latency, notes §16): VampCache
clip selection, turn-loop vamp scheduling (energy + VAD triggers, skip
logic, false-fire cancel), per-turn telemetry shape, the voice-turn model
override, and the configurable energy barge-in chunk count."""

from __future__ import annotations

import asyncio
import json
import time

import pytest

from tests.gateway._voice_module_loader import load_voice_module
from tests.gateway.test_voice_turn_loop import (
    GREETING_REPLY,
    LOUD_CHUNK,
    QUIET_CHUNK,
    FakeSTT,
    FakeTTSFactory,
    FakeTransport,
    _eventually,
    install_agents,
    say,
    speak_then_block,
)

turn_loop = load_voice_module("turn_loop")
vamp_mod = load_voice_module("vamp")


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

CLIP_PCM = b"\x01\x00" * 7680          # 15360 bytes = two 80ms chunks @48k


class FakeVamp:
    def __init__(self, clips=None, ready=True):
        self._ready = ready
        self._clips = clips if clips is not None else [("Right.", CLIP_PCM)]
        self.picks = []

    @property
    def ready(self):
        return self._ready

    def pick(self):
        if not self._clips:
            return None
        text, pcm = self._clips[0]
        self.picks.append(text)
        return text, pcm


class MarkedTransport(FakeTransport):
    """FakeTransport with the telemetry write-mark surface the real
    DailyTransport grew (reset_write_mark / first_write_t)."""

    def __init__(self):
        super().__init__()
        self.first_write_t = None
        self._armed = False
        self.resets = 0

    def reset_write_mark(self):
        self.resets += 1
        self.first_write_t = None
        self._armed = True

    async def send_audio(self, pcm):
        if self._armed:
            self._armed = False
            self.first_write_t = time.monotonic()
        await super().send_audio(pcm)


def _make_loop(stt, factory, transport, extra=None, vamp=None):
    return turn_loop.VoiceTurnLoop(
        stt, factory, transport, extra=extra or {}, vamp=vamp)


def _telemetry_records(caplog):
    out = []
    for rec in caplog.records:
        msg = rec.getMessage()
        if msg.startswith("voice/telemetry "):
            out.append(json.loads(msg[len("voice/telemetry "):]))
    return out


async def _await_listening_after_greeting(vloop):
    await _eventually(lambda: vloop._state == turn_loop.LISTENING
                      and len(vloop._history) == 1)


# ---------------------------------------------------------------------------
# Pure config resolution
# ---------------------------------------------------------------------------


def test_resolve_turn_model():
    assert turn_loop._resolve_turn_model({}) is None
    assert turn_loop._resolve_turn_model({"model": ""}) is None
    assert turn_loop._resolve_turn_model({"model": "   "}) is None
    assert turn_loop._resolve_turn_model(
        {"model": "claude-haiku-4-5-20251001"}) == "claude-haiku-4-5-20251001"


def test_resolve_vamp_trigger():
    assert turn_loop._resolve_vamp_trigger({}) == "energy"
    assert turn_loop._resolve_vamp_trigger({"vamp_trigger": "vad"}) == "vad"
    assert turn_loop._resolve_vamp_trigger({"vamp_trigger": "off"}) == "off"
    assert turn_loop._resolve_vamp_trigger({"vamp_trigger": "bogus"}) == "energy"


def test_resolve_int_extra():
    assert turn_loop._resolve_int_extra({}, "barge_energy_chunks", 2) == 2
    assert turn_loop._resolve_int_extra(
        {"barge_energy_chunks": 1}, "barge_energy_chunks", 2) == 1
    assert turn_loop._resolve_int_extra(
        {"barge_energy_chunks": "3"}, "barge_energy_chunks", 2) == 3
    assert turn_loop._resolve_int_extra(
        {"barge_energy_chunks": 0}, "barge_energy_chunks", 2) == 2
    assert turn_loop._resolve_int_extra(
        {"barge_energy_chunks": "x"}, "barge_energy_chunks", 2) == 2


# ---------------------------------------------------------------------------
# VampCache (clip cache, no network)
# ---------------------------------------------------------------------------


def test_vamp_cache_not_ready_until_clips_exist():
    cache = vamp_mod.VampCache("key", "voice-1")
    assert cache.ready is False
    assert cache.pick() is None


def test_vamp_cache_pick_never_repeats_consecutively():
    cache = vamp_mod.VampCache("key", "voice-1")
    cache._clips = [("a", b"\x01"), ("b", b"\x02"), ("c", b"\x03")]
    assert cache.ready is True
    picks = [cache.pick()[0] for _ in range(50)]
    assert all(x != y for x, y in zip(picks, picks[1:]))
    assert set(picks) <= {"a", "b", "c"}


def test_vamp_cache_single_clip_repeats_allowed():
    cache = vamp_mod.VampCache("key", "voice-1")
    cache._clips = [("only", b"\x01")]
    assert cache.pick() == ("only", b"\x01")
    assert cache.pick() == ("only", b"\x01")


def test_vamp_cache_start_requires_voice_id():
    cache = vamp_mod.VampCache("key", "")
    assert cache.start() is None
    assert cache.ready is False


@pytest.mark.asyncio
async def test_vamp_cache_background_synthesis_and_failures(monkeypatch):
    cache = vamp_mod.VampCache("key", "voice-1", texts=["ok one", "bad", "ok two"])

    async def fake_synth(text):
        if text == "bad":
            raise RuntimeError("boom")
        return b"\x01\x00" * 100

    monkeypatch.setattr(cache, "_synthesize_clip", fake_synth)
    task = cache.start()
    assert task is not None
    assert cache.ready is False          # disabled until synthesis lands
    await task
    assert cache.ready is True
    assert [t for t, _ in cache._clips] == ["ok one", "ok two"]


def test_trim_silence_strips_head_and_tail():
    win = vamp_mod._TRIM_WIN_BYTES
    quiet = b"\x05\x00" * (win // 2)     # rms 5 — below the trim floor
    loud = b"\x00\x20" * (win // 2)      # rms ~8k
    pcm = quiet * 3 + loud * 2 + quiet * 3
    trimmed = vamp_mod.trim_silence(pcm)
    assert trimmed == loud * 2


# ---------------------------------------------------------------------------
# Turn-loop vamp scheduling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_energy_trigger_fires_vamp_then_turn_audio_queues_behind(
        monkeypatch, caplog):
    agents = install_agents(monkeypatch, [
        say(GREETING_REPLY), say("The substantive answer arrives now.")])
    stt, factory = FakeSTT(), FakeTTSFactory()
    transport = MarkedTransport()
    vamp = FakeVamp()
    vloop = _make_loop(stt, factory, transport, vamp=vamp)
    task = asyncio.create_task(vloop.run())
    try:
        await _await_listening_after_greeting(vloop)
        caplog.set_level("INFO", logger=turn_loop.__name__)
        # >=2 hot chunks (speech), then quiet: the 3rd quiet chunk fires.
        for _ in range(3):
            await vloop.on_inbound_audio(LOUD_CHUNK)
        for _ in range(2):
            await vloop.on_inbound_audio(QUIET_CHUNK)
        assert transport.chunks == []            # not yet
        await vloop.on_inbound_audio(QUIET_CHUNK)
        # clip enqueued in 80ms chunks, BEFORE any turn exists
        assert transport.chunks == [CLIP_PCM[:7680], CLIP_PCM[7680:]]
        assert vamp.picks == ["Right."]
        # extra quiet chunks must not re-fire
        for _ in range(4):
            await vloop.on_inbound_audio(QUIET_CHUNK)
        assert len(transport.chunks) == 2
        # the substantive turn follows as usual
        stt.push_text("what time is it?")
        stt.push_end_of_turn_step()
        await _eventually(lambda: len(vloop._history) == 3)
        records = _telemetry_records(caplog)
        turn_recs = [r for r in records if r["event"] == "voice_turn"
                     and r["vad_end_ms"] is not None]
        assert len(turn_recs) == 1
        assert turn_recs[0]["vamp"]["fired"] is True
        assert turn_recs[0]["vamp"]["trigger"] == "energy"
        assert turn_recs[0]["vamp"]["text"] == "Right."
        assert turn_recs[0]["vamp"]["false_fire"] is False
        # perceived first audio = the vamp frame, measured vs speech end
        assert turn_recs[0]["totals"]["perceived_first_audio_ms"] is not None
    finally:
        await vloop.stop()
        await task


@pytest.mark.asyncio
async def test_vamp_rearms_after_turn_completes(monkeypatch):
    agents = install_agents(monkeypatch, [
        say(GREETING_REPLY), say("First answer."), say("Second answer.")])
    stt, factory = FakeSTT(), FakeTTSFactory()
    transport = MarkedTransport()
    vamp = FakeVamp(clips=[("Okay —", CLIP_PCM)])
    vloop = _make_loop(stt, factory, transport, vamp=vamp)
    task = asyncio.create_task(vloop.run())
    try:
        await _await_listening_after_greeting(vloop)
        for _ in range(2):
            await vloop.on_inbound_audio(LOUD_CHUNK)
        for _ in range(3):
            await vloop.on_inbound_audio(QUIET_CHUNK)
        assert len(vamp.picks) == 1
        stt.push_text("first question?")
        stt.push_end_of_turn_step()
        await _eventually(lambda: len(vloop._history) == 3)
        # next cycle: vamp re-armed
        for _ in range(2):
            await vloop.on_inbound_audio(LOUD_CHUNK)
        for _ in range(3):
            await vloop.on_inbound_audio(QUIET_CHUNK)
        assert len(vamp.picks) == 2
        assert len(agents) >= 2
    finally:
        await vloop.stop()
        await task


@pytest.mark.asyncio
async def test_vad_trigger_fires_on_short_horizon_crossing(monkeypatch):
    install_agents(monkeypatch, [say(GREETING_REPLY)])
    stt, factory = FakeSTT(), FakeTTSFactory()
    transport = MarkedTransport()
    vamp = FakeVamp()
    vloop = _make_loop(stt, factory, transport,
                       extra={"vamp_trigger": "vad"}, vamp=vamp)
    task = asyncio.create_task(vloop.run())
    try:
        await _await_listening_after_greeting(vloop)
        # energy silence onset must NOT fire in vad mode
        for _ in range(2):
            await vloop.on_inbound_audio(LOUD_CHUNK)
        for _ in range(5):
            await vloop.on_inbound_audio(QUIET_CHUNK)
        assert vamp.picks == []
        # short-horizon crossing with pending text fires the vamp
        stt.push_text("a question")
        stt.push({"type": "step", "total_duration_s": 1.0, "vad": [
            {"horizon_s": 0.5, "inactivity_prob": 0.8},
            {"horizon_s": 1.0, "inactivity_prob": 0.3},
            {"horizon_s": 2.0, "inactivity_prob": 0.1},
            {"horizon_s": 3.0, "inactivity_prob": 0.1},
        ]})
        await _eventually(lambda: vamp.picks == ["Right."])
        assert transport.chunks[:2] == [CLIP_PCM[:7680], CLIP_PCM[7680:]]
    finally:
        await vloop.stop()
        await task


@pytest.mark.asyncio
async def test_vamp_skipped_when_clips_not_ready(monkeypatch):
    install_agents(monkeypatch, [say(GREETING_REPLY)])
    stt, factory = FakeSTT(), FakeTTSFactory()
    transport = MarkedTransport()
    vamp = FakeVamp(ready=False)
    vloop = _make_loop(stt, factory, transport, vamp=vamp)
    task = asyncio.create_task(vloop.run())
    try:
        await _await_listening_after_greeting(vloop)
        for _ in range(2):
            await vloop.on_inbound_audio(LOUD_CHUNK)
        for _ in range(4):
            await vloop.on_inbound_audio(QUIET_CHUNK)
        assert vamp.picks == []
        assert transport.chunks == []
    finally:
        await vloop.stop()
        await task


@pytest.mark.asyncio
async def test_vamp_trigger_off_disables_firing(monkeypatch):
    install_agents(monkeypatch, [say(GREETING_REPLY)])
    stt, factory = FakeSTT(), FakeTTSFactory()
    transport = MarkedTransport()
    vamp = FakeVamp()
    vloop = _make_loop(stt, factory, transport,
                       extra={"vamp_trigger": "off"}, vamp=vamp)
    task = asyncio.create_task(vloop.run())
    try:
        await _await_listening_after_greeting(vloop)
        for _ in range(2):
            await vloop.on_inbound_audio(LOUD_CHUNK)
        for _ in range(4):
            await vloop.on_inbound_audio(QUIET_CHUNK)
        assert vamp.picks == []
    finally:
        await vloop.stop()
        await task


@pytest.mark.asyncio
async def test_false_fire_cancelled_when_user_resumes(monkeypatch, caplog):
    agents = install_agents(monkeypatch, [
        say(GREETING_REPLY), say("Answer to the full question.")])
    stt, factory = FakeSTT(), FakeTTSFactory()
    transport = MarkedTransport()
    vamp = FakeVamp()
    vloop = _make_loop(stt, factory, transport, vamp=vamp)
    task = asyncio.create_task(vloop.run())
    try:
        await _await_listening_after_greeting(vloop)
        caplog.set_level("INFO", logger=turn_loop.__name__)
        for _ in range(2):
            await vloop.on_inbound_audio(LOUD_CHUNK)
        for _ in range(3):
            await vloop.on_inbound_audio(QUIET_CHUNK)
        assert len(vamp.picks) == 1          # vamp fired (mid-utterance)
        # ...but the user was only pausing: speech resumes while the clip
        # is still inside its playout window -> remaining audio dropped.
        await vloop.on_inbound_audio(LOUD_CHUNK)
        assert transport.clear_calls == 0    # one hot chunk = echo guard
        await vloop.on_inbound_audio(LOUD_CHUNK)
        assert transport.clear_calls == 1
        # no second vamp for this cycle
        for _ in range(4):
            await vloop.on_inbound_audio(QUIET_CHUNK)
        assert len(vamp.picks) == 1
        # the eventual turn's telemetry carries the false fire
        stt.push_text("the full question?")
        stt.push_end_of_turn_step()
        await _eventually(lambda: len(vloop._history) == 3)
        recs = [r for r in _telemetry_records(caplog)
                if r["event"] == "voice_turn" and r["vamp"]["fired"]]
        assert recs and recs[0]["vamp"]["false_fire"] is True
        assert len(agents) == 2
    finally:
        await vloop.stop()
        await task


@pytest.mark.asyncio
async def test_no_vamp_cache_means_no_vamp(monkeypatch):
    install_agents(monkeypatch, [say(GREETING_REPLY)])
    stt, factory = FakeSTT(), FakeTTSFactory()
    transport = MarkedTransport()
    vloop = _make_loop(stt, factory, transport, vamp=None)
    task = asyncio.create_task(vloop.run())
    try:
        await _await_listening_after_greeting(vloop)
        for _ in range(2):
            await vloop.on_inbound_audio(LOUD_CHUNK)
        for _ in range(4):
            await vloop.on_inbound_audio(QUIET_CHUNK)
        assert transport.chunks == []
    finally:
        await vloop.stop()
        await task


# ---------------------------------------------------------------------------
# Telemetry shape (§16)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_turn_telemetry_record_shape_and_call_summary(
        monkeypatch, caplog):
    install_agents(monkeypatch, [
        say(GREETING_REPLY), say("It is quarter past three.")])
    stt, factory = FakeSTT(), FakeTTSFactory()
    transport = MarkedTransport()
    vloop = _make_loop(stt, factory, transport, vamp=FakeVamp())
    caplog.set_level("INFO", logger=turn_loop.__name__)
    task = asyncio.create_task(vloop.run())
    try:
        await _await_listening_after_greeting(vloop)
        for _ in range(2):
            await vloop.on_inbound_audio(LOUD_CHUNK)
        for _ in range(3):
            await vloop.on_inbound_audio(QUIET_CHUNK)
        stt.push_text("what time is it?")
        stt.push_end_of_turn_step()
        await _eventually(lambda: len(vloop._history) == 3)
    finally:
        await vloop.stop()
        await task
    records = _telemetry_records(caplog)
    turns = [r for r in records if r["event"] == "voice_turn"
             and r["vad_end_ms"] is not None]
    assert len(turns) == 1
    rec = turns[0]
    # every §16 field is present
    for key in ("vad_end_ms", "flush_result", "flush_done_ms",
                "agent_start_ms", "first_delta_ms", "first_sentence_ms",
                "tts_first_audio_ms", "first_frame_written_ms",
                "vamp", "totals", "status", "turn"):
        assert key in rec, key
    assert rec["status"] == "ok"
    assert rec["flush_result"] == "ack"
    assert rec["vamp"]["fired_at_ms"] is not None
    # vamp fired BEFORE vad-end (it rides the faster energy signal)
    assert rec["vamp"]["fired_at_ms"] <= rec["vad_end_ms"]
    assert rec["totals"]["perceived_first_audio_ms"] is not None
    assert rec["totals"]["turn_total_ms"] >= 0
    # ordering sanity on the substantive legs
    assert rec["agent_start_ms"] <= rec["first_delta_ms"]
    assert rec["first_delta_ms"] <= rec["first_sentence_ms"]
    # per-call summary emitted on stop
    summaries = [r for r in records if r["event"] == "voice_call_summary"]
    assert len(summaries) == 1
    summary = summaries[0]
    assert summary["turns"] == 1
    assert summary["vamp_fired"] == 1
    assert summary["vamp_false_fires"] == 0
    assert summary["perceived_first_audio_ms"]["n"] == 1
    assert "median" in summary["perceived_first_audio_ms"]
    assert "p90" in summary["perceived_first_audio_ms"]


# ---------------------------------------------------------------------------
# Configurable energy barge-in chunk count
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_single_chunk_energy_barge_in_behind_config(monkeypatch):
    agents = install_agents(monkeypatch, [
        speak_then_block("A long reply about to be cut by one hot chunk. ")])
    stt, factory = FakeSTT(), FakeTTSFactory()
    transport = MarkedTransport()
    vloop = _make_loop(stt, factory, transport,
                       extra={"barge_energy_chunks": 1})
    task = asyncio.create_task(vloop.run())
    try:
        await _eventually(lambda: vloop._state == turn_loop.SPEAKING)
        await vloop.on_inbound_audio(LOUD_CHUNK)
        await _eventually(lambda: vloop._state == turn_loop.LISTENING,
                          msg="1-chunk energy barge-in never fired")
        assert agents[0].interrupt_event.is_set()
    finally:
        await vloop.stop()
        await task
