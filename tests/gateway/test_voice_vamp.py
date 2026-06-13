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
    # These vamp tests drive end-of-turn / barge-in via Gradium VAD step
    # events, which ENG-555 kept behind turn_detector="gradium" as a fallback
    # (Smart Turn is the new default and has its own tests in
    # test_voice_turn_loop.py). The vamp triggers themselves (energy/vad) are
    # detector-independent; pinning gradium keeps the step-driven turn flow
    # these tests assert. A test can still override turn_detector via extra.
    merged = {"turn_detector": "gradium"}
    merged.update(extra or {})
    return turn_loop.VoiceTurnLoop(
        stt, factory, transport, extra=merged, vamp=vamp)


def _telemetry_records(caplog):
    out = []
    for rec in caplog.records:
        msg = rec.getMessage()
        if not msg.startswith("voice/telemetry "):
            continue
        payload = msg[len("voice/telemetry "):]
        # The same prefix carries the (non-JSON) "sink unwritable" warning;
        # only the structured records are JSON objects.
        if not payload.lstrip().startswith("{"):
            continue
        out.append(json.loads(payload))
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


def test_resolve_reasoning_override():
    """platforms.voice.extra.reasoning_effort overrides the gateway
    reasoning effort for voice turns only (thinking tokens delay the
    first spoken sentence). None/invalid = use the gateway default."""
    assert turn_loop._resolve_reasoning_override({}) is None
    assert turn_loop._resolve_reasoning_override(
        {"reasoning_effort": "none"}) == {"enabled": False}
    assert turn_loop._resolve_reasoning_override(
        {"reasoning_effort": "low"}) == {"enabled": True, "effort": "low"}
    assert turn_loop._resolve_reasoning_override(
        {"reasoning_effort": "bogus"}) is None
    assert turn_loop._resolve_reasoning_override(
        {"reasoning_effort": ""}) is None


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
# Energy-trigger dual-confirm (VAD-step veto) + retry window
#
# Verify run 2026-06-12: the one-shot `==` trigger MISSED 1/3 clean turns
# (perceived first audio blew out to ~3.9s) and FALSE-FIRED once during a
# mid-utterance pause (turn 8). The trigger now (a) retries across a
# bounded window instead of firing on exactly the Nth quiet chunk, (b) is
# vetoed by a fresh speech-positive VAD step (semantic VAD keeps the
# short-horizon inactivity LOW through an intra-utterance pause), and
# (c) catches up on any silence-confirming step — including vad-end.
# ---------------------------------------------------------------------------


def push_short_quiet_step(stt, p=0.8):
    """Step whose SHORT horizon confirms silence but whose end-of-turn
    (2.0s) horizon does not fire — silence just began."""
    stt.push({"type": "step", "total_duration_s": 1.0, "vad": [
        {"horizon_s": 0.5, "inactivity_prob": p},
        {"horizon_s": 1.0, "inactivity_prob": 0.3},
        {"horizon_s": 2.0, "inactivity_prob": 0.1},
        {"horizon_s": 3.0, "inactivity_prob": 0.1},
    ]})


@pytest.mark.asyncio
async def test_energy_fire_retries_when_clips_become_ready_late(monkeypatch):
    """Miss mode 1 (verify run): clips not ready at the exact Nth quiet
    chunk used to skip the vamp FOR THE WHOLE TURN. The fire window now
    retries on later quiet chunks."""
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
        for _ in range(3):
            await vloop.on_inbound_audio(QUIET_CHUNK)
        assert vamp.picks == []              # skipped: clips not ready
        vamp._ready = True
        await vloop.on_inbound_audio(QUIET_CHUNK)   # 4th chunk: retry fires
        assert vamp.picks == ["Right."]
    finally:
        await vloop.stop()
        await task


@pytest.mark.asyncio
async def test_speech_step_vetoes_energy_fire_until_vad_confirms(monkeypatch):
    """Dual-confirm: a fresh speech-positive VAD step blocks the energy
    fire (the user is mid-utterance); the fire happens the moment a
    silence-confirming step lands."""
    install_agents(monkeypatch, [say(GREETING_REPLY)])
    stt, factory = FakeSTT(), FakeTTSFactory()
    transport = MarkedTransport()
    vamp = FakeVamp()
    vloop = _make_loop(stt, factory, transport, vamp=vamp)
    task = asyncio.create_task(vloop.run())
    try:
        await _await_listening_after_greeting(vloop)
        for _ in range(2):
            await vloop.on_inbound_audio(LOUD_CHUNK)
        stt.push_speech_step()
        await _eventually(lambda: vloop._last_step_speech_t is not None)
        for _ in range(5):
            await vloop.on_inbound_audio(QUIET_CHUNK)
        assert vamp.picks == []              # vetoed, not fired
        assert vloop._vamp_veto_deferred >= 1
        push_short_quiet_step(stt)           # VAD confirms silence
        await _eventually(lambda: vamp.picks == ["Right."])
        assert transport.chunks[:2] == [CLIP_PCM[:7680], CLIP_PCM[7680:]]
    finally:
        await vloop.stop()
        await task


@pytest.mark.asyncio
async def test_stale_speech_step_cannot_veto(monkeypatch):
    """The veto rides the STT round-trip; a stale speech step (older than
    VAMP_VETO_FRESH_S) must not block the energy fire — a stalled stream
    can only delay the vamp by the freshness window."""
    install_agents(monkeypatch, [say(GREETING_REPLY)])
    monkeypatch.setattr(turn_loop, "VAMP_VETO_FRESH_S", 0.05)
    stt, factory = FakeSTT(), FakeTTSFactory()
    transport = MarkedTransport()
    vamp = FakeVamp()
    vloop = _make_loop(stt, factory, transport, vamp=vamp)
    task = asyncio.create_task(vloop.run())
    try:
        await _await_listening_after_greeting(vloop)
        for _ in range(2):
            await vloop.on_inbound_audio(LOUD_CHUNK)
        stt.push_speech_step()
        await _eventually(lambda: vloop._last_step_speech_t is not None)
        await asyncio.sleep(0.12)            # veto goes stale
        for _ in range(3):
            await vloop.on_inbound_audio(QUIET_CHUNK)
        assert vamp.picks == ["Right."]
    finally:
        await vloop.stop()
        await task


@pytest.mark.asyncio
async def test_mid_utterance_pause_does_not_false_fire(monkeypatch, caplog):
    """The turn-8 false fire (verify run): a ~240ms intra-utterance pause
    with semantic VAD still speech-positive. The veto holds through the
    pause; the vamp fires exactly once at the REAL end of the utterance."""
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
        # First clause, then a pause with VAD still saying SPEECH.
        for _ in range(2):
            await vloop.on_inbound_audio(LOUD_CHUNK)
        stt.push_speech_step()
        await _eventually(lambda: vloop._last_step_speech_t is not None)
        for _ in range(4):
            await vloop.on_inbound_audio(QUIET_CHUNK)
        assert vamp.picks == []              # the old trigger fired HERE
        # The user resumes (no clip is playing, so no cancel needed).
        for _ in range(3):
            await vloop.on_inbound_audio(LOUD_CHUNK)
        stt.push_speech_step()
        await asyncio.sleep(0.02)
        # Real end of turn: energy silence + VAD confirmation.
        for _ in range(3):
            await vloop.on_inbound_audio(QUIET_CHUNK)
        assert vamp.picks == []              # still vetoed (fresh speech step)
        push_short_quiet_step(stt)
        await _eventually(lambda: vamp.picks == ["Right."])
        stt.push_text("the full question?")
        stt.push_end_of_turn_step()
        await _eventually(lambda: len(vloop._history) == 3)
        recs = [r for r in _telemetry_records(caplog)
                if r["event"] == "voice_turn" and r["vamp"]["fired"]]
        assert recs and recs[0]["vamp"]["false_fire"] is False
        assert len(vamp.picks) == 1
    finally:
        await vloop.stop()
        await task


@pytest.mark.asyncio
async def test_vad_end_step_catches_up_a_vetoed_fire(monkeypatch, caplog):
    """Miss mode 2: if the veto held all the way to vad-end, the
    end-of-turn step itself confirms silence and fires the vamp before
    the turn starts — the vamp can be late, never absent."""
    install_agents(monkeypatch, [
        say(GREETING_REPLY), say("The substantive answer.")])
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
        stt.push_speech_step()
        await _eventually(lambda: vloop._last_step_speech_t is not None)
        for _ in range(4):
            await vloop.on_inbound_audio(QUIET_CHUNK)
        assert vamp.picks == []              # vetoed
        stt.push_text("what time is it")
        stt.push_end_of_turn_step()          # confirms silence AND ends turn
        await _eventually(lambda: vamp.picks == ["Right."])
        await _eventually(lambda: len(vloop._history) == 3)
        recs = [r for r in _telemetry_records(caplog)
                if r["event"] == "voice_turn" and r["vad_end_ms"] is not None]
        assert recs and recs[0]["vamp"]["fired"] is True
        assert recs[0]["vamp"]["trigger"] == "energy"
        assert recs[0]["vamp"]["veto_deferred_chunks"] >= 1
        # "what time is it" has no terminal punctuation -> no eager start
        assert recs[0]["eager_start"] is False
    finally:
        await vloop.stop()
        await task


@pytest.mark.asyncio
async def test_fire_window_bounded(monkeypatch):
    """Past VAMP_ENERGY_MAX_SILENCE_CHUNKS the vamp stays silent: that
    deep into the silence the vad-end path owns the turn, and a clip
    landing seconds after the user stopped would read as a non sequitur."""
    install_agents(monkeypatch, [say(GREETING_REPLY)])
    stt, factory = FakeSTT(), FakeTTSFactory()
    transport = MarkedTransport()
    vamp = FakeVamp()
    vloop = _make_loop(stt, factory, transport, vamp=vamp)
    task = asyncio.create_task(vloop.run())
    try:
        await _await_listening_after_greeting(vloop)
        for _ in range(2):
            await vloop.on_inbound_audio(LOUD_CHUNK)
        stt.push_speech_step()
        await _eventually(lambda: vloop._last_step_speech_t is not None)
        for _ in range(turn_loop.VAMP_ENERGY_MAX_SILENCE_CHUNKS + 1):
            await vloop.on_inbound_audio(QUIET_CHUNK)
        assert vamp.picks == []              # vetoed through the window
        push_short_quiet_step(stt)           # confirmation arrives too late
        await asyncio.sleep(0.1)
        assert vamp.picks == []
        await vloop.on_inbound_audio(QUIET_CHUNK)
        assert vamp.picks == []
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
                "vamp", "totals", "status", "turn", "eager_start"):
        assert key in rec, key
    assert rec["status"] == "ok"
    assert rec["flush_result"] == "ack"
    # "what time is it?" ends in terminal punctuation -> eager start
    assert rec["eager_start"] is True
    assert rec["vamp"]["fired_at_ms"] is not None
    assert rec["vamp"]["veto_deferred_chunks"] == 0
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
    # Config-drift guards (verify run: substantive first-audio regressed
    # 2.6s -> ~4s; the matrix's Haiku override may have been absent): the
    # summary names the effective per-turn config so a live run is
    # self-describing.
    assert summary["model_override"] is None        # extra.model unset here
    assert summary["reasoning_effort"] == "gateway-default"
    assert summary["eager_starts"] == 1
    assert summary["perceived_first_audio_ms"]["n"] == 1
    assert "median" in summary["perceived_first_audio_ms"]
    assert "p90" in summary["perceived_first_audio_ms"]


@pytest.mark.asyncio
async def test_call_summary_names_model_and_reasoning_overrides(
        monkeypatch, caplog):
    install_agents(monkeypatch, [say(GREETING_REPLY)])
    stt, factory = FakeSTT(), FakeTTSFactory()
    transport = MarkedTransport()
    vloop = _make_loop(stt, factory, transport, extra={
        "model": "claude-haiku-4-5-20251001", "reasoning_effort": "none"})
    caplog.set_level("INFO", logger=turn_loop.__name__)
    task = asyncio.create_task(vloop.run())
    await _await_listening_after_greeting(vloop)
    await vloop.stop()
    await task
    summaries = [r for r in _telemetry_records(caplog)
                 if r["event"] == "voice_call_summary"]
    assert summaries[0]["model_override"] == "claude-haiku-4-5-20251001"
    assert summaries[0]["reasoning_effort"] == "none"


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
