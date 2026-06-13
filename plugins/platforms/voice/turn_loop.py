"""Voice turn orchestrator.

One VoiceTurnLoop per live call. Flow per turn:
  STT semantic VAD says end-of-turn  ->  flush STT, collect utterance
  -> fresh AIAgent with stream_delta_callback (api_server pattern,
     gateway/platforms/api_server.py:1604-1657 + 3510-3554)
  -> deltas marshaled via loop.call_soon_threadsafe into an asyncio queue
  -> sentence buffer (mirrors tools/tts_tool.py:2523-2641; regex from
     tts_tool.py:2426; NOT imported — that function writes to speakers)
  -> sentences -> per-turn Gradium TTS socket -> transport
Barge-in: caller speech while the agent is thinking/speaking ->
  agent.interrupt() (same call as api_server.py:4091) + tts.abort()
  + transport.clear_output().
Agent speaks first: run() executes a greeting turn before listening
(notes §14 item 8).

Live-verified Gradium facts this loop relies on (notes §15):
  - STT step VAD carries FOUR horizons (0.5/1.0/2.0/3.0 s) — always look
    horizons up by value, never by index.
  - Both sockets emit a leading {"type":"ready",...}; unknown message
    types are ignored here by construction (only text/step/flushed/error
    are handled).
  - GradiumSTT.flush() returns Optional[int]; None means no live session
    so no "flushed" event will ever arrive — skip the flushed-wait.
  - GradiumTTSTurn.end() blocks until the server's end_of_stream; per its
    docstring it is wrapped in asyncio.wait_for so a stalled server
    cannot hang the turn.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import os
import re
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Sentence-final punctuation: used to track how much UNSYNTHESIZED text the
# TTS server is holding (it only synthesizes on [.!?] or an explicit
# <flush>); long punctuation-free stretches get a forced flush on idle.
_PUNCT_RE = re.compile(r"[.!?]")
LONG_FLUSH_LEN = 100

END_OF_TURN_HORIZON = 2.0
END_OF_TURN_PROB = 0.7
# ---------------------------------------------------------------------------
# Turn detection (ENG-555): Smart Turn replaces Gradium's semantic-VAD
# step-prob end-of-turn / barge-in logic. The Gradium step path is kept
# behind turn_detector="gradium" as a fallback; "smart_turn" (default)
# drives turns from a local ONNX endpoint model + a local RMS energy gate.
# Gradium STT STILL transcribes (text events are consumed regardless) — only
# its VAD-step-driven turn/barge DECISIONS are replaced.
TURN_DETECTORS = ("smart_turn", "gradium")
DEFAULT_TURN_DETECTOR = "smart_turn"
# Continuous local silence (RMS below the floor) after speech before a Smart
# Turn check runs. The model needs the utterance to have paused; 350 ms is
# the natural end-of-clause gap without clipping mid-sentence pauses.
DEFAULT_VAD_STOP_SECS = 0.35
# Smart Turn endpoint probability (>= => turn complete). The model emits a
# probability in [0, 1]; 0.5 is the model's own decision boundary.
DEFAULT_SMART_TURN_THRESHOLD = 0.5
# Speculative start: a LOWER endpoint probability that is confident enough to
# begin the agent turn on the transcript-so-far, overlapping LLM TTFT with
# the tail of user speech. If more speech then arrives, the speculative turn
# is interrupted and restarted with the fuller transcript.
DEFAULT_SPECULATIVE_START = True
DEFAULT_SPECULATIVE_THRESHOLD = 0.55
# Hard fallback: if local silence persists this long without Smart Turn ever
# crossing the threshold (model false-negative, or a genuinely trailing-off
# utterance), finalize the turn anyway. Bounds worst-case end-of-turn
# latency to this value.
DEFAULT_SMART_TURN_FALLBACK_SECS = 2.5
# Re-run Smart Turn at most this often while silence holds (the model is
# ~13 ms but we do not need it every 80 ms chunk).
SMART_TURN_MIN_INTERVAL_S = 0.12
# Barge-in gating during agent playback (Smart Turn path). Energy alone is
# echo/cough-prone, so a real interruption must clear BOTH a minimum sustained
# duration AND a minimum word count from Gradium's partial transcript.
DEFAULT_ALLOW_INTERRUPTIONS = True
DEFAULT_MIN_INTERRUPTION_DURATION_MS = 500
DEFAULT_MIN_INTERRUPTION_WORDS = 2
# Barge-in on VAD evidence of user speech during THINKING/SPEAKING: the
# short-horizon inactivity probability collapses within ~1-2 steps of real
# speech, while finalized `text` events can lag seconds behind (live
# measurement 2026-06-12: 2-8s during overlapping speech). N consecutive
# speech-positive steps guard against one-step blips/echo.
BARGE_VAD_HORIZON = 0.5
BARGE_VAD_SPEECH_PROB = 0.5
BARGE_VAD_CONSEC_STEPS = 2
# Local energy trigger on inbound caller PCM (s16le mono): the Gradium VAD
# round-trip costs 0.6-1.8s during overlapping speech (measured live
# 2026-06-12); raw input energy is available instantly. N consecutive 80ms
# chunks above the RMS floor = user is talking over the agent. Daily/browser
# AEC + noise suppression keep agent echo and room noise below this floor;
# the slower VAD/text triggers remain as fallback.
ENERGY_BARGE_RMS = 500
# Two 80ms chunks: a bare "Wait—" is exactly 2 hot chunks before its
# inter-word dip (measured on the synthetic barge utterance); 4 chunks
# missed it entirely and lost the race to the ~700ms server-side VAD.
# Overridable via extra.barge_energy_chunks (1 = the experimental 80ms
# single-chunk trigger; keep default 2 — one chunk is blip/echo-prone).
ENERGY_BARGE_CHUNKS = 2
# Vamp (perceived-latency acknowledgment clips, notes §16): fired on a
# FAST end-of-turn signal, before the (slow, ~550ms) vad-end confirmation.
# Two trigger designs, selectable via extra.vamp_trigger:
#   energy: quiet 80ms inbound chunks after >=M hot chunks (the inverse
#           of the energy barge-in trigger), DUAL-CONFIRMED against the
#           short-horizon VAD: a fresh speech-positive step vetoes the
#           fire (semantic VAD keeps short-horizon inactivity LOW through
#           an intra-utterance pause — verify run 2026-06-12 false fire,
#           turn 8), and any silence-confirming step — including the
#           vad-end step itself — fires a deferred vamp immediately. The
#           fire window RETRIES (>=, bounded) instead of one-shotting on
#           the exact Nth chunk (verify run miss: 1/3 clean turns lost
#           the vamp and perceived first-audio blew out to ~3.9s).
#   vad:    first short-horizon VAD crossing (0.5s inactivity_prob rising
#           through VAMP_VAD_PROB) — semantically smarter, but rides the
#           STT round-trip.
VAMP_TRIGGERS = ("energy", "vad", "off")
DEFAULT_VAMP_TRIGGER = "energy"
VAMP_ENERGY_SILENCE_CHUNKS = 3      # 240ms of quiet after speech
VAMP_ENERGY_MIN_SPEECH_CHUNKS = 2   # >=160ms of speech before quiet counts
# Upper edge of the energy fire window: ~960ms after silence onset the
# vad-end path (~550-760ms live) owns the turn, and a clip landing seconds
# late reads as a non sequitur. Also bounds retry attempts/log volume.
VAMP_ENERGY_MAX_SILENCE_CHUNKS = 12
# Dual-confirm thresholds on the 0.5s-horizon inactivity probability:
# <= SPEECH_PROB the step asserts the user is mid-utterance (veto);
# >= CLEAR_PROB it confirms silence (fires a deferred vamp). Between the
# two the step is ambiguous and the last assertion stands.
VAMP_VETO_SPEECH_PROB = 0.5
VAMP_VETO_CLEAR_PROB = 0.6
# A speech-positive step older than this cannot veto: steps ride the STT
# round-trip, and a DEAF/stalled stream must only be able to delay the
# vamp by this bound, never suppress it (the miss rate is the metric).
VAMP_VETO_FRESH_S = 0.4
VAMP_VAD_HORIZON = 0.5
VAMP_VAD_PROB = 0.7
# False-fire recovery: the user resuming speech while the vamp clip is
# still playing means the vamp fired mid-utterance — drop the remaining
# clip audio after this many hot chunks (2 = echo/blip guard, same
# reasoning as the barge-in trigger).
VAMP_CANCEL_HOT_CHUNKS = 2
# Cancel window slack past the clip's nominal duration (transit + jitter).
VAMP_PLAYOUT_SLACK_S = 0.3
FLUSHED_WAIT_TIMEOUT_S = 2.0
# If the first sentence hasn't reached a sentence-final punctuation this
# long after its first words went to TTS, force a <flush> once so audio
# starts (prosody costs less than dead air on the first clause).
FIRST_SENTENCE_FLUSH_S = 0.6
TTS_END_TIMEOUT_S = 30.0
# Bound on awaiting the executor-run agent after a turn ends (normally the
# future is already done; after an interrupt a well-behaved agent returns
# promptly). The thread itself cannot be killed, but the event loop must
# never block on a generation that ignores its interrupt.
RUN_FUTURE_TIMEOUT_S = 60.0
DEFAULT_GREETING_PROMPT = (
    "(The user just joined a voice call with you. Greet them by voice in "
    "one short, warm sentence and ask how you can help.)"
)
DEFAULT_FILLER_TEXT = "One moment."

LISTENING, THINKING, SPEAKING = "listening", "thinking", "speaking"

# Durable telemetry sink (ENG-555): the voice/telemetry JSON records are
# the per-call audit trail (latency legs, vamp behavior, billable ASR
# seconds), but log lines vanish at the container's default log level and
# on log rotation. Every record therefore ALSO appends to a JSONL file on
# the agent volume (/opt/data persists across container restarts):
#   docker exec <agent> tail /opt/data/voice-telemetry.jsonl
# Fail-soft: an unwritable volume logs ONE warning and never raises into
# the call path.
TELEMETRY_SINK_PATH = "/opt/data/voice-telemetry.jsonl"
_sink_warned = False


def emit_telemetry(record: Dict[str, Any]) -> None:
    """Emit one voice/telemetry record: log it at WARNING (visible at the
    default container log level) and append it to the durable JSONL sink.
    Open-append-close per record — a handful of records per call, and the
    record must be on disk the moment the line is logged."""
    global _sink_warned
    line = json.dumps(record)
    logger.warning("voice/telemetry %s", line)
    try:
        with open(TELEMETRY_SINK_PATH, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except OSError as exc:
        if not _sink_warned:
            _sink_warned = True
            logger.warning(
                "voice/telemetry sink unwritable (%s): %s — records stay "
                "in the process logs only", TELEMETRY_SINK_PATH, exc)


def _rms(pcm: bytes) -> float:
    """RMS of s16le mono PCM (audioop-free: removed in Python 3.13).
    Subsamples every 4th frame — plenty for an 80ms energy gate."""
    samples = memoryview(pcm).cast("h")[::4]
    if len(samples) == 0:
        return 0.0
    return (sum(s * s for s in samples) / len(samples)) ** 0.5


def _resolve_turn_model(extra: Dict[str, Any]) -> Optional[str]:
    """platforms.voice.extra.model overrides the gateway default model for
    voice turns only (latency: a faster model for spoken replies without
    touching the agent's main model). None = use the gateway model."""
    raw = (extra or {}).get("model")
    if raw is None:
        return None
    model = str(raw).strip()
    return model or None


def _resolve_reasoning_override(extra: Dict[str, Any]) -> Optional[dict]:
    """platforms.voice.extra.reasoning_effort overrides the gateway
    reasoning effort for voice turns only: thinking tokens run before the
    first text delta, i.e. straight on top of substantive first-audio
    (verify run 2026-06-12: first_delta ~3.2s after agent_start on the
    default model — model choice and reasoning effort are the two
    config-side levers). None = use the gateway default."""
    raw = (extra or {}).get("reasoning_effort")
    if raw is None or not str(raw).strip():
        return None
    from hermes_constants import parse_reasoning_effort

    parsed = parse_reasoning_effort(str(raw))
    if parsed is None:
        logger.warning("voice/turn: invalid reasoning_effort %r ignored", raw)
    return parsed


def _resolve_int_extra(extra: Dict[str, Any], key: str, default: int) -> int:
    raw = (extra or {}).get(key)
    if raw is None:
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        logger.warning("voice/turn: invalid %s %r ignored", key, raw)
        return default
    if value < 1:
        logger.warning("voice/turn: %s must be >= 1; %r ignored", key, raw)
        return default
    return value


def _resolve_vamp_trigger(extra: Dict[str, Any]) -> str:
    raw = (extra or {}).get("vamp_trigger")
    if raw is None:
        return DEFAULT_VAMP_TRIGGER
    trigger = str(raw).strip().lower()
    if trigger not in VAMP_TRIGGERS:
        logger.warning("voice/turn: invalid vamp_trigger %r ignored "
                       "(valid: %s)", raw, "|".join(VAMP_TRIGGERS))
        return DEFAULT_VAMP_TRIGGER
    return trigger


def _resolve_float_extra(extra: Dict[str, Any], key: str, default: float) -> float:
    raw = (extra or {}).get(key)
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError):
        logger.warning("voice/turn: invalid %s %r ignored", key, raw)
        return default
    if value < 0:
        logger.warning("voice/turn: %s must be >= 0; %r ignored", key, raw)
        return default
    return value


def _resolve_bool_extra(extra: Dict[str, Any], key: str, default: bool) -> bool:
    raw = (extra or {}).get(key)
    if raw is None:
        return default
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        s = raw.strip().lower()
        if s in ("true", "1", "yes", "on"):
            return True
        if s in ("false", "0", "no", "off"):
            return False
    logger.warning("voice/turn: invalid %s %r ignored", key, raw)
    return default


def _resolve_nonneg_int_extra(extra: Dict[str, Any], key: str, default: int) -> int:
    """Like _resolve_int_extra but allows 0 (e.g. min_interruption_words=0
    to disable the word gate, or a 0ms duration floor)."""
    raw = (extra or {}).get(key)
    if raw is None:
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        logger.warning("voice/turn: invalid %s %r ignored", key, raw)
        return default
    if value < 0:
        logger.warning("voice/turn: %s must be >= 0; %r ignored", key, raw)
        return default
    return value


def _resolve_turn_detector(extra: Dict[str, Any]) -> str:
    raw = (extra or {}).get("turn_detector")
    if raw is None:
        return DEFAULT_TURN_DETECTOR
    mode = str(raw).strip().lower()
    if mode not in TURN_DETECTORS:
        logger.warning("voice/turn: invalid turn_detector %r ignored "
                       "(valid: %s)", raw, "|".join(TURN_DETECTORS))
        return DEFAULT_TURN_DETECTOR
    return mode


def _resolve_max_iterations(extra: Dict[str, Any]) -> int:
    """platforms.voice.extra.max_turns caps the per-utterance agent loop;
    falls back to the api_server default (HERMES_MAX_ITERATIONS env, 90)."""
    raw = extra.get("max_turns")
    if raw is not None:
        try:
            return int(raw)
        except (TypeError, ValueError):
            logger.warning("voice/turn: invalid max_turns %r ignored", raw)
    return int(os.getenv("HERMES_MAX_ITERATIONS", "90"))


def _create_voice_agent(
    session_id: str,
    stream_delta_callback,
    tool_progress_callback,
    *,
    extra: Optional[Dict[str, Any]] = None,
):
    """Fresh agent per turn — mirrors api_server._create_agent
    (gateway/platforms/api_server.py:1029-1065) with platform='voice'."""
    from run_agent import AIAgent
    from gateway.run import (
        GatewayRunner,
        _load_gateway_config,
        _resolve_gateway_model,
        _resolve_runtime_agent_kwargs,
    )
    from hermes_cli.tools_config import _get_platform_tools

    runtime_kwargs = _resolve_runtime_agent_kwargs()
    user_config = _load_gateway_config()
    return AIAgent(
        model=_resolve_turn_model(extra or {}) or _resolve_gateway_model(user_config),
        **runtime_kwargs,
        max_iterations=_resolve_max_iterations(extra or {}),
        quiet_mode=True,
        verbose_logging=False,
        enabled_toolsets=sorted(_get_platform_tools(user_config, "voice")),
        session_id=session_id,
        platform="voice",
        stream_delta_callback=stream_delta_callback,
        tool_progress_callback=tool_progress_callback,
        fallback_model=GatewayRunner._load_fallback_model(),
        reasoning_config=(_resolve_reasoning_override(extra or {})
                          or GatewayRunner._load_reasoning_config()),
    )


def _build_turn_detector():
    """Construct the Smart Turn detector + rolling 16 kHz buffer. Isolated
    in a module function so tests monkeypatch it with fakes (the real
    detector pulls onnxruntime/transformers, which the loop logic does not
    need)."""
    try:
        import turn_detection
    except ImportError:
        from . import turn_detection
    return turn_detection.SmartTurnDetector(), turn_detection.RollingAudioBuffer()


def _rms_for_detector(pcm: bytes) -> float:
    """RMS gate for the Smart Turn path (matches turn_detection.rms_energy /
    the legacy _rms here)."""
    return _rms(pcm)


class VoiceTurnLoop:
    def __init__(self, stt, tts_factory, transport, *, extra: Dict[str, Any],
                 vamp=None):
        """tts_factory(on_audio) -> opened GradiumTTSTurn (one per turn).
        vamp: optional VampCache (vamp.py) — pre-synthesized acknowledgment
        clips fired on the fast end-of-turn trigger."""
        self._stt = stt
        self._tts_factory = tts_factory
        self._transport = transport
        self._extra = extra or {}
        self._greeting = self._extra.get("greeting_prompt") or DEFAULT_GREETING_PROMPT
        self._filler = self._extra.get("filler_text") or DEFAULT_FILLER_TEXT
        self._session_id = f"voice-{uuid.uuid4().hex[:12]}"
        self._history: List[Dict[str, str]] = []
        self._pending_text: List[str] = []
        self._state = LISTENING
        # Vamp state. _vamp_armed gates ONE vamp per turn cycle (re-armed
        # when a turn completes); _vamp_playing_until bounds the false-fire
        # cancel window; the _utt counters implement the energy
        # silence-onset trigger (inverse of the barge-in energy trigger).
        self._vamp = vamp
        self._vamp_trigger = _resolve_vamp_trigger(self._extra)
        self._vamp_silence_chunks = _resolve_int_extra(
            self._extra, "vamp_energy_silence_chunks",
            VAMP_ENERGY_SILENCE_CHUNKS)
        self._vamp_min_speech = _resolve_int_extra(
            self._extra, "vamp_energy_min_speech_chunks",
            VAMP_ENERGY_MIN_SPEECH_CHUNKS)
        self._vamp_armed = True
        self._vamp_playing_until = 0.0
        self._vamp_resume_hot = 0
        self._utt_hot_chunks = 0
        self._utt_quiet_chunks = 0
        self._last_hot_t: Optional[float] = None
        # Dual-confirm veto state (energy trigger): receipt times of the
        # latest speech-asserting / silence-confirming short-horizon VAD
        # steps, plus a per-cycle count of quiet chunks the veto deferred
        # (telemetry: deferred*80ms = latency the veto added).
        self._last_step_speech_t: Optional[float] = None
        self._last_step_quiet_t: Optional[float] = None
        self._vamp_veto_deferred = 0
        self._energy_barge_chunks = _resolve_int_extra(
            self._extra, "barge_energy_chunks", ENERGY_BARGE_CHUNKS)
        # -- Smart Turn detection (ENG-555) --------------------------------
        self._turn_detector_mode = _resolve_turn_detector(self._extra)
        self._vad_stop_secs = _resolve_float_extra(
            self._extra, "vad_stop_secs", DEFAULT_VAD_STOP_SECS)
        self._smart_turn_threshold = _resolve_float_extra(
            self._extra, "smart_turn_threshold", DEFAULT_SMART_TURN_THRESHOLD)
        self._speculative_start = _resolve_bool_extra(
            self._extra, "speculative_start", DEFAULT_SPECULATIVE_START)
        self._speculative_threshold = _resolve_float_extra(
            self._extra, "speculative_threshold", DEFAULT_SPECULATIVE_THRESHOLD)
        self._smart_turn_fallback_secs = _resolve_float_extra(
            self._extra, "smart_turn_fallback_secs",
            DEFAULT_SMART_TURN_FALLBACK_SECS)
        self._allow_interruptions = _resolve_bool_extra(
            self._extra, "allow_interruptions", DEFAULT_ALLOW_INTERRUPTIONS)
        self._min_interruption_words = _resolve_nonneg_int_extra(
            self._extra, "min_interruption_words",
            DEFAULT_MIN_INTERRUPTION_WORDS)
        self._min_interruption_duration_ms = _resolve_nonneg_int_extra(
            self._extra, "min_interruption_duration_ms",
            DEFAULT_MIN_INTERRUPTION_DURATION_MS)
        # Detector + 8s rolling buffer built lazily (the real one pulls
        # onnxruntime); tests inject a fake via _build_turn_detector. Only
        # built when turn_detector="smart_turn".
        self._detector = None
        self._audio_buffer = None
        # Energy-silence onset state (LISTENING): how long inbound audio has
        # been continuously quiet since the last hot chunk, and the last time
        # a Smart Turn check ran (rate-limit). _smart_speech_seen gates the
        # detector to fire only after the user actually spoke this cycle.
        self._silence_run_started: Optional[float] = None
        self._last_smart_check_t: float = 0.0
        self._smart_speech_seen = False
        self._smart_check_inflight = False
        # Speculative-start bookkeeping: the speculative turn's transcript and
        # whether a speculative turn is currently live (so resumed speech can
        # restart it). Per-turn telemetry counters banked into the record.
        self._speculative_active = False
        self._speculative_text = ""
        self._spec_started_count = 0
        self._spec_restarted_count = 0
        # Barge-in gating (Smart Turn path): when the current run of inbound
        # energy during agent playback began (for min_interruption_duration_ms)
        # and the word count of the partial transcript heard during it.
        self._barge_energy_started: Optional[float] = None
        self._barge_partial_words = 0
        # Telemetry (notes §16): one structured record per turn cycle,
        # opened at the first cycle event (vamp fire or vad-end), emitted
        # as a JSON log line when the turn ends; per-call summary on stop.
        self._tel: Dict[str, Any] = {}
        self._call_stats: List[Dict[str, Any]] = []
        # VAD speech tracking: consecutive speech-positive steps drive the
        # fast barge-in trigger; _speech_seen lets end-of-turn finalize even
        # before the (slow) finalized text has arrived — flush forces it out.
        self._speech_steps = 0
        self._speech_seen = False
        self._energy_hot_chunks = 0
        # VAD barge-in requires a silence->speech TRANSITION within the
        # current turn: a freshly-opened ASR stream can report
        # speech-positive steps on pure silence for its first moments
        # (observed live 2026-06-12 — it cut the greeting short), and
        # those must never count as a barge-in.
        self._silence_seen_in_turn = False
        # Per-turn delta/tool sinks behind stable trampolines, so an agent
        # can be CONSTRUCTED before its turn starts (construction overlaps
        # the flush wait) and still stream into the right turn's queue.
        self._delta_sink = None
        self._tool_sink = None
        # Pre-constructed agent for the next turn (concurrent.futures.Future
        # from run_in_executor), made while waiting for the STT flush ack.
        self._spare_agent_future = None
        # One-element list so the executor thread can publish the live agent
        # for cross-thread interrupt (api_server agent_ref pattern, 3505-3508).
        self._agent_ref: List[Optional[Any]] = [None]
        # Per-turn interrupt latch: a barge-in landing while the executor
        # thread is still CONSTRUCTING the agent (agent_ref[0] unset) must
        # not be lost — _run checks it right after construction.
        self._interrupt_latch: Optional[threading.Event] = None
        self._tts = None
        self._turn_task: Optional[asyncio.Task] = None
        self._finalize_task: Optional[asyncio.Task] = None
        self._stopped = asyncio.Event()
        self._awaiting_flush_id: Optional[int] = None
        self._last_flushed_id: Optional[int] = None
        self._flushed = asyncio.Event()
        # Timing instrumentation: per-turn sequence + reference timestamp
        # (vad-end for user turns; turn-start for the greeting). All
        # voice/timing logs report ms since this reference.
        self._turn_seq = 0
        self._t_ref: Optional[float] = None

    def _mark(self, leg: str, **fields: Any) -> None:
        """INFO timing log: ms since the current turn's reference point."""
        now = time.monotonic()
        ref = self._t_ref if self._t_ref is not None else now
        extra = "".join(f" {k}={v}" for k, v in fields.items())
        logger.info("voice/timing turn=%d %s t=+%.0fms%s",
                    self._turn_seq, leg, (now - ref) * 1000.0, extra)

    # -- telemetry (notes §16) ------------------------------------------------

    def _tel_open(self) -> None:
        if not self._tel:
            self._tel = {"opened_at": time.monotonic()}
            if self._last_hot_t is not None:
                self._tel["speech_end"] = self._last_hot_t
            reset = getattr(self._transport, "reset_write_mark", None)
            if reset is not None:
                reset()

    def _tel_set(self, key: str, value: Any = None) -> None:
        """Record a telemetry timestamp (default: now) or value. First
        write wins — retries/restarts must not overwrite the first leg."""
        self._tel_open()
        self._tel.setdefault(key, time.monotonic() if value is None else value)

    def _tel_emit(self, status: str) -> None:
        """Emit the per-turn structured JSON log line and bank the record
        for the per-call summary. All offsets are ms since speech_end (the
        last energy-hot inbound chunk — the closest proxy for the end of
        the user's utterance; vad_end lags it by ~550ms live)."""
        tel, self._tel = self._tel, {}
        if not tel:
            return
        speech_end = tel.get("speech_end") or tel.get("vad_end") \
            or tel.get("opened_at")
        first_written = getattr(self._transport, "first_write_t", None)

        def off(t: Optional[float]) -> Optional[int]:
            return None if t is None else int(round((t - speech_end) * 1000))

        perceived = off(first_written)
        substantive = off(tel.get("tts_first_audio"))
        record = {
            "event": "voice_turn",
            "turn": self._turn_seq,
            "status": status,
            "eager_start": bool(tel.get("eager_start")),
            "turn_detector": tel.get("turn_detector", self._turn_detector_mode),
            "eot_reason": tel.get("eot_reason"),
            "speculative_started": bool(tel.get("speculative_started")),
            "speculative_restarted": bool(tel.get("speculative_restarted")),
            "vad_end_ms": off(tel.get("vad_end")),
            "flush_result": tel.get("flush_result"),
            "flush_done_ms": off(tel.get("flush_done")),
            "agent_start_ms": off(tel.get("agent_start")),
            "first_delta_ms": off(tel.get("first_delta")),
            "first_sentence_ms": off(tel.get("first_sentence")),
            "tts_first_audio_ms": substantive,
            "first_frame_written_ms": perceived,
            "vamp": {
                "fired": "vamp_fired_at" in tel,
                "fired_at_ms": off(tel.get("vamp_fired_at")),
                "trigger": tel.get("vamp_trigger"),
                "text": tel.get("vamp_text"),
                "false_fire": bool(tel.get("vamp_false_fire")),
                "veto_deferred_chunks": self._vamp_veto_deferred,
            },
            "totals": {
                "perceived_first_audio_ms": perceived,
                "substantive_first_audio_ms": substantive,
                "turn_total_ms": off(time.monotonic()),
            },
        }
        self._vamp_veto_deferred = 0
        emit_telemetry(record)
        self._call_stats.append(record)

    def _emit_call_summary(self) -> None:
        turns = [r for r in self._call_stats if r["vad_end_ms"] is not None]

        def stats(key: str) -> Optional[Dict[str, int]]:
            vals = sorted(r["totals"][key] for r in turns
                          if r["totals"][key] is not None)
            if not vals:
                return None
            return {"median": vals[len(vals) // 2],
                    "p90": vals[min(len(vals) - 1, int(len(vals) * 0.9))],
                    "n": len(vals)}

        raw_effort = self._extra.get("reasoning_effort")
        record = {
            "event": "voice_call_summary",
            "session": self._session_id,
            "turns": len(turns),
            # Effective per-turn config, so a live run is self-describing
            # (the 2.6s->~4s substantive regression in the verify run is
            # unexplainable from the telemetry alone when the model /
            # reasoning overrides are not recorded).
            "model_override": _resolve_turn_model(self._extra),
            "reasoning_effort": (str(raw_effort).strip()
                                 if raw_effort is not None
                                 and str(raw_effort).strip()
                                 else "gateway-default"),
            "eager_starts": sum(1 for r in turns if r["eager_start"]),
            "vamp_fired": sum(1 for r in turns if r["vamp"]["fired"]),
            "vamp_false_fires": sum(
                1 for r in self._call_stats if r["vamp"]["false_fire"]),
            "barge_ins": sum(1 for r in self._call_stats
                             if r["status"] == "interrupted"),
            "perceived_first_audio_ms": stats("perceived_first_audio_ms"),
            "substantive_first_audio_ms": stats("substantive_first_audio_ms"),
        }
        emit_telemetry(record)

    # -- vamp -----------------------------------------------------------------

    async def _fire_vamp(self, source: str) -> None:
        """Write one pre-synthesized acknowledgment clip to the transport.
        Skips (a missed vamp is fine — the pipeline covers) when: no cache,
        clips not ready yet, already fired this cycle, trigger disabled, or
        the substantive turn is already underway (state left LISTENING)."""
        if (self._vamp is None or not self._vamp_armed
                or self._vamp_trigger == "off"
                or self._state != LISTENING):
            return
        if not self._vamp.ready:
            self._mark("vamp-skip", reason="clips-not-ready", source=source)
            return
        picked = self._vamp.pick()
        if picked is None:
            return
        text, pcm = picked
        self._vamp_armed = False
        self._vamp_resume_hot = 0
        now = time.monotonic()
        duration_s = len(pcm) / 2.0 / 48000.0
        self._vamp_playing_until = now + duration_s + VAMP_PLAYOUT_SLACK_S
        self._tel_set("vamp_fired_at", now)
        self._tel_set("vamp_trigger", source)
        self._tel_set("vamp_text", text)
        self._mark("vamp-fired", source=source, dur_ms=int(duration_s * 1000),
                   text=repr(text))
        # 80ms chunks: barge-in's clear_output() drops unplayed audio at
        # chunk granularity, and the substantive reply naturally queues
        # behind the clip boundary.
        for offset in range(0, len(pcm), 7680):
            await self._transport.send_audio(pcm[offset:offset + 7680])

    def _cancel_vamp(self) -> None:
        """False fire: the user resumed speaking while the clip was still
        playing — drop the remaining (unplayed) clip audio."""
        self._vamp_playing_until = 0.0
        self._vamp_resume_hot = 0
        self._tel_set("vamp_false_fire", True)
        self._transport.clear_output()
        self._mark("vamp-false-fire-cancelled")

    # -- agent plumbing -----------------------------------------------------

    def _delta_tramp(self, delta: str) -> None:
        sink = self._delta_sink
        if sink is not None:
            sink(delta)

    def _tool_tramp(self, event_type, tool_name=None, preview=None,
                    args=None, **kwargs) -> None:
        sink = self._tool_sink
        if sink is not None:
            sink(event_type, tool_name=tool_name, preview=preview,
                 args=args, **kwargs)

    def _make_agent(self):
        """Construct a turn agent (executor thread). Construction does not
        need the user message, so it can run before the turn starts."""
        return _create_voice_agent(
            self._session_id, self._delta_tramp, self._tool_tramp,
            extra=self._extra)

    def _preconstruct_agent(self) -> "concurrent.futures.Future":
        """Build the next turn's agent on a worker thread. Returns a
        concurrent Future (thread-safe .result() from the turn executor)."""
        fut: "concurrent.futures.Future" = concurrent.futures.Future()

        def _build() -> None:
            try:
                fut.set_result(self._make_agent())
            except BaseException as e:    # surface in the consuming turn
                fut.set_exception(e)

        threading.Thread(
            target=_build, name="voice-agent-prewarm", daemon=True).start()
        return fut

    # -- main ---------------------------------------------------------------

    def _ensure_detector(self) -> None:
        """Build the Smart Turn detector + rolling buffer and warm the ONNX
        session on a worker thread. Model load (~hundreds of ms) overlaps the
        greeting so the first user turn never waits on it. Fail-soft: a build
        failure logs and leaves the detector None — _run_smart_turn then
        returns None and the hard fallback still finalizes turns."""
        if self._turn_detector_mode != "smart_turn" or self._detector is not None:
            return
        try:
            detector, buffer = _build_turn_detector()
        except Exception:
            logger.exception("voice/turn: smart-turn detector build failed; "
                             "end-of-turn falls back to the silence timeout")
            return
        self._detector = detector
        self._audio_buffer = buffer

        def _warm() -> None:
            try:
                detector.load()
            except Exception:
                logger.exception("voice/turn: smart-turn model warm-up failed")

        threading.Thread(target=_warm, name="voice-smart-turn-warm",
                         daemon=True).start()

    async def run(self) -> None:
        self._ensure_detector()
        consumer = asyncio.create_task(self._consume_stt())
        canned = self._extra.get("greeting_text")
        if canned:
            # Canned greeting: instant TTS, no LLM round-trip. The first
            # real turn's agent is pre-warmed while the greeting plays.
            self._spare_agent_future = self._preconstruct_agent()
            self._turn_task = asyncio.create_task(self._speak_canned(canned))
        else:
            # Agent speaks first (notes §14 item 8): greeting turn before
            # listening.
            self._start_turn(self._greeting, record_user=False)
        await self._stopped.wait()
        consumer.cancel()
        try:
            await consumer
        except (asyncio.CancelledError, Exception):
            pass

    async def stop(self) -> None:
        # Cancel a pending finalize BEFORE barge-in: barge-in awaits the
        # in-flight turn, and a still-live finalize could _start_turn in
        # that window, orphaning a fresh turn past teardown.
        if self._finalize_task is not None and not self._finalize_task.done():
            self._finalize_task.cancel()
            try:
                await self._finalize_task
            except (asyncio.CancelledError, Exception):
                pass
        await self._barge_in()           # kill any in-flight turn
        self._tel_emit("call-ended")
        self._emit_call_summary()
        self._stopped.set()

    async def _speak_canned(self, text: str) -> None:
        """Speak fixed text through TTS without an agent turn (greeting)."""
        self._t_ref = time.monotonic()
        self._mark("canned-greeting-start", chars=len(text))
        self._state = SPEAKING
        self._silence_seen_in_turn = False
        try:
            self._tts = await self._tts_factory(self._transport.send_audio)
            await self._tts.send_text(text)
            await asyncio.wait_for(self._tts.end(), timeout=TTS_END_TIMEOUT_S)
            self._history.append({"role": "assistant", "content": text})
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("voice/turn: canned greeting failed")
        finally:
            self._tts = None
            if self._state != LISTENING:
                self._state = LISTENING

    # -- inbound audio (fast energy barge-in) --------------------------------

    async def on_inbound_audio(self, pcm: bytes) -> None:
        """Called by the adapter for every inbound 80ms caller chunk, in
        parallel with STT. In Smart Turn mode this owns BOTH turn decisions:
          THINKING/SPEAKING: sustained energy (gated by min duration + the
            partial-transcript word count) = the user talking over the agent
            -> barge in WITHOUT the STT round-trip.
          LISTENING: the chunk feeds the rolling 16kHz buffer; continuous
            silence after speech (>= vad_stop_secs) runs the Smart Turn ONNX
            endpoint model -> speculative start / finalize / hard fallback.
            Quiet chunks after speech also fire the vamp clip (inverse
            trigger); hot chunks while a vamp clip plays = false fire."""
        hot = _rms(pcm) >= ENERGY_BARGE_RMS
        # Feed the rolling window (Smart Turn path only) for every chunk so
        # the 8s endpoint window is always current when a check runs.
        if self._turn_detector_mode == "smart_turn" and self._audio_buffer is not None:
            self._audio_buffer.push(pcm)
        if self._state in (THINKING, SPEAKING):
            self._utt_hot_chunks = 0
            self._utt_quiet_chunks = 0
            if self._turn_detector_mode == "gradium":
                # Fallback path: bare consecutive-hot-chunk barge-in.
                if not hot:
                    self._energy_hot_chunks = 0
                    return
                self._energy_hot_chunks += 1
                if self._energy_hot_chunks >= self._energy_barge_chunks:
                    self._energy_hot_chunks = 0
                    self._mark("barge-in-trigger", source="energy",
                               state=self._state)
                    await self._barge_in()
                return
            # Smart Turn path: energy starts/extends a barge run; the run
            # must clear BOTH min_interruption_duration_ms AND
            # min_interruption_words (from the partial transcript heard
            # during the run) before it counts as an interruption.
            now = time.monotonic()
            if hot:
                if self._barge_energy_started is None:
                    self._barge_energy_started = now
                    self._barge_partial_words = 0
                await self._maybe_energy_barge_in()
            else:
                # A brief dip is tolerated as long as the run is short; a
                # sustained gap (>= one vad_stop window) resets the run so a
                # cough does not accumulate toward an interruption.
                if (self._barge_energy_started is not None
                        and now - self._barge_energy_started
                        > self._min_interruption_duration_ms / 1000.0
                        + self._vad_stop_secs):
                    self._barge_energy_started = None
                    self._barge_partial_words = 0
            return
        self._energy_hot_chunks = 0
        self._barge_energy_started = None
        now = time.monotonic()
        if hot:
            self._last_hot_t = now
            self._utt_hot_chunks += 1
            self._utt_quiet_chunks = 0
            self._smart_speech_seen = True
            self._silence_run_started = None
            if now < self._vamp_playing_until:
                self._vamp_resume_hot += 1
                if self._vamp_resume_hot >= VAMP_CANCEL_HOT_CHUNKS:
                    self._cancel_vamp()
            return
        self._vamp_resume_hot = 0
        if self._utt_hot_chunks >= self._vamp_min_speech:
            self._utt_quiet_chunks += 1
            if self._vamp_energy_in_window():
                if self._vamp_step_veto_active(now):
                    self._vamp_veto_deferred += 1
                else:
                    await self._fire_vamp("energy")
        # Smart Turn end-of-turn: continuous local silence after speech runs
        # the endpoint model (rate-limited), or hits the hard fallback.
        if self._turn_detector_mode == "smart_turn":
            await self._smart_turn_on_silence(now)

    async def _maybe_energy_barge_in(self) -> None:
        """Smart Turn barge-in gate: interrupt the agent only when a run of
        inbound energy has lasted >= min_interruption_duration_ms AND the
        partial transcript heard during it has >= min_interruption_words.
        Kills false barge-in on a single um/cough/echo.

        When a SPECULATIVE turn is live, the same trigger means the user kept
        talking past the speculative cut: interrupt and RESTART the turn with
        the fuller transcript instead of just ending it (speculative_restarted
        telemetry). For a normal turn it is a plain barge-in."""
        if (not self._allow_interruptions
                or self._state not in (THINKING, SPEAKING)
                or self._barge_energy_started is None):
            return
        elapsed_ms = (time.monotonic() - self._barge_energy_started) * 1000.0
        if (elapsed_ms < self._min_interruption_duration_ms
                or self._barge_partial_words < self._min_interruption_words):
            return
        words = self._barge_partial_words
        self._barge_energy_started = None
        self._barge_partial_words = 0
        if self._speculative_active:
            await self._restart_speculative_turn(words, int(elapsed_ms))
            return
        self._mark("barge-in-trigger", source="smart-energy",
                   state=self._state, words=words, ms=int(elapsed_ms))
        await self._barge_in()

    async def _restart_speculative_turn(self, words: int, ms: int) -> None:
        """The user kept speaking past a speculative start: interrupt the
        speculative turn (which has not produced substantive audio the user
        would notice cut — that is the whole point of the lower threshold)
        and restart on the accumulated transcript. The interrupted turn's
        _execute_turn re-queues its (unspoken) user_message into _pending_text,
        so the restart sees the full utterance."""
        self._spec_restarted_count += 1
        self._tel_set("speculative_restarted", True)
        self._mark("speculative-restart", words=words, ms=ms)
        await self._barge_in()
        # _execute_turn's interrupt path re-inserts the speculative
        # user_message at the head of _pending_text when no audio was heard.
        text = " ".join(t for t in self._pending_text if t).strip()
        self._pending_text.clear()
        self._speech_seen = False
        self._speech_steps = 0
        self._speculative_active = False
        self._smart_speech_seen = False
        self._silence_run_started = None
        if not text:
            return
        agent_future = self._spare_agent_future
        self._spare_agent_future = None
        self._start_turn(text, record_user=True, agent_future=agent_future)

    async def _smart_turn_on_silence(self, now: float) -> None:
        """Drive end-of-turn from local silence + the Smart Turn ONNX model.

        Runs only while LISTENING with no finalize in flight, after the user
        has actually spoken this cycle. Once silence has held vad_stop_secs,
        run the endpoint model (rate-limited to SMART_TURN_MIN_INTERVAL_S):
          - prob >= smart_turn_threshold -> finalize the turn;
          - prob >= speculative_threshold (and not yet started) -> start the
            agent speculatively on the transcript-so-far (restart handled by
            the resumed-speech path / finalize);
          - silence past smart_turn_fallback_secs -> finalize regardless
            (covers a Smart Turn false-negative / a trailing-off utterance)."""
        if (self._state != LISTENING or self._finalize_task is not None
                or not self._smart_speech_seen):
            return
        if self._silence_run_started is None:
            self._silence_run_started = now
        silence_held = now - self._silence_run_started
        if silence_held < self._vad_stop_secs:
            return
        # Hard fallback: finalize regardless of the model's verdict.
        if silence_held >= self._smart_turn_fallback_secs:
            self._mark("smart-turn-fallback", silence_ms=int(silence_held * 1000))
            self._begin_finalize(reason="fallback")
            return
        if (now - self._last_smart_check_t < SMART_TURN_MIN_INTERVAL_S
                or self._smart_check_inflight):
            return
        self._last_smart_check_t = now
        prob = await self._run_smart_turn()
        if prob is None:
            return
        if prob >= self._smart_turn_threshold:
            self._mark("smart-turn-complete", prob=round(prob, 3))
            self._begin_finalize(reason="endpoint")
        elif (self._speculative_start and not self._speculative_active
                and prob >= self._speculative_threshold):
            self._begin_speculative_start(prob)

    async def _run_smart_turn(self) -> Optional[float]:
        """Run the ONNX endpoint model on the current 16kHz window in the
        executor. Returns the probability, or None when the detector is
        unavailable / the window is empty (treated as not-end-of-turn)."""
        if self._detector is None or self._audio_buffer is None:
            return None
        window = self._audio_buffer.window_16k()
        if window is None or len(window) == 0:
            return None
        self._smart_check_inflight = True
        try:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None, self._detector.predict_endpoint, window)
        except Exception:
            logger.exception("voice/turn: smart-turn inference failed")
            return None
        finally:
            self._smart_check_inflight = False

    def _begin_finalize(self, *, reason: str) -> None:
        """Open the turn-cycle telemetry record and launch _finalize_user_turn
        (Smart Turn path). Mirrors the bookkeeping the Gradium vad-end block
        did inline, then resets the per-cycle silence/speech gate so the next
        utterance starts clean."""
        self._turn_seq += 1
        self._t_ref = time.monotonic()
        self._tel_set("vad_end", self._t_ref)
        self._tel_set("turn_detector", "smart_turn")
        self._tel_set("eot_reason", reason)
        if self._last_hot_t is not None:
            self._tel["speech_end"] = self._last_hot_t
        self._mark("smart-turn-end", reason=reason)
        self._smart_speech_seen = False
        self._silence_run_started = None
        if self._audio_buffer is not None:
            self._audio_buffer.clear()
        self._finalize_task = asyncio.create_task(self._finalize_user_turn())

    def _begin_speculative_start(self, prob: float) -> None:
        """Speculative start: begin the agent turn on the transcript-so-far
        before full end-of-turn confidence, overlapping LLM TTFT with the
        tail of user speech. The turn stays cancellable: if more speech
        arrives (energy resumes -> a new utterance cycle), finalize/​barge-in
        interrupts and restarts it with the fuller transcript."""
        text = " ".join(t for t in self._pending_text if t).strip()
        if not text:
            return                       # nothing to speculate on yet
        self._turn_seq += 1
        self._t_ref = time.monotonic()
        self._tel_set("vad_end", self._t_ref)
        self._tel_set("turn_detector", "smart_turn")
        self._tel_set("speculative_started", True)
        if self._last_hot_t is not None:
            self._tel["speech_end"] = self._last_hot_t
        self._speculative_active = True
        self._speculative_text = text
        self._spec_started_count += 1
        self._smart_speech_seen = False
        # Clear the accumulated transcript: the speculative turn now OWNS this
        # text. A restart rebuilds the full utterance from the speculative
        # text (re-queued by _execute_turn on interrupt) + any tail that
        # arrives after — see the text handler / _restart_speculative_turn.
        self._pending_text.clear()
        self._speech_seen = False
        self._speech_steps = 0
        self._mark("speculative-start", prob=round(prob, 3), chars=len(text))
        agent_future = self._spare_agent_future
        self._spare_agent_future = None
        # record_user is deferred to the final (possibly restarted) turn so a
        # restart does not double-log the user utterance.
        self._start_turn(text, record_user=True, agent_future=agent_future)

    def _vamp_energy_in_window(self) -> bool:
        """True while the energy trigger may (still) fire this cycle."""
        return (self._vamp_trigger == "energy" and self._vamp_armed
                and self._vamp_silence_chunks <= self._utt_quiet_chunks
                <= VAMP_ENERGY_MAX_SILENCE_CHUNKS)

    def _vamp_step_veto_active(self, now: float) -> bool:
        """A FRESH speech-positive VAD step asserts the user is
        mid-utterance (e.g. an intra-utterance pause that semantic VAD
        sees through) — defer the energy fire until a silence-confirming
        step lands or the assertion goes stale."""
        t_speech = self._last_step_speech_t
        if t_speech is None or now - t_speech > VAMP_VETO_FRESH_S:
            return False
        t_quiet = self._last_step_quiet_t
        return t_quiet is None or t_quiet < t_speech

    # -- STT event pump -----------------------------------------------------

    async def _consume_stt(self) -> None:
        async for msg in self._stt.events():
            mtype = msg.get("type")
            if mtype == "text":
                text = msg.get("text", "")
                # Smart Turn path: a partial transcript heard DURING agent
                # playback feeds the barge-in word gate (energy alone is
                # cough/echo-prone). It must not by itself trigger a barge —
                # the energy detector owns that decision once the word +
                # duration thresholds are both met.
                if self._turn_detector_mode == "smart_turn":
                    if (self._state in (THINKING, SPEAKING)
                            and self._finalize_task is None):
                        self._barge_partial_words += len(text.split())
                        # A speculative turn used the transcript-so-far; a
                        # transcript TAIL that arrives after it started proves
                        # the user kept talking past the speculative cut, even
                        # if the energy run already lapsed. Restart on the
                        # fuller utterance so the speculative generation is not
                        # answering a truncated question.
                        if self._speculative_active:
                            self._pending_text.append(text)
                            full = " ".join(
                                t for t in self._pending_text if t).strip()
                            if (full != self._speculative_text
                                    and len(text.split())
                                    >= self._min_interruption_words):
                                await self._restart_speculative_turn(
                                    len(text.split()), 0)
                            continue
                        await self._maybe_energy_barge_in()
                else:
                    # Gradium path: finalized text during THINKING/SPEAKING is
                    # an interjection. While finalize awaits the flush ack,
                    # text events are the forced tail of the utterance that
                    # ALREADY started the eager turn — not a new interjection.
                    if (self._state in (THINKING, SPEAKING)
                            and self._finalize_task is None):
                        self._mark("barge-in-trigger", source="stt-text",
                                   state=self._state)
                        await self._barge_in()
                self._pending_text.append(text)
            elif mtype == "step":
                await self._stt.maybe_rotate(msg)
                # Horizons looked up by value (FOUR arrive live, §15).
                probs = {v.get("horizon_s"): v.get("inactivity_prob", 0.0)
                         for v in msg.get("vad", [])}
                if os.getenv("VOICE_DEBUG_VAD") and any(
                        0.02 < p < 0.98 for p in probs.values()):
                    self._mark("vad-step", probs={k: round(v, 3)
                                                  for k, v in probs.items()})
                # Fast barge-in: short-horizon inactivity collapsing means
                # the user is speaking NOW — do not wait for finalized text
                # (which can lag seconds during overlapping speech).
                p_short = probs.get(BARGE_VAD_HORIZON)
                if p_short is not None and p_short >= 0.9:
                    self._silence_seen_in_turn = True
                # Dual-confirm bookkeeping for the energy vamp trigger,
                # plus the catch-up fire: a silence-confirming step
                # (including the vad-end step itself) releases a fire the
                # veto deferred — the vamp can be late, never absent.
                if p_short is not None:
                    t_step = time.monotonic()
                    if p_short <= VAMP_VETO_SPEECH_PROB:
                        self._last_step_speech_t = t_step
                    elif p_short >= VAMP_VETO_CLEAR_PROB:
                        self._last_step_quiet_t = t_step
                        if (self._state == LISTENING
                                and self._vamp_energy_in_window()):
                            await self._fire_vamp("energy")
                # Vamp on the FIRST short-horizon VAD crossing (independent
                # of the turn detector — vamp_trigger="vad" still works under
                # Smart Turn): fires ahead of the end-of-turn confirmation.
                if (self._vamp_trigger == "vad"
                        and self._state == LISTENING
                        and self._finalize_task is None
                        and (self._pending_text or self._smart_speech_seen)
                        and probs.get(VAMP_VAD_HORIZON, 0.0) >= VAMP_VAD_PROB):
                    await self._fire_vamp("vad")
                # Gradium step-driven barge-in + end-of-turn are REPLACED by
                # Smart Turn (ENG-555); kept only behind turn_detector=gradium
                # as a fallback. The Smart Turn path drives both from the
                # local energy gate + ONNX endpoint model in on_inbound_audio.
                if self._turn_detector_mode != "gradium":
                    continue
                if p_short is not None and p_short <= BARGE_VAD_SPEECH_PROB:
                    self._speech_steps += 1
                else:
                    self._speech_steps = 0
                if self._speech_steps >= BARGE_VAD_CONSEC_STEPS:
                    self._speech_seen = True
                    if (self._state in (THINKING, SPEAKING)
                            and self._silence_seen_in_turn):
                        self._mark("barge-in-trigger", source="vad-step",
                                   state=self._state)
                        await self._barge_in()
                if ((self._pending_text or self._speech_seen)
                        and self._state == LISTENING
                        and self._finalize_task is None):
                    if probs.get(END_OF_TURN_HORIZON, 0.0) >= END_OF_TURN_PROB:
                        self._turn_seq += 1
                        self._t_ref = time.monotonic()
                        self._tel_set("vad_end", self._t_ref)
                        if self._last_hot_t is not None:
                            # vad-end is authoritative for the cycle: a
                            # false-fired vamp opened the record early, so
                            # refresh the speech-end proxy.
                            self._tel["speech_end"] = self._last_hot_t
                        self._mark("vad-end-detected",
                                   probs={k: round(v, 3) for k, v in probs.items()})
                        # Run finalize as a task: it waits for the "flushed"
                        # ack, which only THIS consumer loop can deliver.
                        # (Inlining it here deadlocks the ack path and would
                        # turn every user turn into a full flush-timeout.)
                        self._finalize_task = asyncio.create_task(
                            self._finalize_user_turn())
            elif mtype == "flushed":
                self._last_flushed_id = msg.get("flush_id")
                self._mark("flushed-raw", msg=msg,
                           awaiting=self._awaiting_flush_id)
                if self._last_flushed_id == self._awaiting_flush_id:
                    self._flushed.set()
            elif mtype == "error":
                logger.warning("voice/turn: stt error event: %s", msg.get("message"))

    async def _finalize_user_turn(self) -> None:
        try:
            # Pre-construct the turn agent NOW so the ~0.5s construction
            # overlaps the flush round-trip instead of following it.
            if self._spare_agent_future is None:
                self._mark("agent-preconstruct-start")
                self._spare_agent_future = self._preconstruct_agent()
            self._flushed.clear()
            self._awaiting_flush_id = None
            # EAGER START: when the transcript at vad-end already ends in
            # terminal punctuation it is (almost always) complete — start
            # the turn NOW instead of serializing the ~370ms flush
            # round-trip. Punctuation-less transcripts are missing their
            # tail (the ASR delay is ~800ms vs the ~550ms vad-end lag —
            # measured live: a tail arrived on every fast turn) and an
            # eager start would only buy an interrupt+restart, so those
            # wait for the flush. If a tail arrives despite the
            # punctuation, the eager turn is interrupted pre-audio and
            # restarted with the full utterance.
            eager_text = " ".join(t for t in self._pending_text if t).strip()
            if eager_text and eager_text[-1] not in ".!?":
                eager_text = ""
            if eager_text:
                self._pending_text.clear()
                self._speech_seen = False
                self._speech_steps = 0
                self._tel_set("eager_start", True)
                agent_future = self._spare_agent_future
                self._spare_agent_future = None
                self._start_turn(eager_text, record_user=True,
                                 agent_future=agent_future)
            flush_id = await self._stt.flush()
            self._mark("flush-sent", flush_id=flush_id)
            if flush_id is not None:
                self._awaiting_flush_id = flush_id
                # The ack may have been consumed between flush() and the
                # assignment above — _last_flushed_id catches that window.
                if self._last_flushed_id != flush_id:
                    try:
                        await asyncio.wait_for(
                            self._flushed.wait(), timeout=FLUSHED_WAIT_TIMEOUT_S)
                        self._mark("flushed-ack", flush_id=flush_id)
                        self._tel_set("flush_result", "ack")
                    except asyncio.TimeoutError:
                        self._mark("flushed-TIMEOUT", flush_id=flush_id,
                                   last_seen=self._last_flushed_id)
                        self._tel_set("flush_result", "timeout")
                else:
                    self._tel_set("flush_result", "ack")
            else:
                # No live STT session — no "flushed" event will ever
                # arrive (notes §15 addendum), skip the wait entirely.
                self._tel_set("flush_result", "skipped")
            self._tel_set("flush_done")
            late_text = " ".join(t for t in self._pending_text if t).strip()
            if eager_text and not late_text:
                return                    # eager turn was complete — done
            if eager_text and late_text:
                if self._turn_task is None or self._turn_task.done():
                    # The eager turn already ended — a real barge-in killed
                    # it (re-queueing eager_text into pending). Leave the
                    # accumulated text for the normal flow; restarting here
                    # would duplicate the utterance mid-speech.
                    return
                # Rare: the flush delivered a transcript tail. Restart the
                # turn with the full utterance (the eager turn cannot have
                # spoken yet; we rebuild the utterance explicitly).
                self._mark("eager-restart", extra_chars=len(late_text))
                await self._barge_in()
                self._pending_text.clear()
                self._speech_seen = False
                self._speech_steps = 0
                agent_future = self._spare_agent_future
                self._spare_agent_future = None
                self._start_turn(f"{eager_text} {late_text}",
                                 record_user=True, agent_future=agent_future)
                return
            utterance = late_text
            self._pending_text.clear()
            self._speech_seen = False
            self._speech_steps = 0
            if utterance:
                agent_future = self._spare_agent_future
                self._spare_agent_future = None
                self._start_turn(utterance, record_user=True,
                                 agent_future=agent_future)
            else:
                # Cycle ended with no turn (keep the spare agent for the
                # next finalize): close out the telemetry record and let
                # the vamp fire again next utterance.
                self._tel_emit("no-turn")
                self._vamp_armed = True
        finally:
            self._finalize_task = None

    # -- agent turn ---------------------------------------------------------

    def _start_turn(self, user_message: str, *, record_user: bool,
                    agent_future=None) -> None:
        self._state = THINKING
        self._silence_seen_in_turn = False
        if self._t_ref is None:          # greeting turn has no vad-end
            self._t_ref = time.monotonic()
        self._mark("turn-start", chars=len(user_message))
        # Fresh latch BEFORE the task exists so a barge-in can never land
        # between turn creation and the executor publishing the agent.
        self._interrupt_latch = threading.Event()
        self._turn_task = asyncio.create_task(
            self._execute_turn(user_message, record_user=record_user,
                               agent_future=agent_future))

    async def _execute_turn(self, user_message: str, *, record_user: bool,
                            agent_future=None) -> None:
        loop = asyncio.get_running_loop()
        q: "asyncio.Queue[tuple]" = asyncio.Queue()
        agent_ref = self._agent_ref
        agent_ref[0] = None
        interrupt_latch = self._interrupt_latch

        # Threadsafe marshal — mirrors api_server._enqueue (1619-1631).
        def _enqueue(item: tuple) -> None:
            try:
                running_loop = asyncio.get_running_loop()
            except RuntimeError:
                running_loop = None
            try:
                if running_loop is loop:
                    q.put_nowait(item)
                else:
                    loop.call_soon_threadsafe(q.put_nowait, item)
            except RuntimeError:
                pass

        # Mirrors api_server._delta (1633-1635).
        first_delta_seen = threading.Event()

        def _delta(delta: str) -> None:
            if delta:
                if not first_delta_seen.is_set():
                    first_delta_seen.set()
                    self._mark("first-delta")
                    self._tel_set("first_delta")
                _enqueue(("delta", delta))

        # Signature mirrors api_server._tool_progress (1637).
        def _tool_progress(event_type, tool_name=None, preview=None,
                           args=None, **kwargs) -> None:
            if event_type == "tool.started":
                _enqueue(("tool", tool_name))

        # Route the stable trampolines into THIS turn's queue. One live
        # turn at a time, so plain assignment is safe.
        self._delta_sink = _delta
        self._tool_sink = _tool_progress

        def _run():
            # Executor body mirrors api_server._run_agent (3510-3554).
            if agent_future is not None:
                # Pre-constructed in finalize; bounded so a hung
                # construction can never wedge the turn executor forever.
                agent = agent_future.result(timeout=RUN_FUTURE_TIMEOUT_S)
            else:
                self._mark("agent-constructing")
                agent = self._make_agent()
            agent_ref[0] = agent
            self._mark("agent-start")
            self._tel_set("agent_start")
            if interrupt_latch.is_set():
                # A barge-in/stop landed while the agent was still being
                # constructed: interrupt before the first iteration runs.
                try:
                    agent.interrupt("user barge-in (voice)")
                except Exception:
                    pass
            try:
                return agent.run_conversation(
                    user_message=user_message,
                    conversation_history=list(self._history),
                    task_id=self._session_id,
                )
            finally:
                _enqueue(("done", None))

        run_future = loop.run_in_executor(None, _run)
        # Pre-warm the NEXT turn's agent while this one runs: construction
        # (~0.5s of config/toolset loading) drops off the critical path of
        # every subsequent turn.
        if self._spare_agent_future is None:
            self._spare_agent_future = self._preconstruct_agent()

        interrupted = False
        first_audio_seen = False

        async def _on_audio(pcm: bytes) -> None:
            nonlocal first_audio_seen
            if not first_audio_seen:
                first_audio_seen = True
                self._mark("first-tts-audio-chunk", bytes=len(pcm))
                self._tel_set("tts_first_audio")
            await self._transport.send_audio(pcm)

        try:
            self._tts = await self._tts_factory(_on_audio)
            self._mark("tts-socket-open")
            self._state = SPEAKING
            await self._pump_deltas_to_tts(q)
            try:
                # end() blocks until the server's final audio; bounded wait
                # per the GradiumTTSTurn.end docstring.
                await asyncio.wait_for(self._tts.end(), timeout=TTS_END_TIMEOUT_S)
            except asyncio.TimeoutError:
                logger.warning("voice/turn: tts end timed out; aborting socket")
                await self._tts.abort()
        except asyncio.CancelledError:
            interrupted = True
            raise
        finally:
            self._tts = None
            if interrupted and agent_ref[0] is not None:
                # The latch may have raced agent construction; now that the
                # agent surely exists, re-issue the direct interrupt so the
                # bounded await below resolves promptly.
                try:
                    agent_ref[0].interrupt("user barge-in (voice)")
                except Exception:
                    pass
            try:
                result = await asyncio.wait_for(
                    run_future, timeout=RUN_FUTURE_TIMEOUT_S)
            except asyncio.TimeoutError:
                logger.error(
                    "voice/turn: agent run did not finish within %.0fs after "
                    "the turn ended; abandoning executor thread (it may "
                    "still be running and burning tokens)",
                    RUN_FUTURE_TIMEOUT_S)
                result = None
            except Exception:
                logger.exception("voice/turn: agent run failed")
                result = None
            if result and isinstance(result, dict) and not interrupted:
                final = result.get("final_response", "")
                if record_user:
                    self._history.append({"role": "user", "content": user_message})
                if final:
                    self._history.append({"role": "assistant", "content": final})
            elif interrupted and record_user and not first_audio_seen:
                # Barged in before the user heard ANY of the reply: their
                # words must not vanish — feed them into the next turn.
                self._pending_text.insert(0, user_message)
            agent_ref[0] = None
            self._delta_sink = None
            self._tool_sink = None
            if self._state != LISTENING:
                self._state = LISTENING
            # Clear the speculative flag once the (possibly restarted) turn
            # ends: a speculative turn that ran to completion needs the flag
            # cleared so the next utterance is not mistaken for a restart.
            self._speculative_active = False
            self._barge_energy_started = None
            self._barge_partial_words = 0
            self._tel_emit("interrupted" if interrupted else "ok")
            self._vamp_armed = True

    async def _pump_deltas_to_tts(self, q: "asyncio.Queue[tuple]") -> None:
        """Stream deltas to TTS at word granularity.

        Gradium only synthesizes on sentence-final punctuation (or an
        explicit <flush>), so forwarding words as they arrive lets audio
        start the moment the model emits the first '.', instead of after
        the whole reply (live test 2026-06-12: first audio ~350ms after
        the sentence-final token, regardless of message granularity).
        Words are never split across messages (Gradium inserts whitespace
        BETWEEN messages); ``buf`` holds at most one partial word.
        ``unsynthesized`` tracks text the server is holding without a
        sentence boundary — a long punctuation-free stretch is force-flushed
        on idle (mirrors the old LONG_FLUSH_LEN behavior, server-side)."""
        buf = ""
        unsynthesized = 0
        filler_spoken = False
        first_text_sent = False
        first_send_t: Optional[float] = None
        punct_ever = False
        first_flush_done = False

        def _mark_first_send(kind_: str, text: str) -> None:
            nonlocal first_text_sent, first_send_t
            if not first_text_sent:
                first_text_sent = True
                first_send_t = time.monotonic()
                self._mark("first-text-to-tts",
                           kind=kind_, chars=len(text))

        async def _send(fragment: str, kind_: str) -> None:
            nonlocal unsynthesized, punct_ever
            if not fragment:
                return
            _mark_first_send(kind_, fragment)
            await self._tts.send_text(fragment)
            m = None
            for m in _PUNCT_RE.finditer(fragment):
                pass
            if m is not None:
                if not punct_ever:
                    self._tel_set("first_sentence")
                punct_ever = True
                unsynthesized = len(fragment) - m.end()
            else:
                unsynthesized += len(fragment)

        async def _maybe_first_flush() -> None:
            # First-audio guard: if the model's first sentence is dragging
            # on with no sentence-final punctuation, force one <flush> so
            # the user hears SOMETHING (~350ms later) instead of dead air.
            nonlocal first_flush_done, unsynthesized
            if (first_send_t is not None and not punct_ever
                    and not first_flush_done and unsynthesized > 0
                    and time.monotonic() - first_send_t
                    > FIRST_SENTENCE_FLUSH_S):
                first_flush_done = True
                self._mark("first-sentence-forced-flush",
                           held_chars=unsynthesized)
                self._tel_set("first_sentence")
                await self._tts.send_text("<flush>")
                unsynthesized = 0

        while True:
            try:
                kind, payload = await asyncio.wait_for(q.get(), timeout=0.2)
            except asyncio.TimeoutError:
                await _maybe_first_flush()
                if buf and len(buf) + unsynthesized > LONG_FLUSH_LEN:
                    await _send(buf, "long-flush")
                    buf = ""
                if unsynthesized > LONG_FLUSH_LEN:
                    await self._tts.send_text("<flush>")
                    unsynthesized = 0
                continue
            if kind == "tool":
                if not filler_spoken:
                    filler_spoken = True
                    _mark_first_send("filler", self._filler)
                    await self._tts.send_filler(self._filler)
            elif kind == "delta":
                buf += payload
                # Send everything up to the last whitespace (complete
                # words); keep the trailing partial word.
                cut = max(buf.rfind(" "), buf.rfind("\n"), buf.rfind("\t"))
                if cut >= 0:
                    await _send(buf[:cut + 1], "words")
                    buf = buf[cut + 1:]
                await _maybe_first_flush()
            elif kind == "done":
                if buf.strip():
                    await _send(buf, "done-tail")
                return

    # -- barge-in -----------------------------------------------------------

    async def _barge_in(self) -> None:
        t0 = time.monotonic()
        # Latch first: if the executor thread is still constructing the
        # agent, _run picks this up right after construction and interrupts
        # before the first iteration (the direct path below would miss it).
        latch = self._interrupt_latch
        if latch is not None:
            latch.set()
        agent = self._agent_ref[0]
        if agent is not None:
            try:
                agent.interrupt("user barge-in (voice)")   # api_server.py:4091
            except Exception:
                pass
        tts = self._tts
        if tts is not None:
            # Mute synchronously FIRST: no further TTS audio may reach the
            # transport while abort()'s socket close is in flight.
            tts.mute()
        self._transport.clear_output()
        logger.info("voice/timing barge-in audio-cleared in %.0fms",
                    (time.monotonic() - t0) * 1000.0)
        if tts is not None:
            await tts.abort()
        if self._turn_task is not None and not self._turn_task.done():
            self._turn_task.cancel()
            try:
                await self._turn_task
            except (asyncio.CancelledError, Exception):
                pass
        self._turn_task = None
        self._state = LISTENING
        logger.info("voice/turn: barge-in handled in %.0fms",
                    (time.monotonic() - t0) * 1000.0)
