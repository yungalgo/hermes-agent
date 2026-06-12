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
ENERGY_BARGE_CHUNKS = 2
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


def _rms(pcm: bytes) -> float:
    """RMS of s16le mono PCM (audioop-free: removed in Python 3.13).
    Subsamples every 4th frame — plenty for an 80ms energy gate."""
    samples = memoryview(pcm).cast("h")[::4]
    if len(samples) == 0:
        return 0.0
    return (sum(s * s for s in samples) / len(samples)) ** 0.5


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
        model=_resolve_gateway_model(),
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
        reasoning_config=GatewayRunner._load_reasoning_config(),
    )


class VoiceTurnLoop:
    def __init__(self, stt, tts_factory, transport, *, extra: Dict[str, Any]):
        """tts_factory(on_audio) -> opened GradiumTTSTurn (one per turn)."""
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

    async def run(self) -> None:
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
        parallel with STT. Sustained energy while the agent is
        thinking/speaking barges in WITHOUT waiting for the STT round-trip
        (0.6-1.8s measured live)."""
        if self._state not in (THINKING, SPEAKING):
            self._energy_hot_chunks = 0
            return
        if _rms(pcm) >= ENERGY_BARGE_RMS:
            self._energy_hot_chunks += 1
        else:
            self._energy_hot_chunks = 0
            return
        if self._energy_hot_chunks >= ENERGY_BARGE_CHUNKS:
            self._energy_hot_chunks = 0
            self._mark("barge-in-trigger", source="energy", state=self._state)
            await self._barge_in()

    # -- STT event pump -----------------------------------------------------

    async def _consume_stt(self) -> None:
        async for msg in self._stt.events():
            mtype = msg.get("type")
            if mtype == "text":
                # While finalize awaits the flush ack, text events are the
                # forced tail of the utterance that ALREADY started the
                # eager turn — not a new interjection. Finalize handles the
                # restart; real interruptions in that window are caught by
                # the (faster) VAD-step trigger below.
                if (self._state in (THINKING, SPEAKING)
                        and self._finalize_task is None):
                    self._mark("barge-in-trigger", source="stt-text",
                               state=self._state)
                    await self._barge_in()
                self._pending_text.append(msg.get("text", ""))
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
                    except asyncio.TimeoutError:
                        self._mark("flushed-TIMEOUT", flush_id=flush_id,
                                   last_seen=self._last_flushed_id)
            # else: no live STT session — no "flushed" event will ever
            # arrive (notes §15 addendum), skip the wait entirely.
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
            # else: keep the spare agent for the next finalize.
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
