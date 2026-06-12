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
import logging
import os
import re
import threading
import uuid
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Mirrors tools/tts_tool.py:2426 / 2524 / 2525.
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])(?:\s|\n)|(?:\n\n)")
MIN_SENTENCE_LEN = 20
LONG_FLUSH_LEN = 100

END_OF_TURN_HORIZON = 2.0
END_OF_TURN_PROB = 0.7
FLUSHED_WAIT_TIMEOUT_S = 2.0
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

    # -- main ---------------------------------------------------------------

    async def run(self) -> None:
        consumer = asyncio.create_task(self._consume_stt())
        # Agent speaks first (notes §14 item 8): greeting turn before listening.
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

    # -- STT event pump -----------------------------------------------------

    async def _consume_stt(self) -> None:
        async for msg in self._stt.events():
            mtype = msg.get("type")
            if mtype == "text":
                if self._state in (THINKING, SPEAKING):
                    await self._barge_in()
                self._pending_text.append(msg.get("text", ""))
            elif mtype == "step":
                await self._stt.maybe_rotate(msg)
                if (self._pending_text and self._state == LISTENING
                        and self._finalize_task is None):
                    # Horizons looked up by value (FOUR arrive live, §15).
                    probs = {v.get("horizon_s"): v.get("inactivity_prob", 0.0)
                             for v in msg.get("vad", [])}
                    if probs.get(END_OF_TURN_HORIZON, 0.0) >= END_OF_TURN_PROB:
                        # Run finalize as a task: it waits for the "flushed"
                        # ack, which only THIS consumer loop can deliver.
                        # (Inlining it here deadlocks the ack path and would
                        # turn every user turn into a full flush-timeout.)
                        self._finalize_task = asyncio.create_task(
                            self._finalize_user_turn())
            elif mtype == "flushed":
                self._last_flushed_id = msg.get("flush_id")
                if self._last_flushed_id == self._awaiting_flush_id:
                    self._flushed.set()
            elif mtype == "error":
                logger.warning("voice/turn: stt error event: %s", msg.get("message"))

    async def _finalize_user_turn(self) -> None:
        try:
            self._flushed.clear()
            self._awaiting_flush_id = None
            flush_id = await self._stt.flush()
            if flush_id is not None:
                self._awaiting_flush_id = flush_id
                # The ack may have been consumed between flush() and the
                # assignment above — _last_flushed_id catches that window.
                if self._last_flushed_id != flush_id:
                    try:
                        await asyncio.wait_for(
                            self._flushed.wait(), timeout=FLUSHED_WAIT_TIMEOUT_S)
                    except asyncio.TimeoutError:
                        pass              # proceed with what we have
            # else: no live STT session — no "flushed" event will ever
            # arrive (notes §15 addendum), skip the wait entirely.
            utterance = " ".join(t for t in self._pending_text if t).strip()
            self._pending_text.clear()
            if utterance:
                self._start_turn(utterance, record_user=True)
        finally:
            self._finalize_task = None

    # -- agent turn ---------------------------------------------------------

    def _start_turn(self, user_message: str, *, record_user: bool) -> None:
        self._state = THINKING
        # Fresh latch BEFORE the task exists so a barge-in can never land
        # between turn creation and the executor publishing the agent.
        self._interrupt_latch = threading.Event()
        self._turn_task = asyncio.create_task(
            self._execute_turn(user_message, record_user=record_user))

    async def _execute_turn(self, user_message: str, *, record_user: bool) -> None:
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
        def _delta(delta: str) -> None:
            if delta:
                _enqueue(("delta", delta))

        # Signature mirrors api_server._tool_progress (1637).
        def _tool_progress(event_type, tool_name=None, preview=None,
                           args=None, **kwargs) -> None:
            if event_type == "tool.started":
                _enqueue(("tool", tool_name))

        def _run():
            # Executor body mirrors api_server._run_agent (3510-3554).
            agent = _create_voice_agent(
                self._session_id, _delta, _tool_progress, extra=self._extra)
            agent_ref[0] = agent
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

        interrupted = False
        try:
            self._tts = await self._tts_factory(self._transport.send_audio)
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
            agent_ref[0] = None
            if self._state != LISTENING:
                self._state = LISTENING

    async def _pump_deltas_to_tts(self, q: "asyncio.Queue[tuple]") -> None:
        """Sentence buffering — mirrors tts_tool.py:2599-2641."""
        buf = ""
        filler_spoken = False
        while True:
            try:
                kind, payload = await asyncio.wait_for(q.get(), timeout=0.5)
            except asyncio.TimeoutError:
                if len(buf) > LONG_FLUSH_LEN:           # tts_tool.py:2604-2607
                    await self._tts.send_text(buf)
                    buf = ""
                continue
            if kind == "tool":
                if not filler_spoken:
                    filler_spoken = True
                    await self._tts.send_filler(self._filler)
            elif kind == "delta":
                buf += payload
                while True:                              # tts_tool.py:2630-2641
                    m = _SENTENCE_BOUNDARY_RE.search(buf)
                    if m is None:
                        break
                    sentence, buf = buf[:m.end()], buf[m.end():]
                    # Merge short fragments into the next sentence
                    # (tts_tool.py:2637-2640).
                    if len(sentence.strip()) < MIN_SENTENCE_LEN:
                        buf = sentence + buf
                        break
                    await self._tts.send_text(sentence)
            elif kind == "done":
                if buf.strip():
                    await self._tts.send_text(buf)
                return

    # -- barge-in -----------------------------------------------------------

    async def _barge_in(self) -> None:
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
            await tts.abort()
        self._transport.clear_output()
        if self._turn_task is not None and not self._turn_task.done():
            self._turn_task.cancel()
            try:
                await self._turn_task
            except (asyncio.CancelledError, Exception):
                pass
        self._turn_task = None
        self._state = LISTENING
        logger.info("voice/turn: barge-in handled")
