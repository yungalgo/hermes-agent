"""Gradium streaming ASR websocket client with semantic VAD + session rotation.

Protocol (docs.gradium.ai/guides/speech-to-text; verified against the LIVE
websocket 2026-06-12):
  send: {"type":"setup","model_name":"default","input_format":"pcm"}
        {"type":"audio","audio":"<b64 s16le mono 24kHz>"}  (80ms = 1920 samples)
        {"type":"flush","flush_id":N} / {"type":"end_of_stream"}
  recv: {"type":"ready","sample_rate":24000,"frame_size":1920,
         "delay_in_frames":10,...}                          (first message)
        {"type":"text","text":...,"start_s":...}
        {"type":"step","vad":[{"horizon_s":h,"inactivity_prob":p},...],
         "step_idx":N,"step_duration_s":0.08,
         "total_duration_s":...}                            (every 80ms)
        {"type":"end_text"} {"type":"flushed","flush_id":N} {"type":"end_of_stream"}
  NOTE (live delta vs docs): vad carries FOUR horizons — 0.5/1.0/2.0/3.0 —
  not three; consumers must look horizons up by value, never by index.

Gradium model-API sessions are capped at 300s (notes §12): GradiumSTT rotates
the underlying socket proactively at ROTATE_AFTER_S during caller silence,
overlapping old and new sockets so no audio is dropped.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from typing import Any, AsyncIterator, Dict, Optional

logger = logging.getLogger(__name__)

ASR_URL = "wss://api.gradium.ai/api/speech/asr"
SAMPLE_RATE = 24000          # "pcm" input_format default
CHUNK_SAMPLES = 1920         # 80 ms
ROTATE_AFTER_S = 240.0       # rotate well before the 300s session cap
SILENCE_PROB_FOR_ROTATE = 0.8  # horizon-0.5 inactivity required to rotate
# The server kills sessions at 300s of WALL CLOCK, not processed audio.
# maybe_rotate only runs on step events (which require audio inflow), so a
# call with no inbound audio (empty room, muted caller) would die silently
# at 300s (observed live 2026-06-12: 1008 "Session exceeded maximum
# duration"). The watchdog force-rotates on wall-clock age and reconnects
# a session whose socket died.
HARD_ROTATE_AFTER_S = 270.0
WATCHDOG_INTERVAL_S = 5.0
# DEAF session: socket alive, audio being sent, but NO step events coming
# back. Observed live 2026-06-12 (twice): ~90-120s into a session the step
# stream just stops while the websocket stays open — end-of-turn detection
# goes dark for minutes until the wall-clock rotation finally replaces the
# session. The server emits a step every 80ms of audio, so multiple seconds
# of step silence while we are actively sending is unambiguous.
DEAF_AFTER_S = 10.0
AUDIO_RECENT_S = 2.0


class _Session:
    """One ASR websocket. Created by GradiumSTT; not used directly.

    ``next_flush_id`` is supplied by the parent GradiumSTT so flush ids are
    monotonic across the WHOLE call, not per socket: rotated sessions share
    one events queue, and a per-socket counter would let a stale ack from
    the old socket collide with (and falsely satisfy) a new flush wait.
    """

    def __init__(
        self,
        api_key: str,
        out: "asyncio.Queue[Dict[str, Any]]",
        next_flush_id,
    ):
        self._api_key = api_key
        self._out = out
        self._ws = None
        self._recv_task: Optional[asyncio.Task] = None
        self._next_flush_id = next_flush_id
        self.total_duration_s = 0.0
        self.closed = False
        self.opened_at = 0.0
        self.last_step_t = 0.0
        self.last_audio_sent_t = 0.0

    @property
    def dead(self) -> bool:
        """Socket gone without close(): the recv loop has exited."""
        return (not self.closed and self._recv_task is not None
                and self._recv_task.done())

    async def open(self) -> None:
        import websockets
        self._ws = await websockets.connect(
            ASR_URL, additional_headers={"x-api-key": self._api_key}
        )
        await self._ws.send(json.dumps(
            {"type": "setup", "model_name": "default", "input_format": "pcm"}
        ))
        self.opened_at = time.monotonic()
        self.last_step_t = self.opened_at
        self._recv_task = asyncio.create_task(self._recv_loop())

    async def _recv_loop(self) -> None:
        try:
            async for raw in self._ws:
                msg = json.loads(raw)
                if msg.get("type") == "step":
                    self.total_duration_s = float(msg.get("total_duration_s") or 0.0)
                    self.last_step_t = time.monotonic()
                await self._out.put(msg)
                if msg.get("type") == "end_of_stream":
                    break
        except asyncio.CancelledError:
            raise
        except Exception as e:
            if not self.closed:
                logger.warning("voice/stt: socket error: %s", e)
                await self._out.put({"type": "error", "message": str(e)})

    async def send_audio(self, pcm: bytes) -> None:
        self.last_audio_sent_t = time.monotonic()
        await self._ws.send(json.dumps(
            {"type": "audio", "audio": base64.b64encode(pcm).decode("ascii")}
        ))

    async def flush(self) -> int:
        flush_id = self._next_flush_id()
        await self._ws.send(json.dumps({"type": "flush", "flush_id": flush_id}))
        return flush_id

    async def close(self) -> None:
        self.closed = True
        try:
            if self._ws is not None:
                await self._ws.send(json.dumps({"type": "end_of_stream"}))
                await self._ws.close()
        except Exception:
            pass
        if self._recv_task is not None:
            self._recv_task.cancel()
            try:
                await self._recv_task
            except (asyncio.CancelledError, Exception):
                pass


class GradiumSTT:
    """Rotating-session ASR stream. One instance per voice call.

    Usage:
        stt = GradiumSTT(api_key)
        await stt.start()
        ... await stt.send_audio(pcm) ...           # from the transport reader
        async for msg in stt.events(): ...          # text/step/flushed/...
        await stt.stop()
    """

    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("GRADIUM_API_KEY is required for the voice platform")
        self._api_key = api_key
        self._events: "asyncio.Queue[Dict[str, Any]]" = asyncio.Queue()
        self._session: Optional[_Session] = None
        self._rotating = False
        self._watchdog: Optional[asyncio.Task] = None
        # Call-scoped flush counter: monotonic across session rotations so a
        # flush id is never reused and stale acks can never match a new wait.
        self._flush_seq = 0

    def _next_flush_id(self) -> int:
        self._flush_seq += 1
        return self._flush_seq

    async def start(self) -> None:
        self._session = _Session(self._api_key, self._events, self._next_flush_id)
        await self._session.open()
        self._watchdog = asyncio.create_task(self._watch())

    async def _watch(self) -> None:
        """Wall-clock session keeper: reconnect dead sessions, force-rotate
        before the server's 300s wall-clock kill (which maybe_rotate cannot
        see when no audio is flowing)."""
        while True:
            await asyncio.sleep(WATCHDOG_INTERVAL_S)
            try:
                s = self._session
                if s is None or self._rotating:
                    continue
                now = time.monotonic()
                if s.dead:
                    logger.warning("voice/stt: session died; reconnecting")
                    await self._replace_session(s)
                elif now - s.opened_at > HARD_ROTATE_AFTER_S:
                    logger.info(
                        "voice/stt: wall-clock rotation at %.0fs session age",
                        now - s.opened_at)
                    await self._replace_session(s)
                elif (now - s.last_step_t > DEAF_AFTER_S
                        and now - s.last_audio_sent_t < AUDIO_RECENT_S):
                    logger.warning(
                        "voice/stt: DEAF session (no step for %.0fs while "
                        "sending audio); rotating", now - s.last_step_t)
                    await self._replace_session(s)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning("voice/stt: watchdog error: %s", e)

    async def _replace_session(self, old: "_Session") -> None:
        """Open a fresh session, then close the old one (overlap, no gap)."""
        if self._rotating:
            return
        self._rotating = True
        try:
            fresh = _Session(self._api_key, self._events, self._next_flush_id)
            await fresh.open()
            self._session = fresh
            await old.close()
        finally:
            self._rotating = False

    async def send_audio(self, pcm: bytes) -> None:
        if self._session is None or self._session.closed:
            return
        await self._session.send_audio(pcm)

    async def flush(self) -> Optional[int]:
        """Request a transcript flush. Returns the flush_id to await, or
        None when no session is live (no ``flushed`` event will ever
        arrive — callers pair flush waits with a timeout). Ids are
        monotonic across the whole call, including session rotations."""
        if self._session is None or self._session.closed:
            return None
        return await self._session.flush()

    async def maybe_rotate(self, latest_step: Dict[str, Any]) -> None:
        """Call on each step msg. Rotates during silence past ROTATE_AFTER_S."""
        if self._rotating or self._session is None:
            return
        if self._session.total_duration_s < ROTATE_AFTER_S:
            return
        probs = {v.get("horizon_s"): v.get("inactivity_prob", 0.0)
                 for v in latest_step.get("vad", [])}
        if probs.get(0.5, 0.0) < SILENCE_PROB_FOR_ROTATE:
            return
        old = self._session
        await self._replace_session(old)       # overlap: new socket live first
        logger.info("voice/stt: rotated session at %.0fs", old.total_duration_s)

    async def events(self) -> AsyncIterator[Dict[str, Any]]:
        while True:
            msg = await self._events.get()
            yield msg

    async def stop(self) -> None:
        if self._watchdog is not None:
            self._watchdog.cancel()
            try:
                await self._watchdog
            except (asyncio.CancelledError, Exception):
                pass
            self._watchdog = None
        if self._session is not None:
            await self._session.close()
