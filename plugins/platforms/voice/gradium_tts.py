"""Gradium streaming TTS websocket client — ONE socket PER AGENT TURN.

Per-turn sockets are the design (notes §12): turn audio rarely exceeds the
300s session cap, and the free tier's 1500-char/session TTS limit becomes a
per-turn budget. <flush> is used sparingly (filler utterances only) because
it costs prosody quality.

Output: 48 kHz s16le mono PCM in 3840-sample (80 ms = 7680 byte) chunks,
base64 in {"type":"audio","audio":...} messages.

Protocol facts verified against the LIVE websocket (2026-06-12):
  - output_format "pcm" is accepted (same wav|pcm|opus enum as the REST
    endpoint POST /api/post/speech/tts); audio chunks are raw s16le with
    no RIFF header.
  - First server message is {"type":"ready", "sample_rate":48000,
    "frame_size":3840, "request_id":..., ...}. Text may be sent
    immediately after setup without waiting for it.
  - Word timestamps arrive as {"type":"text","text":...,"start_s":...,
    "stop_s":...,"stream_id":0} messages interleaved with audio; audio
    messages carry start_s/stop_s/stream_id too.
  - Server closes the turn with {"type":"end_of_stream"} after all audio.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from typing import Awaitable, Callable, Optional

logger = logging.getLogger(__name__)

TTS_URL = "wss://api.gradium.ai/api/speech/tts"
OUTPUT_SAMPLE_RATE = 48000
DEFAULT_VOICE_ID = ""   # server default voice when unset


class GradiumTTSTurn:
    """One TTS socket for one agent turn.

    on_audio(pcm_bytes) is awaited for every decoded audio chunk — the
    caller wires it to the transport's output queue.
    """

    def __init__(
        self,
        api_key: str,
        voice_id: str,
        on_audio: Callable[[bytes], Awaitable[None]],
    ):
        if not api_key:
            raise ValueError("GRADIUM_API_KEY is required for the voice platform")
        self._api_key = api_key
        self._voice_id = voice_id or DEFAULT_VOICE_ID
        self._on_audio = on_audio
        self._ws = None
        self._recv_task: Optional[asyncio.Task] = None
        self._done = asyncio.Event()
        self._aborted = False
        self.chars_sent = 0

    async def open(self) -> None:
        import websockets
        self._ws = await websockets.connect(
            TTS_URL, additional_headers={"x-api-key": self._api_key}
        )
        setup = {"type": "setup", "model_name": "default", "output_format": "pcm"}
        if self._voice_id:
            setup["voice_id"] = self._voice_id
        await self._ws.send(json.dumps(setup))
        self._recv_task = asyncio.create_task(self._recv_loop())

    async def _recv_loop(self) -> None:
        try:
            async for raw in self._ws:
                msg = json.loads(raw)
                mtype = msg.get("type")
                if mtype == "audio":
                    await self._on_audio(base64.b64decode(msg["audio"]))
                elif mtype == "error":
                    logger.warning("voice/tts: server error: %s", msg.get("message"))
                elif mtype == "end_of_stream":
                    break
        except asyncio.CancelledError:
            raise
        except Exception as e:
            if not self._aborted:
                logger.warning("voice/tts: socket error: %s", e)
        finally:
            self._done.set()

    async def send_text(self, text: str) -> None:
        """Send one sentence/fragment. Never split a word across calls —
        Gradium auto-inserts whitespace BETWEEN messages."""
        if self._aborted or not text:
            return
        self.chars_sent += len(text)
        await self._ws.send(json.dumps({"type": "text", "text": text}))

    async def send_filler(self, text: str) -> None:
        """Filler utterance with forced flush (tool-latency masking)."""
        await self.send_text(text + " <flush>")

    async def end(self) -> None:
        """Signal end of turn text; wait for all audio to be delivered.

        NOTE: this blocks until the server sends its final audio and
        end_of_stream. Callers (the turn loop) should wrap this in
        ``asyncio.wait_for`` so a stalled server can't hang the turn.
        """
        if self._aborted:
            return
        try:
            await self._ws.send(json.dumps({"type": "end_of_stream"}))
        except Exception:
            pass
        await self._done.wait()
        await self._cleanup()

    async def abort(self) -> None:
        """Barge-in: hard-close the socket, drop pending audio server-side."""
        self._aborted = True
        await self._cleanup()
        self._done.set()

    async def _cleanup(self) -> None:
        try:
            if self._ws is not None:
                await self._ws.close()
        except Exception:
            pass
        if self._recv_task is not None and not self._recv_task.done():
            self._recv_task.cancel()
            try:
                await self._recv_task
            except (asyncio.CancelledError, Exception):
                pass
