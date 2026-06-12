"""Vamp clips — pre-synthesized acknowledgments for perceived latency.

The perceived-latency spec (demo notes §16) allows the FIRST audio the
caller hears to be a short canned acknowledgment ("Mm — let me think.")
fired on a fast end-of-turn signal, with the substantive reply streaming
in behind it. This module owns the clip cache:

  - At call start the adapter creates a VampCache and start()s it; clips
    are synthesized in a BACKGROUND task via the Gradium TTS REST endpoint
    (POST /api/post/speech/tts, live-verified 2026-06-12: output_format
    "pcm" returns raw 48 kHz s16le mono — the same geometry as the WS TTS
    stream and the Daily mic device, so clips pass straight through).
    Synthesis takes ~2.4 s per clip, which is exactly why generation must
    never gate the call join: the vamp simply stays disabled until ready.
  - pick() returns a random clip, never repeating the previous one.
  - Clips are silence-trimmed so the substantive reply queues tightly
    behind the spoken acknowledgment instead of behind dead air.

The REST endpoint requires an explicit voice_id (live-verified: 400
"either voice or voice_id must be specified" without one). A call with no
configured voice would vamp in a DIFFERENT voice than the agent's WS
default — worse than no vamp — so the cache refuses to start without one
and logs why.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

TTS_REST_URL = "https://api.gradium.ai/api/post/speech/tts"
SAMPLE_RATE = 48000
CHUNK_BYTES = 7680            # 80 ms @ 48 kHz s16le — matches the WS stream
REST_TIMEOUT_S = 30.0
_TRIM_WIN_BYTES = int(SAMPLE_RATE * 0.01) * 2     # 10 ms windows
_TRIM_RMS = 200

DEFAULT_VAMP_TEXTS = [
    "Mm — let me think.",
    "Right.",
    "Okay —",
    "Good question.",
    "One sec.",
]


def _rms(buf: bytes) -> float:
    samples = memoryview(buf).cast("h")
    if len(samples) == 0:
        return 0.0
    return (sum(s * s for s in samples) / len(samples)) ** 0.5


def trim_silence(pcm: bytes) -> bytes:
    """Strip leading/trailing near-silence (TTS clips carry ~0.5 s of
    trailing silence that would delay the substantive reply queued
    behind the vamp)."""
    start, end = 0, len(pcm) - len(pcm) % 2
    while start + _TRIM_WIN_BYTES < end and \
            _rms(pcm[start:start + _TRIM_WIN_BYTES]) <= _TRIM_RMS:
        start += _TRIM_WIN_BYTES
    while end - _TRIM_WIN_BYTES > start and \
            _rms(pcm[end - _TRIM_WIN_BYTES:end]) <= _TRIM_RMS:
        end -= _TRIM_WIN_BYTES
    return pcm[start:end]


class VampCache:
    """In-memory PCM clips, synthesized in the background at call start."""

    def __init__(self, api_key: str, voice_id: str,
                 texts: Optional[List[str]] = None):
        if not api_key:
            raise ValueError("GRADIUM_API_KEY is required for vamp clips")
        self._api_key = api_key
        self._voice_id = voice_id
        self._texts = [t for t in (texts or DEFAULT_VAMP_TEXTS) if t and t.strip()]
        self._clips: List[Tuple[str, bytes]] = []
        self._last_idx: Optional[int] = None
        self._task: Optional[asyncio.Task] = None

    @property
    def ready(self) -> bool:
        return bool(self._clips)

    def start(self) -> Optional[asyncio.Task]:
        """Kick off background synthesis. Never blocks the caller; the
        vamp stays disabled (ready=False) until clips exist."""
        if not self._voice_id:
            logger.warning(
                "voice/vamp: disabled — no voice_id configured (the REST "
                "endpoint requires one, and vamping in a different voice "
                "than the agent's would be worse than no vamp)")
            return None
        if not self._texts:
            logger.warning("voice/vamp: disabled — no vamp texts configured")
            return None
        self._task = asyncio.create_task(self._synthesize_all())
        return self._task

    async def stop(self) -> None:
        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
        self._task = None

    async def _synthesize_all(self) -> None:
        # Sequential on purpose: Gradium's free tier caps concurrent
        # sessions at 3, and the live call already holds STT + TTS
        # sockets — parallel REST synthesis could starve the call itself.
        # The vamp simply becomes usable clip-by-clip as they land.
        t0 = time.monotonic()
        for text in self._texts:
            try:
                pcm = await self._synthesize_clip(text)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning("voice/vamp: clip synthesis failed for %r: %s",
                               text, e)
                continue
            if pcm:
                self._clips.append((text, pcm))
        if self._clips:
            logger.info(
                "voice/vamp: %d/%d clips ready in %.1fs (total %.1fs audio)",
                len(self._clips), len(self._texts), time.monotonic() - t0,
                sum(len(p) for _, p in self._clips) / 2 / SAMPLE_RATE)
        else:
            logger.warning("voice/vamp: NO clips synthesized — vamp disabled "
                           "for this call")

    async def _synthesize_clip(self, text: str) -> bytes:
        import httpx

        async with httpx.AsyncClient(timeout=REST_TIMEOUT_S) as client:
            resp = await client.post(
                TTS_REST_URL,
                headers={"x-api-key": self._api_key},
                json={"text": text, "voice_id": self._voice_id,
                      "output_format": "pcm", "only_audio": True},
            )
            resp.raise_for_status()
        return trim_silence(resp.content)

    def pick(self) -> Optional[Tuple[str, bytes]]:
        """Random clip, never the same one twice in a row. None until
        synthesis has produced at least one clip."""
        if not self._clips:
            return None
        candidates = list(range(len(self._clips)))
        if self._last_idx is not None and len(candidates) > 1:
            candidates.remove(self._last_idx)
        idx = random.choice(candidates)
        self._last_idx = idx
        return self._clips[idx]
