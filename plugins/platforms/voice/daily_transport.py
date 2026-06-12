"""Daily WebRTC transport via daily-python virtual audio devices.

Audio geometry (chosen to avoid resampling entirely — daily-python gives
every virtual device its own sample_rate, verified against the installed
SDK 0.29.1: Daily.create_microphone_device / create_speaker_device both
take ``sample_rate``):
  inbound  (caller -> STT): virtual SPEAKER device @ 24 kHz mono — matches
           Gradium ASR "pcm" input. Read 1920 frames (80 ms) per blocking
           read_frames() call; the device paces reads at real time.
  outbound (TTS -> caller): virtual MICROPHONE device @ 48 kHz mono —
           matches Gradium TTS output (3840-sample/80 ms chunks pass
           through untouched). Blocking write_frames() paces playback;
           barge-in drains the local queue and stops writing.

daily-python allows ONE active virtual speaker per process (and the
upstream demos treat the mic the same way), so devices are process-level
singletons and the adapter enforces a single active call.

API facts verified against the installed daily-python 0.29.1 and the
upstream demos (demos/audio/wav_audio_send.py, wav_audio_receive.py):
  - Daily.init(worker_threads=2, log_level=...) — no args required.
  - create_microphone_device(device_name, sample_rate=16000, channels=1,
    non_blocking=False); same signature for create_speaker_device.
  - CallClient(event_handler=None);
    join(meeting_url, meeting_token=None, client_settings=None,
         completion=None) — completion(JoinData, CallClientError);
    leave(completion=None) — completion(CallClientError);
    update_subscription_profiles(profile_settings, completion=None).
  - client_settings mic selection shape (wav_audio_send.py):
    {"inputs": {"camera": False, "microphone":
        {"isEnabled": True, "settings": {"deviceId": <name>}}}}
  - VirtualMicrophoneDevice.write_frames(frames, completion=None) -> int;
    VirtualSpeakerDevice.read_frames(num_frames, completion=None) ->
    bytestring (empty when no frames were read).
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
import time
from typing import Awaitable, Callable, Optional

logger = logging.getLogger(__name__)

MIC_DEVICE = "hermes-voice-mic"
SPEAKER_DEVICE = "hermes-voice-speaker"
MIC_RATE = 48000                # Gradium TTS output rate (passthrough)
SPEAKER_RATE = 24000            # Gradium ASR "pcm" input rate
IN_CHUNK_FRAMES = 1920          # 80 ms @ 24 kHz — one ASR chunk per read

# Subscribe to participant microphones only (wav_audio_receive.py pattern);
# video is never wanted on a voice call.
_SUBSCRIPTION_PROFILES = {
    "base": {"camera": "unsubscribed", "microphone": "subscribed"}
}

_init_lock = threading.Lock()
_initialized = False
_mic = None
_speaker = None


def _ensure_daily() -> None:
    """Daily.init() + virtual device creation, once per process."""
    global _initialized, _mic, _speaker
    with _init_lock:
        if _initialized:
            return
        from daily import Daily
        Daily.init()
        _mic = Daily.create_microphone_device(
            MIC_DEVICE, sample_rate=MIC_RATE, channels=1)
        _speaker = Daily.create_speaker_device(
            SPEAKER_DEVICE, sample_rate=SPEAKER_RATE, channels=1)
        Daily.select_speaker_device(SPEAKER_DEVICE)
        _initialized = True


class DailyTransport:
    """One Daily call. on_audio_in(pcm) is scheduled onto *loop* for every
    80 ms chunk of caller audio (s16le mono 24 kHz)."""

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        on_audio_in: Callable[[bytes], Awaitable[None]],
    ):
        self._loop = loop
        self._on_audio_in = on_audio_in
        self._client = None
        self._running = False
        self._reader: Optional[threading.Thread] = None
        self._writer: Optional[threading.Thread] = None
        self._keepalive: Optional[threading.Thread] = None
        self._out_q: "queue.Queue[Optional[bytes]]" = queue.Queue()
        self._joined = threading.Event()
        self._join_error: Optional[str] = None
        self._last_audio_in_t = 0.0

    async def join(self, room_url: str, token: str, timeout: float = 15.0) -> None:
        _ensure_daily()
        from daily import CallClient
        self._client = CallClient()
        self._client.update_subscription_profiles(_SUBSCRIPTION_PROFILES)

        def _on_join(data, error):
            self._join_error = str(error) if error else None
            self._joined.set()

        self._client.join(
            room_url,
            meeting_token=token,
            client_settings={
                "inputs": {
                    "camera": False,
                    "microphone": {
                        "isEnabled": True,
                        "settings": {"deviceId": MIC_DEVICE},
                    },
                }
            },
            completion=_on_join,
        )
        await self._loop.run_in_executor(None, self._joined.wait, timeout)
        if not self._joined.is_set():
            self._client.release()
            self._client = None
            raise RuntimeError(f"Daily join timed out after {timeout}s")
        if self._join_error:
            self._client.release()
            self._client = None
            raise RuntimeError(f"Daily join failed: {self._join_error}")
        self._running = True
        self._last_audio_in_t = time.monotonic()
        self._reader = threading.Thread(
            target=self._read_loop, name="voice-daily-reader", daemon=True)
        self._writer = threading.Thread(
            target=self._write_loop, name="voice-daily-writer", daemon=True)
        self._keepalive = threading.Thread(
            target=self._keepalive_loop, name="voice-daily-keepalive",
            daemon=True)
        self._reader.start()
        self._writer.start()
        self._keepalive.start()
        logger.info("voice/daily: joined %s", room_url)

    def _read_loop(self) -> None:
        while self._running:
            frames = _speaker.read_frames(IN_CHUNK_FRAMES)   # blocking 80 ms
            if not frames:
                # Empty reads happen at teardown / before audio flows; avoid
                # a hot spin since only non-empty reads pace real time.
                time.sleep(0.01)
                continue
            self._last_audio_in_t = time.monotonic()
            asyncio.run_coroutine_threadsafe(self._on_audio_in(frames), self._loop)

    def _keepalive_loop(self) -> None:
        # WebRTC DTX: a silent caller stops sending packets entirely and
        # read_frames() BLOCKS (it cannot be relied on to return empties at
        # cadence). Gradium's streaming ASR/VAD assumes a continuous
        # timeline — starving it produces wildly oscillating VAD
        # probabilities (observed live 2026-06-12: false barge-ins,
        # end-of-turn never firing, sparse step events). This thread feeds
        # synthesized 80ms silence whenever real audio stops flowing.
        silence = b"\x00" * (IN_CHUNK_FRAMES * 2)
        while self._running:
            time.sleep(0.08)
            if not self._running:
                return
            if time.monotonic() - self._last_audio_in_t >= 0.16:
                asyncio.run_coroutine_threadsafe(
                    self._on_audio_in(silence), self._loop)

    def _write_loop(self) -> None:
        # Pacing diagnosis: track audio-seconds written vs wall-clock since
        # the burst started. If write_frames does NOT pace at real time,
        # audio_s will outrun wall_s and barge-in cannot stop buffered audio.
        burst_t0 = 0.0
        burst_audio_s = 0.0
        fast_count = 0
        while self._running:
            try:
                chunk = self._out_q.get(timeout=0.5)
            except queue.Empty:
                if burst_audio_s > 0.0:
                    logger.info(
                        "voice/daily: write burst ended audio_s=%.2f wall_s=%.2f",
                        burst_audio_s, time.monotonic() - burst_t0)
                    burst_audio_s = 0.0
                continue
            if chunk is None:
                continue
            if burst_audio_s == 0.0:
                burst_t0 = time.monotonic()
                logger.info("voice/daily: write burst started qsize=%d",
                            self._out_q.qsize())
            t0 = time.monotonic()
            _mic.write_frames(chunk)                          # blocking?
            dt = time.monotonic() - t0
            chunk_s = len(chunk) / 2.0 / MIC_RATE
            burst_audio_s += chunk_s
            if dt < chunk_s * 0.5:
                # Write returned faster than real time — Daily is buffering.
                fast_count += 1
                if fast_count % 12 == 1:
                    logger.info(
                        "voice/daily: FAST write chunk_s=%.3f took=%.3f "
                        "burst_audio_s=%.2f wall_s=%.2f qsize=%d",
                        chunk_s, dt, burst_audio_s,
                        time.monotonic() - burst_t0, self._out_q.qsize())

    async def send_audio(self, pcm: bytes) -> None:
        """Queue agent speech (s16le mono 48 kHz) for the caller."""
        self._out_q.put(pcm)

    def clear_output(self) -> None:
        """Barge-in: drop all queued (unplayed) agent audio."""
        dropped = 0
        try:
            while True:
                self._out_q.get_nowait()
                dropped += 1
        except queue.Empty:
            pass
        logger.info("voice/daily: cleared %d queued chunks", dropped)

    async def leave(self) -> None:
        self._running = False
        self.clear_output()
        if self._client is not None:
            done = threading.Event()
            self._client.leave(completion=lambda _error: done.set())
            await self._loop.run_in_executor(None, done.wait, 10.0)
            self._client.release()
            self._client = None
        for worker in (self._reader, self._writer, self._keepalive):
            if worker is not None and worker.is_alive():
                await self._loop.run_in_executor(None, worker.join, 2.0)
        self._reader = None
        self._writer = None
        self._keepalive = None
        logger.info("voice/daily: left room")
