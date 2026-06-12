#!/usr/bin/env python
"""Synthetic voice caller — latency + barge-in measurement harness.

Joins a Daily room as a fake human (daily-python virtual devices), plays a
pre-generated utterance WAV into the room, and measures with monotonic
timestamps:

  latency mode : T_user_speech_end -> T_first_agent_audio   (voice-to-voice)
  bargein mode : plays a second utterance N ms into the agent's reply and
                 measures T_bargein_speech_start -> T_last_agent_audio
                 (how long the agent keeps talking after being interrupted)
  echo mode    : stays silent forever; use the agent's logs to confirm the
                 agent's own speech never triggers a barge-in.

Target agent setup (standalone mode is the simplest reliable path — no
control-plane dependency): run the agent with `platforms.voice.extra.mode:
standalone` and DAILY_API_KEY in its env. The adapter creates its own
private room and logs "standalone room ready — share this URL: <url>".
Pass that URL via --room-url (or --container to scrape `docker logs`).
The probe mints its own meeting token with DAILY_API_KEY (env).

Utterance WAVs are 48 kHz s16le mono (generate via the Gradium TTS REST
endpoint). The probe keeps the virtual mic fed continuously — silence
chunks between utterances — because the agent's STT only emits VAD steps
for audio it receives.

Usage:
  python scripts/voice_probe.py --container sb-agent-probe --mode latency --runs 3
  python scripts/voice_probe.py --room-url https://x.daily.co/y --mode bargein
"""

from __future__ import annotations

import argparse
import audioop
import os
import queue
import re
import subprocess
import sys
import threading
import time
import wave

import httpx

SAMPLE_RATE = 48000
CHUNK_S = 0.08
CHUNK_FRAMES = int(SAMPLE_RATE * CHUNK_S)          # 3840
CHUNK_BYTES = CHUNK_FRAMES * 2
# Comfort noise, not digital zero: Gradium's VAD degrades on long pure-zero
# stretches (observed live 2026-06-12 — after ~2 turns the end-of-turn
# probability never recovered until session rotation; real mics always carry
# room noise). RMS ~60: far below the agent's energy gates (500) and the
# probe's own speech detector (300).
import random as _random

_random.seed(7)
SILENCE_FRAMES = [
    bytes(b for _ in range(CHUNK_FRAMES)
          for b in int.to_bytes(_random.randint(-90, 90) & 0xFFFF, 2, "little"))
    for _ in range(8)
]
RMS_SPEECH = 300            # s16 RMS above this counts as agent speech
# Quiet gap that ends the agent's reply. Must exceed the worst vamp ->
# substantive gap (~3.1s measured) or the probe mistakes the vamp clip for
# the whole reply and the NEXT run collides with the late substantive audio.
REPLY_END_SILENCE_S = 4.0

# Verify run 2026-06-12: a vamp ack clip followed by the (3.9-4.3s)
# vamp->substantive gap was mistaken for a completed reply, colliding the
# probe's next utterance with the late substantive audio. The reply-end wait
# is now a classifier: short (vamp-length) audio followed by silence stays
# "vamp-gap" (still waiting) until substantive audio arrives or a hard
# timeout passes.
VAMP_MAX_S = 2.0            # speech runs at/below this are vamp-length
VAMP_GAP_TIMEOUT_S = 6.0    # vamp with nothing behind it = the whole reply
RUN_BREAK_S = 0.25          # silence gap that splits speech runs


def _speech_runs(samples, t_start):
    """Contiguous speech runs [(t_first, t_last), ...] at/after t_start."""
    speech = sorted(t for t, rms in samples if t >= t_start and rms > RMS_SPEECH)
    runs = []
    for t in speech:
        if runs and t - runs[-1][1] <= RUN_BREAK_S:
            runs[-1][1] = t
        else:
            runs.append([t, t])
    return [(a, b) for a, b in runs]


def reply_end_state(samples, t_start, now):
    """Classify the reply window: ("waiting"|"vamp-gap"|"done", t_last_speech).

    - no speech yet            -> ("waiting", None)
    - substantive speech seen  -> "done" after REPLY_END_SILENCE_S of quiet
    - only vamp-length speech  -> "vamp-gap" once quiet exceeds
      REPLY_END_SILENCE_S (keep waiting for the real reply), "done" only
      after VAMP_GAP_TIMEOUT_S (the ack WAS the whole reply).
    """
    runs = _speech_runs(samples, t_start)
    if not runs:
        return ("waiting", None)
    last = runs[-1][1]
    silence = now - last
    substantive = any(b - a + CHUNK_S > VAMP_MAX_S for a, b in runs)
    if substantive:
        return ("done", last) if silence >= REPLY_END_SILENCE_S else ("waiting", last)
    if silence >= VAMP_GAP_TIMEOUT_S:
        return ("done", last)
    if silence >= REPLY_END_SILENCE_S:
        return ("vamp-gap", last)
    return ("waiting", last)


def first_substantive_start(samples, t_start):
    """Start time of the first speech run longer than vamp-length, or None."""
    for a, b in _speech_runs(samples, t_start):
        if b - a + CHUNK_S > VAMP_MAX_S:
            return a
    return None

DAILY_API = "https://api.daily.co/v1"

MIC = "probe-mic"
SPK = "probe-speaker"


def load_wav(path: str) -> bytes:
    """Load 48kHz mono s16le, trimming leading/trailing silence so
    T_user_speech_end (= last queued chunk) matches the real speech end.
    TTS-generated WAVs carry ~0.5s of trailing silence that would
    otherwise flatter every latency number."""
    with wave.open(path, "rb") as w:
        assert w.getframerate() == SAMPLE_RATE, f"{path}: need 48kHz"
        assert w.getnchannels() == 1 and w.getsampwidth() == 2
        pcm = w.readframes(w.getnframes())
    win = int(SAMPLE_RATE * 0.01) * 2          # 10ms windows
    start, end = 0, len(pcm)
    while start + win < len(pcm) and audioop.rms(pcm[start:start+win], 2) <= 200:
        start += win
    while end - win > start and audioop.rms(pcm[end-win:end], 2) <= 200:
        end -= win
    return pcm[start:end]


def room_url_from_container(container: str) -> str:
    out = subprocess.run(
        ["docker", "logs", container], capture_output=True, text=True
    )
    text = out.stdout + out.stderr
    # s6 images log the gateway to a file, not the container stdout
    out2 = subprocess.run(
        ["docker", "exec", container, "cat", "/opt/data/logs/gateway.log"],
        capture_output=True, text=True,
    )
    text += out2.stdout
    urls = re.findall(r"standalone room ready.*?(https://\S+)", text)
    if not urls:
        raise SystemExit(f"no 'standalone room ready' URL in {container} logs")
    return urls[-1]


def mint_token(room_url: str) -> str:
    api_key = os.environ["DAILY_API_KEY"]
    room_name = room_url.rstrip("/").rsplit("/", 1)[-1]
    r = httpx.post(
        f"{DAILY_API}/meeting-tokens",
        headers={"Authorization": f"Bearer {api_key}"},
        json={"properties": {"room_name": room_name, "is_owner": False,
                             "exp": int(time.time()) + 3600}},
        timeout=15,
    )
    r.raise_for_status()
    return r.json()["token"]


class Probe:
    """Fake human in the room: paced mic writer + timestamped speaker reader."""

    def __init__(self) -> None:
        from daily import Daily
        Daily.init()
        self._mic = Daily.create_microphone_device(
            MIC, sample_rate=SAMPLE_RATE, channels=1)
        self._spk = Daily.create_speaker_device(
            SPK, sample_rate=SAMPLE_RATE, channels=1)
        Daily.select_speaker_device(SPK)
        self._client = None
        self._running = False
        self._speech_q: "queue.Queue[bytes]" = queue.Queue()
        self.last_speech_write: float = 0.0      # monotonic, end of last speech chunk
        self.rms_log: list[tuple[float, int]] = []   # (monotonic, rms) per 80ms
        self._rms_lock = threading.Lock()

    def join(self, room_url: str, token: str) -> None:
        from daily import CallClient
        self._client = CallClient()
        self._client.update_subscription_profiles(
            {"base": {"camera": "unsubscribed", "microphone": "subscribed"}})
        joined = threading.Event()
        err: list = [None]

        def _done(data, error):
            err[0] = error
            joined.set()

        self._client.join(
            room_url, meeting_token=token,
            client_settings={"inputs": {
                "camera": False,
                "microphone": {"isEnabled": True,
                               "settings": {"deviceId": MIC}}}},
            completion=_done)
        if not joined.wait(15) or err[0]:
            raise SystemExit(f"join failed: {err[0]}")
        self._running = True
        threading.Thread(target=self._mic_loop, daemon=True).start()
        threading.Thread(target=self._spk_loop, daemon=True).start()

    def _mic_loop(self) -> None:
        """Continuously feed the mic: speech chunks when queued, else silence.
        Real-time paced by wall clock (blocking write_frames may not pace)."""
        next_t = time.monotonic()
        while self._running:
            try:
                chunk = self._speech_q.get_nowait()
                is_speech = True
            except queue.Empty:
                chunk, is_speech = _random.choice(SILENCE_FRAMES), False
            self._mic.write_frames(chunk)
            if is_speech:
                self.last_speech_write = time.monotonic()
            next_t += CHUNK_S
            delay = next_t - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_t = time.monotonic()

    def _spk_loop(self) -> None:
        while self._running:
            frames = self._spk.read_frames(CHUNK_FRAMES)
            t = time.monotonic()
            if not frames:
                time.sleep(0.005)
                continue
            rms = audioop.rms(frames, 2)
            with self._rms_lock:
                self.rms_log.append((t, rms))

    def say(self, pcm: bytes) -> float:
        """Queue an utterance; returns the wall (monotonic) time the LAST
        speech chunk gets written (i.e. T_user_speech_end), by waiting."""
        n = 0
        for off in range(0, len(pcm), CHUNK_BYTES):
            chunk = pcm[off:off + CHUNK_BYTES]
            if len(chunk) < CHUNK_BYTES:
                chunk = chunk + b"\x00" * (CHUNK_BYTES - len(chunk))
            self._speech_q.put(chunk)
            n += 1
        while not self._speech_q.empty():
            time.sleep(0.01)
        time.sleep(CHUNK_S)         # last chunk is mid-write when q drains
        return self.last_speech_write

    # -- agent-audio observations --------------------------------------------

    def _snapshot(self) -> list[tuple[float, int]]:
        with self._rms_lock:
            return list(self.rms_log)

    def first_audio_after(self, t0: float, timeout: float = 20.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            for t, rms in self._snapshot():
                if t > t0 and rms > RMS_SPEECH:
                    return t
            time.sleep(0.02)
        return None

    def wait_reply_end(self, t_start: float, timeout: float = 60.0):
        """Return the time of the last speech-level chunk once the reply-end
        classifier says the reply is done (vamp-aware; see reply_end_state)."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            state, t_last = reply_end_state(
                self._snapshot(), t_start, time.monotonic())
            if state == "done":
                return t_last
            time.sleep(0.05)
        return None

    def leave(self) -> None:
        self._running = False
        if self._client is not None:
            done = threading.Event()
            self._client.leave(completion=lambda e: done.set())
            done.wait(10)
            self._client.release()


def run_latency(probe: Probe, utt: bytes, runs: int) -> list[float]:
    results = []
    for i in range(runs):
        base = time.monotonic()
        t_end = probe.say(utt)
        t_first = probe.first_audio_after(t_end)
        if t_first is None:
            print(f"run {i+1}: NO AGENT AUDIO within timeout")
            results.append(float("nan"))
            continue
        v2v = t_first - t_end
        print(f"run {i+1}: voice-to-voice = {v2v*1000:.0f} ms")
        results.append(v2v)
        t_last = probe.wait_reply_end(t_first)
        if t_last is None:
            print(f"run {i+1}: reply did not end within timeout")
        time.sleep(1.0)
        del base
    return results


def run_bargein(probe: Probe, utt: bytes, utt2: bytes, runs: int,
                delay_into_reply: float = 1.0) -> list[float]:
    results = []
    for i in range(runs):
        t_end = probe.say(utt)
        t_first = probe.first_audio_after(t_end)
        if t_first is None:
            print(f"run {i+1}: NO AGENT AUDIO; cannot barge in")
            results.append(float("nan"))
            continue
        # wait until N s into the agent's reply, then talk over it
        wait = t_first + delay_into_reply - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        t_barge = time.monotonic()
        print(f"run {i+1}: barge speech start wall={time.time():.3f}")
        probe.say(utt2)
        # The cutoff is when the OLD reply's audio stops: the first >=1.0s
        # gap in agent speech after t_barge. (wait_reply_end's quiet window
        # is wider than the cutoff->new-reply gap — the vamp + substantive
        # reply to the barge utterance start ~2-3s later — so it would
        # measure the END of the NEW reply instead.)
        time.sleep(12.0)                  # observe well past the new reply start
        speech = [t for t, rms in probe._snapshot()
                  if t >= t_barge - 0.5 and rms > RMS_SPEECH]
        t_stop = None
        prev = t_barge
        for t in speech:
            if t - prev >= 2.0:
                t_stop = prev
                break
            prev = t
        if t_stop is None and speech and time.monotonic() - speech[-1] >= 2.0:
            t_stop = speech[-1]
        if t_stop is None:
            print(f"run {i+1}: no audio gap found after barge-in")
            results.append(float("nan"))
        elif t_stop < t_barge:
            print(f"run {i+1}: agent already quiet at barge-in "
                  f"(stopped {1000*(t_barge-t_stop):.0f} ms before)")
            results.append(0.0)
        else:
            cut = t_stop - t_barge
            print(f"run {i+1}: barge-in cutoff = {cut*1000:.0f} ms")
            results.append(cut)
        # let the new turn's reply play out before the next run
        t_reply = probe.first_audio_after(t_barge + 0.5, timeout=20.0)
        if t_reply:
            probe.wait_reply_end(t_reply)
        time.sleep(1.0)
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--room-url")
    ap.add_argument("--container", help="scrape standalone room URL from docker logs")
    ap.add_argument("--mode", choices=["latency", "bargein", "echo"],
                    default="latency")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--utt", default="/tmp/probe_utt1.wav")
    ap.add_argument("--utt2", default="/tmp/probe_utt2.wav")
    ap.add_argument("--barge-delay", type=float, default=1.0,
                    help="seconds into the agent reply to start talking over it")
    ap.add_argument("--settle", type=float, default=14.0,
                    help="seconds to wait after join (greeting playout)")
    args = ap.parse_args()

    room_url = args.room_url or (
        room_url_from_container(args.container) if args.container else None)
    if not room_url:
        raise SystemExit("--room-url or --container required")
    print(f"room: {room_url}")
    token = mint_token(room_url)

    utt = load_wav(args.utt)
    utt2 = load_wav(args.utt2) if os.path.exists(args.utt2) else utt

    probe = Probe()
    probe.join(room_url, token)
    print(f"joined; settling {args.settle}s (greeting playout)...")
    time.sleep(args.settle)

    try:
        if args.mode == "latency":
            res = run_latency(probe, utt, args.runs)
        elif args.mode == "bargein":
            res = run_bargein(probe, utt, utt2, args.runs, args.barge_delay)
        else:
            print("echo mode: staying silent 60s; check agent logs for "
                  "spurious barge-in triggers")
            time.sleep(60)
            res = []
        ok = sorted(x for x in res if x == x)
        if ok:
            med = ok[len(ok) // 2]
            print(f"median over {len(ok)} runs: {med*1000:.0f} ms")
    finally:
        probe.leave()


if __name__ == "__main__":
    main()
