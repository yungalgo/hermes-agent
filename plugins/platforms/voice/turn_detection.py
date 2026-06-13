"""Local end-of-turn detection — Smart Turn v3 ONNX + RMS energy gate.

Replaces Gradium's semantic-VAD step probabilities for turn / barge-in
decisions (Gradium STT still transcribes; only its VAD-driven turn logic is
gone — see turn_loop.py). Two pieces live here:

  SmartTurnDetector
    Wraps the bundled pipecat-ai/smart-turn-v3 ONNX model (CPU, int8).
    ``predict_endpoint(audio_16k_f32) -> float`` returns the probability
    (already sigmoid-normalised by the graph) that the user's turn is
    complete. Model facts (proven in the spike, confirmed against
    pipecat-ai/smart-turn inference.py):
      - input: 16 kHz mono float32, an 8 s window (padded/truncated to
        128000 samples), run through a WhisperFeatureExtractor(chunk_length=8,
        do_normalize=True) into 80-mel log-spectrogram features;
      - ONNX input tensor name "input_features", shape (1, 80, 800);
      - output[0][0] is the endpoint PROBABILITY in [0, 1] (>0.5 = complete),
        NOT a raw logit — no extra sigmoid here;
      - ~13 ms CPU inference (int8, smart-turn-v3.2-cpu.onnx, 8.3 MB).
    All heavy deps (onnxruntime, transformers, numpy, scipy) import LAZILY
    inside load()/predict so the module stays importable — and the turn-loop
    logic stays unit-testable — without them. Tests inject a fake detector.

  RollingAudioBuffer
    Maintains the trailing 8 s of inbound caller audio, resampled
    24 kHz -> 16 kHz mono float32 (inbound is 24 kHz s16le 80 ms chunks; see
    daily_transport.py). Resampling is scipy.signal.resample_poly (no
    librosa). ``window_16k()`` returns the current float32 window for the
    detector.

  rms_energy()
    Cheap local RMS of an s16le 80 ms chunk — the silence/speech gate that
    decides WHEN to run the (relatively heavier) Smart Turn model. This
    replaces all reliance on Gradium step VAD for turn/barge timing.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Optional

logger = logging.getLogger(__name__)

# Smart Turn model (baked into the image; see Dockerfile). The repo id +
# filename are overridable via env only for an emergency model swap — the
# image bake is the supported path and needs no network at runtime.
SMART_TURN_REPO = os.getenv("VOICE_SMART_TURN_REPO", "pipecat-ai/smart-turn-v3")
SMART_TURN_FILE = os.getenv("VOICE_SMART_TURN_FILE", "smart-turn-v3.2-cpu.onnx")
# Where the Dockerfile bakes the ONNX (vendored COPY target). Checked before
# any HF download so the runtime never reaches the network.
BAKED_MODEL_PATH = os.getenv(
    "VOICE_SMART_TURN_PATH",
    "/opt/hermes/plugins/platforms/voice/models/smart-turn-v3.2-cpu.onnx",
)

SAMPLE_RATE_IN = 24000       # inbound caller PCM (Gradium ASR "pcm" rate)
SAMPLE_RATE_MODEL = 16000    # Smart Turn input rate
WINDOW_SECONDS = 8
WINDOW_SAMPLES = SAMPLE_RATE_MODEL * WINDOW_SECONDS   # 128000
# 24k -> 16k is a 2/3 ratio (up=2, down=3) for resample_poly.
_RESAMPLE_UP = 2
_RESAMPLE_DOWN = 3


def rms_energy(pcm: bytes) -> float:
    """RMS of s16le mono PCM (audioop-free; removed in Python 3.13).
    Subsamples every 4th frame — plenty for an 80 ms energy gate. Mirrors
    turn_loop._rms so the energy gate matches the legacy barge trigger."""
    samples = memoryview(pcm).cast("h")[::4]
    if len(samples) == 0:
        return 0.0
    return (sum(s * s for s in samples) / len(samples)) ** 0.5


class RollingAudioBuffer:
    """Trailing WINDOW_SECONDS of inbound audio at 16 kHz mono float32.

    Inbound chunks are 24 kHz s16le; each is converted to float32 [-1, 1],
    resampled 24k->16k via polyphase, and appended to a ring kept at exactly
    WINDOW_SAMPLES (left-truncated). numpy/scipy import lazily on first push
    so this class is importable without them (tests that exercise the
    turn-loop logic never push real audio)."""

    def __init__(self) -> None:
        self._np = None
        self._resample_poly = None
        self._buf = None             # np.ndarray float32, <= WINDOW_SAMPLES

    def _ensure_libs(self) -> None:
        if self._np is not None:
            return
        import numpy as np
        from scipy.signal import resample_poly
        self._np = np
        self._resample_poly = resample_poly
        self._buf = np.zeros(0, dtype=np.float32)

    def push(self, pcm: bytes) -> None:
        """Append one inbound 24 kHz s16le chunk, resampled to 16 kHz."""
        if not pcm:
            return
        self._ensure_libs()
        np = self._np
        s16 = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        if s16.size == 0:
            return
        f16 = self._resample_poly(s16, _RESAMPLE_UP, _RESAMPLE_DOWN).astype(
            np.float32)
        self._buf = np.concatenate([self._buf, f16])
        if self._buf.size > WINDOW_SAMPLES:
            self._buf = self._buf[-WINDOW_SAMPLES:]

    def window_16k(self):
        """Current trailing window (np.float32, <= WINDOW_SAMPLES). Empty
        array before any audio. The detector pads/truncates to 8 s."""
        self._ensure_libs()
        return self._buf

    def clear(self) -> None:
        """Drop buffered audio — called when a turn finalizes so the next
        utterance's window does not include the previous one's tail."""
        if self._np is not None:
            self._buf = self._np.zeros(0, dtype=self._np.float32)


class SmartTurnDetector:
    """Bundled Smart Turn v3 ONNX endpoint predictor (CPU).

    load() is idempotent and thread-safe; predict_endpoint runs the graph.
    All heavy deps import inside load(), so constructing the detector (and
    importing this module) is free of onnxruntime/transformers/numpy."""

    def __init__(self, model_path: Optional[str] = None) -> None:
        self._model_path = model_path or BAKED_MODEL_PATH
        self._session = None
        self._feature_extractor = None
        self._np = None
        self._lock = threading.Lock()

    # -- model lifecycle ----------------------------------------------------

    def _resolve_model_path(self) -> str:
        """Baked path first (the supported, network-free path). Falls back to
        a HF cache download only if the bake is missing — fail LOUD if even
        that is unavailable rather than silently degrading turn detection."""
        if os.path.exists(self._model_path):
            return self._model_path
        logger.warning(
            "voice/turn-detect: baked model missing at %s; falling back to "
            "hf_hub_download(%s, %s) — this needs network and should not "
            "happen in the published image",
            self._model_path, SMART_TURN_REPO, SMART_TURN_FILE)
        from huggingface_hub import hf_hub_download
        return hf_hub_download(repo_id=SMART_TURN_REPO, filename=SMART_TURN_FILE)

    def load(self) -> None:
        """Build the ONNX session + Whisper feature extractor once."""
        if self._session is not None:
            return
        with self._lock:
            if self._session is not None:
                return
            import numpy as np
            import onnxruntime as ort
            from transformers import WhisperFeatureExtractor

            path = self._resolve_model_path()
            opts = ort.SessionOptions()
            # Single-threaded CPU inference: a turn check is ~13 ms and runs
            # in a run_in_executor worker; we do not want it spawning an
            # intra-op pool that contends with the gateway's threads.
            opts.intra_op_num_threads = 1
            opts.inter_op_num_threads = 1
            self._session = ort.InferenceSession(
                path, sess_options=opts, providers=["CPUExecutionProvider"])
            self._feature_extractor = WhisperFeatureExtractor(chunk_length=8)
            self._np = np
            logger.info("voice/turn-detect: Smart Turn model loaded from %s", path)

    @property
    def loaded(self) -> bool:
        return self._session is not None

    # -- inference ----------------------------------------------------------

    def predict_endpoint(self, audio_16k_f32) -> float:
        """Probability in [0, 1] that the user's turn is complete.

        ``audio_16k_f32`` is a 16 kHz mono float32 numpy array (any length;
        padded/truncated to the last 8 s). Returns 0.0 for empty input —
        an empty window is never end-of-turn. The graph output is already a
        probability (sigmoid baked in), so no extra activation here."""
        self.load()
        np = self._np
        audio = np.asarray(audio_16k_f32, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            return 0.0
        # Keep the LAST 8 s (the turn's tail carries the endpoint cue); the
        # feature extractor pads short input up to max_length.
        if audio.size > WINDOW_SAMPLES:
            audio = audio[-WINDOW_SAMPLES:]
        inputs = self._feature_extractor(
            audio,
            sampling_rate=SAMPLE_RATE_MODEL,
            return_tensors="np",
            padding="max_length",
            max_length=WINDOW_SAMPLES,
            truncation=True,
            do_normalize=True,
        )
        feats = inputs.input_features.squeeze(0).astype(np.float32)
        feats = np.expand_dims(feats, axis=0)
        outputs = self._session.run(None, {"input_features": feats})
        return float(np.asarray(outputs[0]).reshape(-1)[0])
