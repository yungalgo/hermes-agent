"""Tests for the local Smart Turn end-of-turn detector module (ENG-555).

The real ONNX model (onnxruntime + transformers) is NOT loaded here — those
heavy deps import lazily inside SmartTurnDetector.load()/predict_endpoint and
RollingAudioBuffer.push(). These tests cover:
  - rms_energy() (pure stdlib, the silence/speech gate),
  - the module's geometry constants,
  - the detector's empty-input contract via a stubbed session/extractor,
  - the resolve-model-path fallback ordering,
without requiring numpy/scipy/onnxruntime to be installed.
"""

from __future__ import annotations

import sys
import types

import pytest

from tests.gateway._voice_module_loader import load_voice_module

td = load_voice_module("turn_detection")


# -- rms_energy --------------------------------------------------------------


def test_rms_energy_silence_is_zero():
    assert td.rms_energy(b"\x00\x00" * 1920) == 0.0


def test_rms_energy_empty_is_zero():
    assert td.rms_energy(b"") == 0.0


def test_rms_energy_loud_above_floor():
    loud = td.rms_energy(b"\x00\x20" * 1920)   # 0x2000 = 8192 per sample
    quiet = td.rms_energy(b"\x10\x00" * 1920)  # 0x0010 = 16 per sample
    assert loud > 1000.0
    assert quiet < 100.0
    assert loud > quiet


def test_geometry_constants():
    # 24kHz in, 16kHz model, 8s window -> 128000 samples; 2/3 polyphase.
    assert td.SAMPLE_RATE_IN == 24000
    assert td.SAMPLE_RATE_MODEL == 16000
    assert td.WINDOW_SECONDS == 8
    assert td.WINDOW_SAMPLES == 128000
    assert td._RESAMPLE_UP == 2 and td._RESAMPLE_DOWN == 3


# -- detector contract (without the real model) ------------------------------


def _np_available() -> bool:
    try:
        import numpy  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _np_available(), reason="numpy not installed")
def test_predict_endpoint_empty_returns_zero():
    """Empty window -> 0.0 (never end-of-turn), without touching the model.
    Uses real numpy for faithful array semantics; the session is a stub that
    must NOT be called for empty input."""
    import numpy as np

    det = td.SmartTurnDetector(model_path="/nonexistent/model.onnx")
    det._np = np
    det._session = types.SimpleNamespace(
        run=lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("session must not run on empty input")))
    det._feature_extractor = lambda *a, **k: None
    assert det.predict_endpoint(np.zeros(0, dtype=np.float32)) == 0.0


@pytest.mark.skipif(not _np_available(), reason="numpy not installed")
def test_predict_endpoint_returns_session_probability():
    """Non-empty window -> the session's output probability, with real numpy
    and a stub session + passthrough feature extractor (so onnxruntime /
    transformers are not needed). Verifies the output is read as a probability
    (no extra sigmoid) and the input tensor name is 'input_features'."""
    import numpy as np

    det = td.SmartTurnDetector(model_path="/nonexistent/model.onnx")
    det._np = np
    seen = {}

    def _run(_outputs, feed):
        seen["feed_keys"] = list(feed.keys())
        return [np.array([[0.83]], dtype=np.float32)]

    det._session = types.SimpleNamespace(run=_run)

    class _Feats:
        # WhisperFeatureExtractor output exposes .input_features (1, 80, T).
        input_features = np.zeros((1, 80, 4), dtype=np.float32)

    det._feature_extractor = lambda *a, **k: _Feats()
    prob = det.predict_endpoint(np.full(16000, 0.1, dtype=np.float32))
    assert prob == pytest.approx(0.83)
    assert seen["feed_keys"] == ["input_features"]


def test_resolve_model_path_prefers_baked(tmp_path):
    baked = tmp_path / "smart-turn.onnx"
    baked.write_bytes(b"\x00")
    det = td.SmartTurnDetector(model_path=str(baked))
    assert det._resolve_model_path() == str(baked)


def test_resolve_model_path_falls_back_to_hub(monkeypatch):
    det = td.SmartTurnDetector(model_path="/definitely/missing/model.onnx")
    called = {}

    def _fake_download(repo_id, filename):
        called["repo"] = repo_id
        called["file"] = filename
        return "/tmp/downloaded.onnx"

    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.hf_hub_download = _fake_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    assert det._resolve_model_path() == "/tmp/downloaded.onnx"
    assert called["repo"] == td.SMART_TURN_REPO
    assert called["file"] == td.SMART_TURN_FILE


def test_rolling_buffer_clear_without_libs_is_safe():
    # clear() before any push (numpy never imported) must not raise.
    buf = td.RollingAudioBuffer()
    buf.clear()
    assert buf._buf is None
