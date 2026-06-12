"""Tests for the bundled Gradium transcription plugin
(plugins/transcription/gradium).

Fixture-driven — no live network calls. The ndjson fixtures mirror the
documented REST protocol, which reuses the websocket message
vocabulary: optional ``ready``/``step`` frames, ``text`` transcript
segments, ``end_text``, ``end_of_stream``, and ``error``.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

import plugins.transcription.gradium as gradium_plugin
from plugins.transcription.gradium.provider import (
    ASR_URL,
    GradiumTranscriptionProvider,
    _content_type_for,
)


def _stream_lines(*, include_unknown: bool = False) -> list:
    lines = [
        json.dumps({"type": "ready", "sample_rate": 24000, "frame_size": 1920}),
        json.dumps(
            {
                "type": "step",
                "step_idx": 1,
                "step_duration_s": 0.08,
                "total_duration_s": 0.08,
                "vad": [
                    {"horizon_s": 0.5, "inactivity_prob": 0.1},
                    {"horizon_s": 1.0, "inactivity_prob": 0.05},
                    {"horizon_s": 2.0, "inactivity_prob": 0.02},
                    {"horizon_s": 3.0, "inactivity_prob": 0.01},
                ],
            }
        ),
        json.dumps({"type": "text", "text": "hello", "start_s": 0.0}),
    ]
    if include_unknown:
        lines.append(json.dumps({"type": "future_thing", "payload": 42}))
    lines.extend(
        [
            json.dumps({"type": "text", "text": " world", "start_s": 0.4}),
            json.dumps({"type": "end_text"}),
            json.dumps({"type": "end_of_stream"}),
        ]
    )
    return lines


def _fake_httpx(*, stream_lines: list = None, status_exc: Exception = None):
    fake = MagicMock()
    resp = MagicMock()
    resp.iter_lines.return_value = iter(stream_lines or [])
    if status_exc is not None:
        resp.raise_for_status.side_effect = status_exc
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=resp)
    cm.__exit__ = MagicMock(return_value=False)
    fake.stream.return_value = cm
    return fake


def _patched_httpx(fake: MagicMock):
    return patch.dict("sys.modules", {"httpx": fake})


@pytest.fixture(autouse=True)
def _tmp_hermes_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    yield tmp_path


@pytest.fixture
def provider(monkeypatch) -> GradiumTranscriptionProvider:
    monkeypatch.setenv("GRADIUM_API_KEY", "test-key")
    return GradiumTranscriptionProvider()


@pytest.fixture
def audio_file(tmp_path):
    f = tmp_path / "voice.wav"
    f.write_bytes(b"RIFFfakewavbytes")
    return f


# ── Metadata ────────────────────────────────────────────────────────────────


class TestMetadata:
    def test_name(self, provider):
        assert provider.name == "gradium"

    def test_display_name(self, provider):
        assert provider.display_name == "Gradium"

    def test_setup_schema_env_var(self, provider):
        schema = provider.get_setup_schema()
        assert schema["name"] == "Gradium"
        assert [v["key"] for v in schema["env_vars"]] == ["GRADIUM_API_KEY"]

    def test_default_model(self, provider):
        assert provider.default_model() == "default"

    def test_list_models_single_entry(self, provider):
        models = provider.list_models()
        assert [m["id"] for m in models] == ["default"]


# ── Availability ────────────────────────────────────────────────────────────


class TestAvailability:
    def test_no_api_key_unavailable(self, monkeypatch):
        monkeypatch.delenv("GRADIUM_API_KEY", raising=False)
        assert GradiumTranscriptionProvider().is_available() is False

    def test_api_key_set_available(self, monkeypatch):
        monkeypatch.setenv("GRADIUM_API_KEY", "k")
        assert GradiumTranscriptionProvider().is_available() is True


# ── Content-type mapping ────────────────────────────────────────────────────


class TestContentType:
    @pytest.mark.parametrize(
        "filename,expected",
        [
            ("a.wav", "audio/wav"),
            ("a.mp3", "audio/mpeg"),
            ("a.ogg", "audio/ogg"),
            ("a.oga", "audio/ogg"),
            ("a.opus", "audio/opus"),
            ("a.flac", "audio/flac"),
            ("a.m4a", "audio/mp4"),
            ("a.webm", "audio/webm"),
            ("a.WAV", "audio/wav"),          # case-insensitive
            ("a.unknown", "audio/wav"),      # fallback
            ("noext", "audio/wav"),
        ],
    )
    def test_mapping(self, filename, expected):
        assert _content_type_for(filename) == expected


# ── transcribe() ────────────────────────────────────────────────────────────


class TestTranscribe:
    def test_success_envelope(self, provider, audio_file):
        fake = _fake_httpx(stream_lines=_stream_lines())
        with _patched_httpx(fake):
            result = provider.transcribe(str(audio_file))
        assert result == {
            "success": True,
            "transcript": "hello world",
            "provider": "gradium",
        }

    def test_request_shape(self, provider, audio_file):
        fake = _fake_httpx(stream_lines=_stream_lines())
        with _patched_httpx(fake):
            provider.transcribe(str(audio_file))
        args, kwargs = fake.stream.call_args
        assert args == ("POST", ASR_URL)
        assert kwargs["content"] == b"RIFFfakewavbytes"
        assert kwargs["headers"]["x-api-key"] == "test-key"
        assert kwargs["headers"]["Content-Type"] == "audio/wav"
        assert kwargs["params"] is None

    def test_ready_and_step_frames_tolerated(self, provider, audio_file):
        """ready handshake and step VAD frames never reach the transcript."""
        fake = _fake_httpx(stream_lines=_stream_lines())
        with _patched_httpx(fake):
            result = provider.transcribe(str(audio_file))
        assert result["transcript"] == "hello world"

    def test_unknown_message_types_skipped(self, provider, audio_file):
        fake = _fake_httpx(stream_lines=_stream_lines(include_unknown=True))
        with _patched_httpx(fake):
            result = provider.transcribe(str(audio_file))
        assert result["success"] is True
        assert result["transcript"] == "hello world"

    def test_non_json_keepalive_lines_skipped(self, provider, audio_file):
        lines = _stream_lines()
        lines.insert(1, "")
        lines.insert(2, "not json")
        fake = _fake_httpx(stream_lines=lines)
        with _patched_httpx(fake):
            result = provider.transcribe(str(audio_file))
        assert result["success"] is True
        assert result["transcript"] == "hello world"

    def test_stops_at_end_of_stream(self, provider, audio_file):
        lines = _stream_lines()
        lines.append(json.dumps({"type": "text", "text": "IGNORED"}))
        fake = _fake_httpx(stream_lines=lines)
        with _patched_httpx(fake):
            result = provider.transcribe(str(audio_file))
        assert result["transcript"] == "hello world"

    def test_language_hint_rides_json_config(self, provider, audio_file):
        fake = _fake_httpx(stream_lines=_stream_lines())
        with _patched_httpx(fake):
            provider.transcribe(str(audio_file), language="ja")
        params = fake.stream.call_args[1]["params"]
        assert json.loads(params["json_config"]) == {"language": "ja"}

    def test_blank_language_omitted(self, provider, audio_file):
        fake = _fake_httpx(stream_lines=_stream_lines())
        with _patched_httpx(fake):
            provider.transcribe(str(audio_file), language="   ")
        assert fake.stream.call_args[1]["params"] is None

    def test_mp3_content_type(self, provider, tmp_path):
        f = tmp_path / "voice.mp3"
        f.write_bytes(b"mp3bytes")
        fake = _fake_httpx(stream_lines=_stream_lines())
        with _patched_httpx(fake):
            provider.transcribe(str(f))
        assert fake.stream.call_args[1]["headers"]["Content-Type"] == "audio/mpeg"


# ── transcribe() error envelopes (never raises) ─────────────────────────────


class TestTranscribeErrors:
    def _assert_error_envelope(self, result):
        assert result["success"] is False
        assert result["transcript"] == ""
        assert result["provider"] == "gradium"
        assert result["error"]

    def test_missing_api_key(self, monkeypatch, audio_file):
        monkeypatch.delenv("GRADIUM_API_KEY", raising=False)
        result = GradiumTranscriptionProvider().transcribe(str(audio_file))
        self._assert_error_envelope(result)
        assert "GRADIUM_API_KEY" in result["error"]

    def test_unreadable_file(self, provider, tmp_path):
        result = provider.transcribe(str(tmp_path / "missing.wav"))
        self._assert_error_envelope(result)
        assert "missing.wav" in result["error"]

    def test_server_error_message(self, provider, audio_file):
        lines = [
            json.dumps({"type": "ready"}),
            json.dumps({"type": "error", "message": "pipeline failed"}),
        ]
        fake = _fake_httpx(stream_lines=lines)
        with _patched_httpx(fake):
            result = provider.transcribe(str(audio_file))
        self._assert_error_envelope(result)
        assert "pipeline failed" in result["error"]

    def test_http_status_error(self, provider, audio_file):
        fake = _fake_httpx(status_exc=RuntimeError("401 unauthorized"))
        with _patched_httpx(fake):
            result = provider.transcribe(str(audio_file))
        self._assert_error_envelope(result)
        assert "401" in result["error"]

    def test_network_exception(self, provider, audio_file):
        fake = MagicMock()
        fake.stream.side_effect = OSError("connection refused")
        with _patched_httpx(fake):
            result = provider.transcribe(str(audio_file))
        self._assert_error_envelope(result)
        assert "connection refused" in result["error"]


# ── register() hook ─────────────────────────────────────────────────────────


class TestRegister:
    def test_registers_provider_instance(self):
        ctx = MagicMock()
        gradium_plugin.register(ctx)
        ctx.register_transcription_provider.assert_called_once()
        (registered,), _ = ctx.register_transcription_provider.call_args
        assert isinstance(registered, GradiumTranscriptionProvider)
        assert registered.name == "gradium"
