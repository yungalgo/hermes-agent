"""Tests for the bundled Gradium TTS plugin (plugins/tts/gradium).

Fixture-driven — no live network calls. The ndjson streaming fixtures
mirror the live-verified Gradium protocol: a ``ready`` handshake first,
base64 ``audio`` chunks with word-timestamp ``text`` messages
interleaved, then a terminal ``end_of_stream``.
"""

from __future__ import annotations

import base64
import json
from unittest.mock import MagicMock, patch

import pytest

import plugins.tts.gradium as gradium_plugin
from plugins.tts.gradium.provider import (
    TTS_URL,
    GradiumTTSProvider,
    _resolve_gradium_format,
)


_PCM_CHUNK_A = b"\x01\x02" * 16
_PCM_CHUNK_B = b"\x03\x04" * 16


def _audio_msg(raw: bytes) -> str:
    return json.dumps(
        {
            "type": "audio",
            "audio": base64.b64encode(raw).decode("ascii"),
            "start_s": 0.0,
            "stop_s": 0.08,
            "stream_id": 0,
        }
    )


def _stream_lines(*, include_unknown: bool = False) -> list:
    """Canonical happy-path message sequence from the live protocol."""
    lines = [
        json.dumps(
            {
                "type": "ready",
                "sample_rate": 48000,
                "frame_size": 3840,
                "request_id": "req-1",
            }
        ),
        _audio_msg(_PCM_CHUNK_A),
        json.dumps(
            {"type": "text", "text": "hello", "start_s": 0.0, "stop_s": 0.3,
             "stream_id": 0}
        ),
    ]
    if include_unknown:
        lines.append(json.dumps({"type": "future_thing", "payload": 42}))
    lines.extend(
        [
            _audio_msg(_PCM_CHUNK_B),
            json.dumps({"type": "end_of_stream"}),
        ]
    )
    return lines


def _fake_httpx(
    *,
    post_content: bytes = b"",
    post_status_exc: Exception = None,
    stream_lines: list = None,
    voices_json=None,
):
    """Build a MagicMock httpx module covering post/stream/get."""
    fake = MagicMock()

    post_resp = MagicMock()
    post_resp.content = post_content
    if post_status_exc is not None:
        post_resp.raise_for_status.side_effect = post_status_exc
    fake.post.return_value = post_resp

    stream_resp = MagicMock()
    stream_resp.iter_lines.return_value = iter(stream_lines or [])
    stream_cm = MagicMock()
    stream_cm.__enter__ = MagicMock(return_value=stream_resp)
    stream_cm.__exit__ = MagicMock(return_value=False)
    fake.stream.return_value = stream_cm

    get_resp = MagicMock()
    get_resp.json.return_value = voices_json if voices_json is not None else []
    fake.get.return_value = get_resp

    return fake


def _patched_httpx(fake: MagicMock):
    return patch.dict("sys.modules", {"httpx": fake})


@pytest.fixture(autouse=True)
def _tmp_hermes_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    yield tmp_path


@pytest.fixture
def provider(monkeypatch) -> GradiumTTSProvider:
    monkeypatch.setenv("GRADIUM_API_KEY", "test-key")
    return GradiumTTSProvider()


# ── Metadata ────────────────────────────────────────────────────────────────


class TestMetadata:
    def test_name(self, provider):
        assert provider.name == "gradium"

    def test_display_name(self, provider):
        assert provider.display_name == "Gradium"

    def test_voice_compatible(self, provider):
        assert provider.voice_compatible is True

    def test_setup_schema_env_var(self, provider):
        schema = provider.get_setup_schema()
        assert schema["name"] == "Gradium"
        keys = [v["key"] for v in schema["env_vars"]]
        assert keys == ["GRADIUM_API_KEY"]

    def test_default_voice_unset_means_server_default(self, provider):
        assert provider.default_voice() is None

    def test_default_voice_from_config(self, provider, tmp_path):
        import yaml

        (tmp_path / "config.yaml").write_text(
            yaml.safe_dump({"tts": {"gradium": {"voice_id": "voice-abc"}}})
        )
        assert provider.default_voice() == "voice-abc"


# ── Availability ────────────────────────────────────────────────────────────


class TestAvailability:
    def test_no_api_key_unavailable(self, monkeypatch):
        monkeypatch.delenv("GRADIUM_API_KEY", raising=False)
        assert GradiumTTSProvider().is_available() is False

    def test_api_key_set_available(self, monkeypatch):
        monkeypatch.setenv("GRADIUM_API_KEY", "k")
        assert GradiumTTSProvider().is_available() is True


# ── Format mapping ──────────────────────────────────────────────────────────


class TestFormatMapping:
    @pytest.mark.parametrize(
        "requested,expected",
        [
            ("wav", ("wav", ".wav")),
            ("pcm", ("pcm", ".pcm")),
            ("opus", ("opus", ".opus")),
            ("ogg", ("opus", ".opus")),       # Opus is the ogg-family native
            ("mp3", ("wav", ".wav")),         # no mp3 upstream → lossless WAV
            ("flac", ("wav", ".wav")),
            ("MP3", ("wav", ".wav")),         # case-insensitive
            ("bogus", ("wav", ".wav")),       # unknown coerces to WAV
            (None, ("wav", ".wav")),
        ],
    )
    def test_resolution(self, requested, expected):
        assert _resolve_gradium_format(requested) == expected


# ── synthesize() ────────────────────────────────────────────────────────────


class TestSynthesize:
    def test_writes_bytes_and_returns_path(self, provider, tmp_path):
        out = tmp_path / "speech.wav"
        fake = _fake_httpx(post_content=b"RIFFfakewav")
        with _patched_httpx(fake):
            written = provider.synthesize("hello world", str(out), format="wav")
        assert written == str(out)
        assert out.read_bytes() == b"RIFFfakewav"
        _, kwargs = fake.post.call_args
        assert kwargs["headers"] == {"x-api-key": "test-key"}
        assert kwargs["json"]["only_audio"] is True
        assert kwargs["json"]["output_format"] == "wav"
        assert fake.post.call_args[0][0] == TTS_URL

    def test_mp3_request_rewrites_extension_to_wav(self, provider, tmp_path):
        out = tmp_path / "speech.mp3"
        fake = _fake_httpx(post_content=b"wav-bytes")
        with _patched_httpx(fake):
            written = provider.synthesize("hi", str(out), format="mp3")
        assert written == str(tmp_path / "speech.wav")
        assert fake.post.call_args[1]["json"]["output_format"] == "wav"

    def test_explicit_voice_wins_over_config(self, provider, tmp_path):
        import yaml

        (tmp_path / "config.yaml").write_text(
            yaml.safe_dump({"tts": {"gradium": {"voice_id": "config-voice"}}})
        )
        fake = _fake_httpx(post_content=b"x")
        with _patched_httpx(fake):
            provider.synthesize(
                "hi", str(tmp_path / "a.wav"), voice="arg-voice", format="wav"
            )
        assert fake.post.call_args[1]["json"]["voice_id"] == "arg-voice"

    def test_config_voice_used_when_no_arg(self, provider, tmp_path):
        import yaml

        (tmp_path / "config.yaml").write_text(
            yaml.safe_dump({"tts": {"gradium": {"voice_id": "config-voice"}}})
        )
        fake = _fake_httpx(post_content=b"x")
        with _patched_httpx(fake):
            provider.synthesize("hi", str(tmp_path / "a.wav"), format="wav")
        assert fake.post.call_args[1]["json"]["voice_id"] == "config-voice"

    def test_no_voice_omits_field(self, provider, tmp_path):
        fake = _fake_httpx(post_content=b"x")
        with _patched_httpx(fake):
            provider.synthesize("hi", str(tmp_path / "a.wav"), format="wav")
        assert "voice_id" not in fake.post.call_args[1]["json"]

    def test_missing_api_key_raises(self, monkeypatch, tmp_path):
        monkeypatch.delenv("GRADIUM_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="GRADIUM_API_KEY"):
            GradiumTTSProvider().synthesize("hi", str(tmp_path / "a.wav"))

    def test_http_error_propagates(self, provider, tmp_path):
        fake = _fake_httpx(post_status_exc=RuntimeError("401 unauthorized"))
        with _patched_httpx(fake):
            with pytest.raises(RuntimeError, match="401"):
                provider.synthesize("hi", str(tmp_path / "a.wav"), format="wav")


# ── stream() ────────────────────────────────────────────────────────────────


class TestStream:
    def test_yields_decoded_audio_chunks_in_order(self, provider):
        fake = _fake_httpx(stream_lines=_stream_lines())
        with _patched_httpx(fake):
            chunks = list(provider.stream("hello", format="opus"))
        assert chunks == [_PCM_CHUNK_A, _PCM_CHUNK_B]

    def test_ready_handshake_tolerated(self, provider):
        """First message is always {"type": "ready", ...} — never audio."""
        fake = _fake_httpx(stream_lines=_stream_lines())
        with _patched_httpx(fake):
            chunks = list(provider.stream("hello"))
        # ready contributed no chunk; both audio messages survived.
        assert len(chunks) == 2

    def test_unknown_message_types_skipped(self, provider):
        fake = _fake_httpx(stream_lines=_stream_lines(include_unknown=True))
        with _patched_httpx(fake):
            chunks = list(provider.stream("hello"))
        assert chunks == [_PCM_CHUNK_A, _PCM_CHUNK_B]

    def test_text_timestamps_not_yielded(self, provider):
        lines = [
            json.dumps({"type": "ready", "sample_rate": 48000}),
            json.dumps({"type": "text", "text": "word", "start_s": 0.0,
                        "stop_s": 0.2, "stream_id": 0}),
            json.dumps({"type": "end_of_stream"}),
        ]
        fake = _fake_httpx(stream_lines=lines)
        with _patched_httpx(fake):
            assert list(provider.stream("word")) == []

    def test_stops_at_end_of_stream(self, provider):
        lines = _stream_lines()
        lines.append(_audio_msg(b"\xff\xff"))  # after end_of_stream: ignored
        fake = _fake_httpx(stream_lines=lines)
        with _patched_httpx(fake):
            chunks = list(provider.stream("hello"))
        assert chunks == [_PCM_CHUNK_A, _PCM_CHUNK_B]

    def test_non_json_keepalive_lines_skipped(self, provider):
        lines = _stream_lines()
        lines.insert(1, "")            # blank keepalive
        lines.insert(2, "not json")    # garbage line
        fake = _fake_httpx(stream_lines=lines)
        with _patched_httpx(fake):
            chunks = list(provider.stream("hello"))
        assert chunks == [_PCM_CHUNK_A, _PCM_CHUNK_B]

    def test_server_error_message_raises(self, provider):
        lines = [
            json.dumps({"type": "ready"}),
            json.dumps({"type": "error", "message": "quota exceeded"}),
        ]
        fake = _fake_httpx(stream_lines=lines)
        with _patched_httpx(fake):
            with pytest.raises(RuntimeError, match="quota exceeded"):
                list(provider.stream("hello"))

    def test_default_format_is_opus(self, provider):
        fake = _fake_httpx(stream_lines=_stream_lines())
        with _patched_httpx(fake):
            list(provider.stream("hello"))
        body = fake.stream.call_args[1]["json"]
        assert body["output_format"] == "opus"
        assert "only_audio" not in body

    def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("GRADIUM_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="GRADIUM_API_KEY"):
            list(GradiumTTSProvider().stream("hello"))


# ── list_voices() ───────────────────────────────────────────────────────────


class TestListVoices:
    def test_maps_catalog_entries(self, provider):
        catalog = [
            {"uid": "v1", "name": "Audrey", "description": "warm female",
             "language": "en", "tags": ["catalog"]},
            {"uid": "v2", "name": "Toby"},
            {"name": "no-uid-entry"},          # dropped: uid required
            "not-a-dict",                       # dropped: malformed
        ]
        fake = _fake_httpx(voices_json=catalog)
        with _patched_httpx(fake):
            voices = provider.list_voices()
        assert voices == [
            {"id": "v1", "display": "Audrey — warm female", "language": "en"},
            {"id": "v2", "display": "Toby"},
        ]
        _, kwargs = fake.get.call_args
        assert kwargs["params"] == {"include_catalog": "true"}

    def test_no_api_key_returns_empty(self, monkeypatch):
        monkeypatch.delenv("GRADIUM_API_KEY", raising=False)
        assert GradiumTTSProvider().list_voices() == []

    def test_network_failure_returns_empty(self, provider):
        fake = MagicMock()
        fake.get.side_effect = OSError("connection refused")
        with _patched_httpx(fake):
            assert provider.list_voices() == []

    def test_non_list_payload_returns_empty(self, provider):
        fake = _fake_httpx(voices_json={"unexpected": "shape"})
        with _patched_httpx(fake):
            assert provider.list_voices() == []


# ── register() hook ─────────────────────────────────────────────────────────


class TestRegister:
    def test_registers_provider_instance(self):
        ctx = MagicMock()
        gradium_plugin.register(ctx)
        ctx.register_tts_provider.assert_called_once()
        (registered,), _ = ctx.register_tts_provider.call_args
        assert isinstance(registered, GradiumTTSProvider)
        assert registered.name == "gradium"
