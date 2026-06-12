"""Gradium TTS backend.

Gradium (https://gradium.ai — the Kyutai/Moshi team's commercial spinoff)
is a streaming-first speech API. This provider implements the
:class:`agent.tts_provider.TTSProvider` plugin surface from issue #30398
— the first in-tree consumer of the hook — and supersedes the stalled
in-tree provider approach (#17382) with the plugin shape plus the
streaming surface that approach lacked.

Protocol facts (verified against the live API, 2026-06):

- ``POST /api/post/speech/tts`` with ``{"text", "voice_id",
  "output_format", "only_audio": true}`` returns the raw audio bytes.
  ``output_format`` accepts ONLY ``wav`` | ``pcm`` | ``opus`` — there is
  no mp3. Requests for unsupported formats are mapped to the closest
  Gradium-native equivalent and the output path's extension is rewritten
  to match (allowed by the ``TTSProvider.synthesize`` contract).
- The same endpoint WITHOUT ``only_audio`` streams newline-delimited
  JSON messages mirroring Gradium's websocket protocol: a
  ``{"type": "ready", ...}`` handshake first, base64 ``{"type": "audio"}``
  chunks, word-timestamp ``{"type": "text"}`` messages interleaved with
  the audio, then a final ``{"type": "end_of_stream"}``. Consumers must
  tolerate and skip unknown message types — the server adds new ones
  without versioning.
- Voices: ``GET /api/voices/?include_catalog=true`` returns
  ``[{uid, name, description, language, tags}, ...]``.
- Auth: ``x-api-key`` header.

No SDK and no new dependency: everything rides on ``httpx``, which is a
core hermes dependency, so :meth:`is_available` only has to check the
API key.
"""

from __future__ import annotations

import base64
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from agent.tts_provider import DEFAULT_OUTPUT_FORMAT, TTSProvider

logger = logging.getLogger(__name__)


API_BASE = "https://api.gradium.ai/api"
TTS_URL = f"{API_BASE}/post/speech/tts"
VOICES_URL = f"{API_BASE}/voices/"

# Synthesis can stream for the length of the spoken audio; keep a generous
# ceiling so long paragraphs don't get cut off mid-sentence.
REQUEST_TIMEOUT_S = 120.0
VOICES_TIMEOUT_S = 30.0

# Map the hermes-side output format vocabulary onto Gradium's native
# format enum (wav | pcm | opus) + the file extension the written audio
# really has. mp3/flac requests get WAV (lossless, universally readable —
# the gateway's ffmpeg step re-encodes downstream when needed); ogg
# requests get Opus.
_FORMAT_MAP: Dict[str, Tuple[str, str]] = {
    "wav": ("wav", ".wav"),
    "pcm": ("pcm", ".pcm"),
    "opus": ("opus", ".opus"),
    "ogg": ("opus", ".opus"),
    "mp3": ("wav", ".wav"),
    "flac": ("wav", ".wav"),
}
_DEFAULT_FORMAT_ENTRY = _FORMAT_MAP["wav"]


def _resolve_gradium_format(requested: Optional[str]) -> Tuple[str, str]:
    """Return ``(gradium_output_format, file_suffix)`` for *requested*."""
    if not isinstance(requested, str):
        return _DEFAULT_FORMAT_ENTRY
    return _FORMAT_MAP.get(requested.strip().lower(), _DEFAULT_FORMAT_ENTRY)


def _load_gradium_config() -> Dict[str, Any]:
    """Read the ``tts.gradium`` section from config.yaml ({} on any failure)."""
    try:
        from hermes_cli.config import load_config

        cfg = load_config() or {}
        tts_cfg = cfg.get("tts") or {}
        section = tts_cfg.get("gradium") or {}
        return section if isinstance(section, dict) else {}
    except Exception as exc:  # noqa: BLE001 — config read is best-effort
        logger.debug("Could not load tts.gradium config: %s", exc)
        return {}


class GradiumTTSProvider(TTSProvider):
    """Gradium REST TTS — non-streaming file synthesis + chunked streaming."""

    @property
    def name(self) -> str:
        return "gradium"

    @property
    def display_name(self) -> str:
        return "Gradium"

    @property
    def voice_compatible(self) -> bool:
        # Output is real speech audio; Gradium can emit Opus natively and
        # the gateway's ffmpeg step converts WAV when a voice bubble needs
        # Opus. Safe to opt in.
        return True

    def is_available(self) -> bool:
        return bool(self._api_key())

    # ── synthesis ───────────────────────────────────────────────────────

    def synthesize(
        self,
        text: str,
        output_path: str,
        *,
        voice: Optional[str] = None,
        model: Optional[str] = None,
        speed: Optional[float] = None,
        format: str = DEFAULT_OUTPUT_FORMAT,
        **extra: Any,
    ) -> str:
        api_key = self._api_key()
        if not api_key:
            raise RuntimeError(
                "GRADIUM_API_KEY is not set — required for the Gradium TTS "
                "provider. Get a key at https://gradium.ai"
            )
        gradium_format, suffix = _resolve_gradium_format(format)
        body: Dict[str, Any] = {
            "text": text,
            "output_format": gradium_format,
            # Without only_audio the endpoint streams JSON messages — see
            # stream() below for that mode.
            "only_audio": True,
        }
        voice_id = self._resolve_voice(voice)
        if voice_id:
            body["voice_id"] = voice_id
        # ``model`` and ``speed`` are accepted for ABC compatibility but
        # ignored: Gradium exposes a single default model and no
        # speech-rate control on this endpoint.

        import httpx

        resp = httpx.post(
            TTS_URL,
            json=body,
            headers=self._headers(api_key),
            timeout=REQUEST_TIMEOUT_S,
        )
        resp.raise_for_status()

        final_path = Path(output_path).with_suffix(suffix)
        final_path.write_bytes(resp.content)
        return str(final_path)

    def stream(
        self,
        text: str,
        *,
        voice: Optional[str] = None,
        model: Optional[str] = None,
        format: str = "opus",
        **extra: Any,
    ) -> Iterator[bytes]:
        """Stream synthesized audio as chunked bytes.

        One request per call = one logical turn, the same per-turn design
        as Gradium's websocket: the ndjson response carries the identical
        message vocabulary (``ready`` handshake, base64 ``audio`` chunks,
        word-timestamp ``text`` messages, terminal ``end_of_stream``).
        Unknown message types are skipped so server-side additions never
        break playback.
        """
        api_key = self._api_key()
        if not api_key:
            raise RuntimeError(
                "GRADIUM_API_KEY is not set — required for the Gradium TTS "
                "provider. Get a key at https://gradium.ai"
            )
        gradium_format, _suffix = _resolve_gradium_format(format)
        body: Dict[str, Any] = {"text": text, "output_format": gradium_format}
        voice_id = self._resolve_voice(voice)
        if voice_id:
            body["voice_id"] = voice_id

        import httpx

        with httpx.stream(
            "POST",
            TTS_URL,
            json=body,
            headers=self._headers(api_key),
            timeout=REQUEST_TIMEOUT_S,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line or not line.strip():
                    continue
                try:
                    msg = json.loads(line)
                except ValueError:
                    # Tolerate keepalive/garbage lines between messages.
                    continue
                if not isinstance(msg, dict):
                    continue
                mtype = msg.get("type")
                if mtype == "audio":
                    audio_b64 = msg.get("audio")
                    if audio_b64:
                        yield base64.b64decode(audio_b64)
                elif mtype == "error":
                    raise RuntimeError(
                        f"Gradium TTS stream error: {msg.get('message')}"
                    )
                elif mtype == "end_of_stream":
                    break
                # "ready" handshake, "text" word timestamps, and any
                # future message types: deliberately ignored.

    # ── catalog / picker metadata ───────────────────────────────────────

    def list_voices(self) -> List[Dict[str, Any]]:
        api_key = self._api_key()
        if not api_key:
            return []
        try:
            import httpx

            resp = httpx.get(
                VOICES_URL,
                params={"include_catalog": "true"},
                headers=self._headers(api_key),
                timeout=VOICES_TIMEOUT_S,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:  # noqa: BLE001 — catalog is best-effort
            logger.debug("Gradium voice listing failed: %s", exc)
            return []
        voices: List[Dict[str, Any]] = []
        if not isinstance(data, list):
            return voices
        for entry in data:
            if not isinstance(entry, dict):
                continue
            uid = entry.get("uid")
            if not uid:
                continue
            name = entry.get("name") or uid
            description = entry.get("description")
            voice: Dict[str, Any] = {
                "id": uid,
                "display": f"{name} — {description}" if description else name,
            }
            language = entry.get("language")
            if language:
                voice["language"] = language
            voices.append(voice)
        return voices

    def default_voice(self) -> Optional[str]:
        # Deliberately avoids the ABC default (which would hit the voices
        # API): an unset voice means "server default voice", which is a
        # valid request.
        return self._resolve_voice(None)

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Gradium",
            "badge": "paid",
            "tag": "Streaming-first TTS (Kyutai team) — wav/pcm/opus",
            "env_vars": [
                {
                    "key": "GRADIUM_API_KEY",
                    "prompt": "Gradium API key",
                    "url": "https://gradium.ai",
                },
            ],
        }

    # ── internals ───────────────────────────────────────────────────────

    @staticmethod
    def _api_key() -> str:
        return (os.environ.get("GRADIUM_API_KEY") or "").strip()

    @staticmethod
    def _headers(api_key: str) -> Dict[str, str]:
        return {"x-api-key": api_key}

    @staticmethod
    def _resolve_voice(voice: Optional[str]) -> Optional[str]:
        """Explicit voice argument > ``tts.gradium.voice_id`` > server default."""
        if isinstance(voice, str) and voice.strip():
            return voice.strip()
        configured = _load_gradium_config().get("voice_id")
        if isinstance(configured, str) and configured.strip():
            return configured.strip()
        return None
