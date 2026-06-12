"""Gradium speech-to-text backend.

Gradium (https://gradium.ai — the Kyutai/Moshi team's commercial
spinoff) is a streaming-first speech API. This provider implements the
:class:`agent.transcription_provider.TranscriptionProvider` plugin
surface — the first in-tree consumer of the
``register_transcription_provider`` hook — for **batch** transcription
of recorded voice messages.

It deliberately uses Gradium's one-shot REST endpoint, not the
realtime websocket: the ``transcribe_audio`` tool surface is
file-in/dict-out, and the REST endpoint is the documented fit for
pre-recorded audio.

Protocol facts (docs.gradium.ai, REST speech-to-text guide):

- ``POST /api/post/speech/asr`` with the raw audio bytes as the request
  body (``Content-Type`` set to the audio MIME type, e.g.
  ``audio/wav``) returns newline-delimited JSON. The message vocabulary
  mirrors the websocket protocol: ``{"type": "text", "text": ...}``
  segments carry the transcript, ``end_text``/``end_of_stream`` mark
  completion, ``error`` reports pipeline failures. Consumers must
  tolerate and skip unknown message types (``ready``, ``step`` VAD
  frames, future additions).
- Optional advanced options (e.g. ``language``) ride in a JSON-encoded
  ``json_config`` query parameter.
- Auth: ``x-api-key`` header.

No SDK and no new dependency: everything rides on ``httpx``, which is a
core hermes dependency, so :meth:`is_available` only has to check the
API key.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.transcription_provider import TranscriptionProvider

logger = logging.getLogger(__name__)


API_BASE = "https://api.gradium.ai/api"
ASR_URL = f"{API_BASE}/post/speech/asr"

# Transcription runs roughly realtime against audio length; voice
# messages are capped upstream by the dispatcher's size validation, so a
# generous fixed ceiling is fine.
REQUEST_TIMEOUT_S = 300.0

# Audio MIME types by file extension. The dispatcher has already
# validated existence/size; unknown extensions fall back to audio/wav,
# matching the format the docs exercise.
_CONTENT_TYPES: Dict[str, str] = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".ogg": "audio/ogg",
    ".oga": "audio/ogg",
    ".opus": "audio/opus",
    ".flac": "audio/flac",
    ".m4a": "audio/mp4",
    ".mp4": "audio/mp4",
    ".webm": "audio/webm",
}
_DEFAULT_CONTENT_TYPE = "audio/wav"


def _content_type_for(file_path: str) -> str:
    return _CONTENT_TYPES.get(Path(file_path).suffix.lower(), _DEFAULT_CONTENT_TYPE)


class GradiumTranscriptionProvider(TranscriptionProvider):
    """Gradium REST batch transcription — file in, envelope dict out."""

    @property
    def name(self) -> str:
        return "gradium"

    @property
    def display_name(self) -> str:
        return "Gradium"

    def is_available(self) -> bool:
        return bool(self._api_key())

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Gradium",
            "badge": "paid",
            "tag": "Streaming-first STT (Kyutai team)",
            "env_vars": [
                {
                    "key": "GRADIUM_API_KEY",
                    "prompt": "Gradium API key",
                    "url": "https://gradium.ai",
                },
            ],
        }

    def list_models(self) -> List[Dict[str, Any]]:
        # Gradium exposes a single default STT model; surfacing it keeps
        # the `hermes tools` model column meaningful.
        return [
            {
                "id": "default",
                "display": "Gradium STT (default)",
                "max_audio_seconds": 300,
            },
        ]

    def transcribe(
        self,
        file_path: str,
        *,
        model: Optional[str] = None,
        language: Optional[str] = None,
        **extra: Any,
    ) -> Dict[str, Any]:
        api_key = self._api_key()
        if not api_key:
            return self._error(
                "GRADIUM_API_KEY is not set — required for the Gradium "
                "transcription provider. Get a key at https://gradium.ai"
            )

        try:
            audio = Path(file_path).read_bytes()
        except OSError as exc:
            return self._error(f"Could not read audio file {file_path}: {exc}")

        headers = {
            "x-api-key": api_key,
            "Content-Type": _content_type_for(file_path),
        }
        params: Dict[str, str] = {}
        if isinstance(language, str) and language.strip():
            params["json_config"] = json.dumps({"language": language.strip()})
        # ``model`` is accepted for ABC compatibility but ignored: the
        # REST endpoint runs Gradium's single default model.

        try:
            import httpx

            parts: List[str] = []
            with httpx.stream(
                "POST",
                ASR_URL,
                content=audio,
                headers=headers,
                params=params or None,
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
                    if mtype == "text":
                        text = msg.get("text")
                        if isinstance(text, str):
                            parts.append(text)
                    elif mtype == "error":
                        return self._error(
                            f"Gradium transcription failed: {msg.get('message')}"
                        )
                    elif mtype in ("end_text", "end_of_stream"):
                        if mtype == "end_of_stream":
                            break
                    # "ready", "step" VAD frames, and any future message
                    # types: deliberately ignored.
        except Exception as exc:  # noqa: BLE001 — ABC contract: never raise
            return self._error(f"Gradium transcription request failed: {exc}")

        transcript = "".join(parts).strip()
        return {
            "success": True,
            "transcript": transcript,
            "provider": self.name,
        }

    # ── internals ───────────────────────────────────────────────────────

    @staticmethod
    def _api_key() -> str:
        return (os.environ.get("GRADIUM_API_KEY") or "").strip()

    def _error(self, message: str) -> Dict[str, Any]:
        return {
            "success": False,
            "transcript": "",
            "error": message,
            "provider": self.name,
        }
