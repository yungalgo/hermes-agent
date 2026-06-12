"""Voice platform plugin — realtime Daily + Gradium voice calls.

Structural template: plugins/platforms/ntfy/adapter.py.
Turn orchestration lives in turn_loop.py (added later); this module owns
plugin registration, requirement checks, and the adapter lifecycle.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    SendResult,
)

logger = logging.getLogger(__name__)

DEFAULT_MODE = "standalone"
VALID_MODES = {"standalone", "orchestrated"}


def _daily_available() -> bool:
    try:
        import daily  # noqa: F401
        return True
    except ImportError:
        return False


def _websockets_available() -> bool:
    try:
        import websockets  # noqa: F401
        return True
    except ImportError:
        return False


def check_requirements() -> bool:
    """Deps importable + Gradium key present (cheap env read, no config load)."""
    if not _daily_available() or not _websockets_available():
        return False
    return bool(os.getenv("GRADIUM_API_KEY", "").strip())


def _resolve_mode(extra: Dict[str, Any]) -> str:
    mode = (extra.get("mode") or os.getenv("VOICE_MODE") or DEFAULT_MODE).strip()
    return mode if mode in VALID_MODES else DEFAULT_MODE


def validate_config(config) -> bool:
    extra = getattr(config, "extra", {}) or {}
    mode = _resolve_mode(extra)
    if mode == "standalone":
        return bool(os.getenv("DAILY_API_KEY", "").strip())
    return bool(
        os.getenv("SECOND_BRAIN_URL", "").strip()
        and os.getenv("SECOND_BRAIN_MCP_KEY", "").strip()
    )


def is_connected(config) -> bool:
    return bool(os.getenv("GRADIUM_API_KEY", "").strip()) and validate_config(config)


def register(ctx) -> None:
    """Plugin entry point — called by the Hermes plugin system at startup."""
    ctx.register_platform(
        name="voice",
        label="Voice",
        adapter_factory=lambda cfg: VoiceAdapter(cfg),
        check_fn=check_requirements,
        validate_config=validate_config,
        is_connected=is_connected,
        required_env=["GRADIUM_API_KEY"],
        install_hint='uv sync --extra voice-platform   # daily-python + websockets',
        pii_safe=True,
        emoji="📞",
        allow_update_command=False,
        platform_hint=(
            "You are on a live voice call. Speak naturally and BRIEFLY — "
            "1-3 short sentences per reply unless asked for detail. Never use "
            "markdown, bullet lists, code blocks, or URLs; everything you "
            "write is read aloud. If a tool call will take a while, say so "
            "in a few words first."
        ),
    )


class VoiceAdapter(BasePlatformAdapter):
    """Daily-room voice call adapter. Lifecycle filled in by the
    integration task; constructor/abstract-method contract fixed here."""

    def __init__(self, config: PlatformConfig):
        platform = Platform("voice")
        super().__init__(config=config, platform=platform)
        extra = config.extra or {}
        self._mode = _resolve_mode(extra)
        self._voice_id = (
            extra.get("voice_id") or os.getenv("VOICE_GRADIUM_VOICE_ID", "")
        )

    async def connect(self) -> bool:
        raise NotImplementedError("wired in the integration task")

    async def disconnect(self) -> None:
        raise NotImplementedError("wired in the integration task")

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        raise NotImplementedError("wired in the integration task")

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        return {"name": "Voice Call", "type": "dm"}
