"""Voice platform plugin — realtime Daily + Gradium voice calls.

Structural template: plugins/platforms/ntfy/adapter.py.
Turn orchestration lives in turn_loop.py (added later); this module owns
plugin registration, requirement checks, and the adapter lifecycle.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, Dict, Optional, Tuple

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    SendResult,
)

logger = logging.getLogger(__name__)

DEFAULT_MODE = "standalone"
VALID_MODES = {"standalone", "orchestrated"}

DAILY_API = "https://api.daily.co/v1"
STANDALONE_ROOM_TTL_S = 3600


def _voice_modules():
    """Sibling voice modules via the discord dual-import pattern
    (plugins/platforms/discord/adapter.py:1991-1993): flat import for the
    test loader, relative import for the production package
    (hermes_plugins.platforms__voice)."""
    try:
        import control_channel
        import daily_transport
        import gradium_stt
        import gradium_tts
        import turn_loop
        import vamp
    except ImportError:
        from . import (
            control_channel,
            daily_transport,
            gradium_stt,
            gradium_tts,
            turn_loop,
            vamp,
        )
    return (control_channel, daily_transport, gradium_stt, gradium_tts,
            turn_loop, vamp)


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
    """Daily-room voice call adapter.

    Orchestrated mode: subscribe to the control plane over outbound SSE;
    ``join_room`` events join the minted Daily room and start the turn
    loop, ``leave_room`` tears the call down. The agent never holds the
    Daily API key.

    Standalone mode: this agent holds DAILY_API_KEY, creates its own
    private room + meeting token via the Daily REST API at connect time,
    joins immediately, and logs the room URL for the owner to share.

    One active call at a time (daily-python virtual devices are
    process-level singletons — see daily_transport.py).
    """

    def __init__(self, config: PlatformConfig):
        platform = Platform("voice")
        super().__init__(config=config, platform=platform)
        extra = config.extra or {}
        self._mode = _resolve_mode(extra)
        self._voice_id = (
            extra.get("voice_id") or os.getenv("VOICE_GRADIUM_VOICE_ID", "")
        )
        self._call_lock = asyncio.Lock()
        self._active_call: Optional[Dict[str, Any]] = None
        self._control = None

    async def connect(self) -> bool:
        control_channel, *_ = _voice_modules()
        if self._mode == "orchestrated":
            self._control = control_channel.ControlChannel(
                os.environ["SECOND_BRAIN_URL"],
                os.environ["SECOND_BRAIN_MCP_KEY"],
                self._handle_control_event,
            )
            self._control.start()
            self._mark_connected()
            logger.info("voice: orchestrated mode — awaiting call commands")
            return True
        # standalone: create our own room + token, join immediately
        room_url, token = await self._create_standalone_room()
        await self._start_call(room_url, token)
        self._mark_connected()
        logger.info("voice: standalone room ready — share this URL: %s", room_url)
        return True

    async def _handle_control_event(self, event: Dict[str, Any]) -> None:
        action = event.get("action")
        if action == "join_room":
            await self._start_call(event["roomUrl"], event["token"])
        elif action == "leave_room":
            await self._end_call()

    async def _start_call(self, room_url: str, token: str) -> None:
        (_, daily_transport, gradium_stt, gradium_tts, turn_loop,
         vamp_mod) = _voice_modules()
        async with self._call_lock:
            if self._active_call is not None:
                await self._end_call_locked()
            loop = asyncio.get_running_loop()
            api_key = os.environ["GRADIUM_API_KEY"]
            extra = self.config.extra or {}
            # Vamp clips synthesize in the BACKGROUND (start() returns
            # immediately) — clip generation must never delay the join;
            # the vamp stays disabled until clips are ready (notes §16).
            vamp_cache = None
            if extra.get("vamp_enabled", True):
                vamp_cache = vamp_mod.VampCache(
                    api_key, self._voice_id,
                    texts=extra.get("vamp_texts"))
                vamp_cache.start()
            stt = gradium_stt.GradiumSTT(api_key)
            await stt.start()

            # Inbound audio fans out to STT and to the turn loop's local
            # energy barge-in (vloop exists only after the transport, so
            # route through a late-bound cell).
            vloop_cell: Dict[str, Any] = {}

            async def on_audio_in(pcm: bytes) -> None:
                vl = vloop_cell.get("vloop")
                if vl is not None:
                    await vl.on_inbound_audio(pcm)
                await stt.send_audio(pcm)

            transport = daily_transport.DailyTransport(loop, on_audio_in)
            await transport.join(room_url, token)

            async def tts_factory(on_audio):
                turn = gradium_tts.GradiumTTSTurn(api_key, self._voice_id, on_audio)
                await turn.open()
                return turn

            vloop = turn_loop.VoiceTurnLoop(
                stt, tts_factory, transport, extra=extra, vamp=vamp_cache)
            vloop_cell["vloop"] = vloop
            task = asyncio.create_task(vloop.run())
            self._active_call = {
                "stt": stt, "transport": transport, "loop": vloop,
                "task": task, "vamp": vamp_cache}
            logger.info("voice: call started in %s", room_url)

    async def _end_call(self) -> None:
        async with self._call_lock:
            await self._end_call_locked()

    async def _end_call_locked(self) -> None:
        call = self._active_call
        if call is None:
            return
        self._active_call = None
        vamp_cache = call.get("vamp")
        if vamp_cache is not None:
            await vamp_cache.stop()
        await call["loop"].stop()
        call["task"].cancel()
        try:
            await call["task"]
        except (asyncio.CancelledError, Exception):
            pass
        await call["stt"].stop()
        await call["transport"].leave()
        logger.info("voice: call ended")

    async def _create_standalone_room(self) -> Tuple[str, str]:
        import httpx

        daily_key = os.environ["DAILY_API_KEY"]
        headers = {"Authorization": f"Bearer {daily_key}"}
        exp = int(time.time()) + STANDALONE_ROOM_TTL_S
        async with httpx.AsyncClient(timeout=15.0) as client:
            room_resp = await client.post(
                f"{DAILY_API}/rooms", headers=headers,
                json={"privacy": "private", "properties": {"exp": exp}},
            )
            room_resp.raise_for_status()
            room = room_resp.json()
            token_resp = await client.post(
                f"{DAILY_API}/meeting-tokens", headers=headers,
                json={"properties": {"room_name": room["name"],
                                     "is_owner": False, "exp": exp}},
            )
            token_resp.raise_for_status()
            tok = token_resp.json()
        return room["url"], tok["token"]

    async def disconnect(self) -> None:
        if self._control is not None:
            await self._control.stop()
            self._control = None
        await self._end_call()

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        """Out-of-band sends (cron etc.): speak if a call is live.

        The gateway never routes normal chat through this adapter (the
        turn loop owns the conversation), but the abstract method must
        exist (base.py:2257). Honest failure beats a hidden queue.
        """
        call = self._active_call
        if call is None:
            return SendResult(success=False, error="no active voice call")
        tts = call["loop"]._tts
        if tts is None:
            return SendResult(
                success=False,
                error="agent not mid-utterance; queueing not supported in v0",
            )
        await tts.send_text(content)
        return SendResult(success=True, message_id="voice")

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        return {"name": "Voice Call", "type": "dm"}
