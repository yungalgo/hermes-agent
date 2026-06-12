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

# Call watchdog (ENG-555 cost leak): an abandoned call — tab closed, or the
# room expired and ejected the agent — used to leave the billable Gradium
# ASR stream running indefinitely (the DTX keep-alive fed it silence
# forever, rotating sessions every 75s). The watchdog tears the call down
# when humans are gone, the call ended remotely, or a hard age cap is hit.
DEFAULT_IDLE_TEARDOWN_S = 60.0   # extra.idle_teardown_s
DEFAULT_MAX_CALL_S = 1800.0      # extra.max_call_s
WATCHDOG_POLL_S = 0.5


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


def _resolve_float_extra(extra: Dict[str, Any], key: str, default: float) -> float:
    raw = extra.get(key)
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        logger.warning("voice: invalid %s=%r; using default %s",
                       key, raw, default)
        return default


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
            await self._end_call("control-leave")

    async def _start_call(self, room_url: str, token: str) -> None:
        (_, daily_transport, gradium_stt, gradium_tts, turn_loop,
         vamp_mod) = _voice_modules()
        async with self._call_lock:
            if self._active_call is not None:
                await self._end_call_locked("replaced-by-new-call")
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
                "task": task, "vamp": vamp_cache,
                "started_at": time.monotonic(), "room_url": room_url}
            self._active_call["watchdog"] = asyncio.create_task(
                self._call_watchdog(transport))
            logger.info("voice: call started in %s", room_url)

    async def _call_watchdog(self, transport) -> None:
        """Tear the call down when it is no longer worth paying for:
          - no human (remote) participant for extra.idle_teardown_s
            (tab closed / never joined),
          - the call ended remotely (room expired, agent ejected, fatal
            client error),
          - call age exceeds extra.max_call_s (hard cost cap).
        Polls every WATCHDOG_POLL_S; "immediate" paths fire on the next
        poll. asr_seconds_est in the teardown summary makes the cost of
        every call auditable."""
        extra = self.config.extra or {}
        idle_teardown_s = _resolve_float_extra(
            extra, "idle_teardown_s", DEFAULT_IDLE_TEARDOWN_S)
        max_call_s = _resolve_float_extra(
            extra, "max_call_s", DEFAULT_MAX_CALL_S)
        idle_since: Optional[float] = None
        while True:
            await asyncio.sleep(WATCHDOG_POLL_S)
            call = self._active_call
            if call is None or call.get("transport") is not transport:
                return
            now = time.monotonic()
            age_s = now - call["started_at"]
            reason = None
            abnormal = transport.abnormal_end
            if abnormal is not None:
                reason = "remote-end"
                logger.warning(
                    "voice: WATCHDOG teardown reason=%s detail=%r age_s=%.0f "
                    "— call ended remotely (ejection/expiry/error)",
                    reason, abnormal, age_s)
            elif age_s >= max_call_s:
                reason = "max-call-duration"
                logger.warning(
                    "voice: WATCHDOG teardown reason=%s age_s=%.0f "
                    "max_call_s=%.0f — hard cost cap hit",
                    reason, age_s, max_call_s)
            elif transport.remote_participant_count == 0:
                if idle_since is None:
                    idle_since = now
                elif now - idle_since >= idle_teardown_s:
                    reason = "no-human-participants"
                    logger.warning(
                        "voice: WATCHDOG teardown reason=%s idle_s=%.0f "
                        "idle_teardown_s=%.0f age_s=%.0f — caller gone "
                        "(tab closed / never joined)",
                        reason, now - idle_since, idle_teardown_s, age_s)
            else:
                idle_since = None
            if reason is not None:
                await self._end_call(reason)
                return

    async def _end_call(self, reason: str) -> None:
        async with self._call_lock:
            await self._end_call_locked(reason)

    async def _end_call_locked(self, reason: str) -> None:
        call = self._active_call
        if call is None:
            return
        self._active_call = None
        # FIRST: kill the billable keep-alive feed. Every await below can
        # take real time, and the keep-alive must not pump paid ASR audio
        # while the call winds down.
        call["transport"].begin_teardown()
        watchdog = call.get("watchdog")
        if watchdog is not None and watchdog is not asyncio.current_task():
            watchdog.cancel()
            try:
                await watchdog
            except (asyncio.CancelledError, Exception):
                pass
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
        summary = {
            "event": "voice_call_teardown",
            "reason": reason,
            "room_url": call.get("room_url"),
            "call_s": round(time.monotonic() - call["started_at"], 1),
            "asr_seconds_est": round(call["stt"].asr_seconds_est, 1),
        }
        # Durable + WARNING-level emit (ENG-555): this is the per-call cost
        # record; it must survive log levels/rotation on the agent volume.
        (_, _, _, _, turn_loop, _) = _voice_modules()
        turn_loop.emit_telemetry(summary)
        logger.info("voice: call ended reason=%s", reason)

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
        await self._end_call("adapter-disconnect")

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
