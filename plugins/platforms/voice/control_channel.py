"""Outbound SSE subscription to the control plane (orchestrated mode).

The agent dials OUT to {SECOND_BRAIN_URL}/api/agents/events with its
existing per-agent sb_ key — no inbound port, works behind any NAT
(notes §14 item 2). The connect/stream/backoff lifecycle mirrors the
in-tree Signal adapter's SSE listener (gateway/platforms/signal.py:
_sse_listener — line-buffered parse, `data:`/comment line handling,
exponential backoff with 20% jitter). Uses hermes-core httpx (0.28.x);
no new dependency.

Events (control-plane contract, Phase 3 Lane B):
    {"action": "join_room", "roomUrl": "...", "token": "..."}
    {"action": "leave_room"}
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
from typing import Any, Awaitable, Callable, Dict, Optional

import httpx

logger = logging.getLogger(__name__)

RETRY_INITIAL = 2.0
RETRY_MAX = 60.0


class ControlChannel:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        on_event: Callable[[Dict[str, Any]], Awaitable[None]],
    ):
        if not base_url or not api_key:
            raise ValueError(
                "SECOND_BRAIN_URL and SECOND_BRAIN_MCP_KEY are required "
                "for orchestrated voice mode")
        self._url = base_url.rstrip("/") + "/api/agents/events"
        self._api_key = api_key
        self._on_event = on_event
        self._task: Optional[asyncio.Task] = None
        self._client: Optional[httpx.AsyncClient] = None
        self._running = False

    def start(self) -> None:
        self._running = True
        self._client = httpx.AsyncClient(timeout=30.0)
        self._task = asyncio.create_task(self._listen())

    async def _listen(self) -> None:
        backoff = RETRY_INITIAL
        while self._running:
            try:
                async with self._client.stream(
                    "GET", self._url,
                    headers={
                        "Accept": "text/event-stream",
                        "Authorization": f"Bearer {self._api_key}",
                    },
                    timeout=None,
                ) as response:
                    if response.status_code != 200:
                        logger.error(
                            "voice/control: subscribe failed (HTTP %d)",
                            response.status_code)
                    else:
                        backoff = RETRY_INITIAL
                        logger.info("voice/control: subscribed to %s", self._url)
                        buffer = ""
                        async for chunk in response.aiter_text():
                            if not self._running:
                                break
                            buffer += chunk
                            while "\n" in buffer:
                                line, buffer = buffer.split("\n", 1)
                                line = line.strip()
                                if not line or line.startswith(":"):
                                    continue
                                if line.startswith("data:"):
                                    await self._dispatch(line[5:].strip())
            except asyncio.CancelledError:
                return
            except Exception as e:
                if self._running:
                    logger.warning(
                        "voice/control: stream error: %s (retry in %.0fs)",
                        e, backoff)
            if self._running:
                # 20% jitter against thundering herd (signal.py pattern).
                await asyncio.sleep(backoff + backoff * 0.2 * random.random())
                backoff = min(backoff * 2, RETRY_MAX)

    async def _dispatch(self, data: str) -> None:
        if not data:
            return
        try:
            event = json.loads(data)
        except json.JSONDecodeError:
            logger.debug("voice/control: non-JSON data line ignored")
            return
        if not isinstance(event, dict) or "action" not in event:
            return
        try:
            await self._on_event(event)
        except Exception:
            logger.exception("voice/control: event handler failed")

    async def stop(self) -> None:
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
        if self._client is not None:
            await self._client.aclose()
            self._client = None
