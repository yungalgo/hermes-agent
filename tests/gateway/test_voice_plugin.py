"""Tests for the voice platform-plugin: registration shape, requirement
gates, and the VoiceAdapter connect/disconnect lifecycle in both modes."""

from __future__ import annotations

import asyncio
import json
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gateway.config import PlatformConfig
from tests.gateway._plugin_adapter_loader import load_plugin_adapter
from tests.gateway._voice_module_loader import load_voice_module

_voice = load_plugin_adapter("voice")
_turn_loop = load_voice_module("turn_loop")


def test_platform_enum_resolves_via_plugin_scan():
    from gateway.config import Platform
    p = Platform("voice")
    assert p.value == "voice"
    assert Platform("voice") is p


def test_check_requirements_false_without_gradium_key(monkeypatch):
    monkeypatch.setattr(_voice, "_daily_available", lambda: True)
    monkeypatch.setattr(_voice, "_websockets_available", lambda: True)
    monkeypatch.delenv("GRADIUM_API_KEY", raising=False)
    assert _voice.check_requirements() is False


def test_check_requirements_true_when_all_present(monkeypatch):
    monkeypatch.setattr(_voice, "_daily_available", lambda: True)
    monkeypatch.setattr(_voice, "_websockets_available", lambda: True)
    monkeypatch.setenv("GRADIUM_API_KEY", "g-test")
    assert _voice.check_requirements() is True


def test_validate_config_standalone_needs_daily_key(monkeypatch):
    cfg = PlatformConfig(enabled=True, extra={"mode": "standalone"})
    monkeypatch.delenv("DAILY_API_KEY", raising=False)
    assert _voice.validate_config(cfg) is False
    monkeypatch.setenv("DAILY_API_KEY", "d-test")
    assert _voice.validate_config(cfg) is True


def test_validate_config_orchestrated_needs_control_plane(monkeypatch):
    cfg = PlatformConfig(enabled=True, extra={"mode": "orchestrated"})
    monkeypatch.delenv("SECOND_BRAIN_URL", raising=False)
    monkeypatch.delenv("SECOND_BRAIN_MCP_KEY", raising=False)
    assert _voice.validate_config(cfg) is False
    monkeypatch.setenv("SECOND_BRAIN_URL", "http://control.test")
    monkeypatch.setenv("SECOND_BRAIN_MCP_KEY", "sb_test")
    assert _voice.validate_config(cfg) is True


def test_register_shape():
    ctx = MagicMock()
    _voice.register(ctx)
    kwargs = ctx.register_platform.call_args.kwargs
    assert kwargs["name"] == "voice"
    assert kwargs["required_env"] == ["GRADIUM_API_KEY"]
    assert "voice call" in kwargs["platform_hint"].lower()


# ---------------------------------------------------------------------------
# Adapter lifecycle (fake sibling modules via _voice_modules monkeypatch)
# ---------------------------------------------------------------------------


class FakeControlChannel:
    def __init__(self, base_url, api_key, on_event):
        self.base_url = base_url
        self.api_key = api_key
        self.on_event = on_event
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True

    async def stop(self):
        self.stopped = True


class FakeSTT:
    def __init__(self, api_key):
        self.api_key = api_key
        self.started = False
        self.stopped = False
        self.asr_seconds_est = 0.0

    async def start(self):
        self.started = True

    async def stop(self):
        self.stopped = True

    async def send_audio(self, pcm):
        self.audio_chunks = getattr(self, "audio_chunks", [])
        self.audio_chunks.append(pcm)


class FakeTransport:
    def __init__(self, loop, on_audio_in):
        self.loop = loop
        self.on_audio_in = on_audio_in
        self.joined = None
        self.left = False
        # Watchdog inputs (real transport: presence events + call state).
        self.remote_participant_count = 0
        self.abnormal_end = None
        self.teardown_begun = False

    async def join(self, room_url, token):
        self.joined = (room_url, token)

    def begin_teardown(self):
        self.teardown_begun = True

    async def leave(self):
        self.left = True

    def clear_output(self):
        pass


class FakeTTSTurn:
    def __init__(self, api_key, voice_id, on_audio):
        self.api_key = api_key
        self.voice_id = voice_id
        self.on_audio = on_audio
        self.opened = False
        self.sent = []

    async def open(self):
        self.opened = True

    async def send_text(self, text):
        self.sent.append(text)


class FakeVampCache:
    def __init__(self, api_key, voice_id, texts=None):
        self.api_key = api_key
        self.voice_id = voice_id
        self.texts = texts
        self.started = False
        self.stopped = False
        self.ready = False

    def start(self):
        self.started = True

    async def stop(self):
        self.stopped = True


class FakeVoiceTurnLoop:
    def __init__(self, stt, tts_factory, transport, *, extra, vamp=None):
        self.stt = stt
        self.tts_factory = tts_factory
        self.transport = transport
        self.extra = extra
        self.vamp = vamp
        self.stopped = False
        self._tts = None
        self._stop_event = asyncio.Event()

    async def run(self):
        await self._stop_event.wait()

    async def stop(self):
        self.stopped = True
        self._stop_event.set()

    async def on_inbound_audio(self, pcm):
        self.inbound_chunks = getattr(self, "inbound_chunks", [])
        self.inbound_chunks.append(pcm)


class FakeModules:
    """Stands in for the six sibling modules returned by _voice_modules."""

    def __init__(self):
        self.control_channels = []
        self.stts = []
        self.transports = []
        self.tts_turns = []
        self.turn_loops = []
        self.vamps = []
        # Cross-object teardown ordering (ENG-555: the billable keep-alive
        # must be stopped BEFORE the slow pipeline winddown).
        self.teardown_events = []

        outer = self

        class _ControlChannel(FakeControlChannel):
            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                outer.control_channels.append(self)

        class _STT(FakeSTT):
            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                outer.stts.append(self)

            async def stop(self):
                await super().stop()
                outer.teardown_events.append("stt.stop")

        class _Transport(FakeTransport):
            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                outer.transports.append(self)

            def begin_teardown(self):
                super().begin_teardown()
                outer.teardown_events.append("transport.begin_teardown")

            async def leave(self):
                await super().leave()
                outer.teardown_events.append("transport.leave")

        class _TTSTurn(FakeTTSTurn):
            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                outer.tts_turns.append(self)

        class _TurnLoop(FakeVoiceTurnLoop):
            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                outer.turn_loops.append(self)

            async def stop(self):
                await super().stop()
                outer.teardown_events.append("loop.stop")

        class _Vamp(FakeVampCache):
            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                outer.vamps.append(self)

        self.modules = (
            SimpleNamespace(ControlChannel=_ControlChannel),
            SimpleNamespace(DailyTransport=_Transport),
            SimpleNamespace(GradiumSTT=_STT),
            SimpleNamespace(GradiumTTSTurn=_TTSTurn),
            SimpleNamespace(VoiceTurnLoop=_TurnLoop,
                            emit_telemetry=_turn_loop.emit_telemetry),
            SimpleNamespace(VampCache=_Vamp),
        )


@pytest.fixture()
def fake_modules(monkeypatch, tmp_path):
    fakes = FakeModules()
    monkeypatch.setattr(_voice, "_voice_modules", lambda: fakes.modules)
    monkeypatch.setenv("GRADIUM_API_KEY", "g-test")
    # Hermetic durable sink (ENG-555): the adapter's teardown telemetry
    # appends here instead of /opt/data on the developer machine.
    fakes.telemetry_path = tmp_path / "voice-telemetry.jsonl"
    monkeypatch.setattr(_turn_loop, "TELEMETRY_SINK_PATH",
                        str(fakes.telemetry_path))
    monkeypatch.setattr(_turn_loop, "_sink_warned", False)
    return fakes


def _orchestrated_adapter(monkeypatch, extra=None):
    monkeypatch.setenv("SECOND_BRAIN_URL", "http://control.test")
    monkeypatch.setenv("SECOND_BRAIN_MCP_KEY", "sb_key_9")
    cfg = PlatformConfig(
        enabled=True, extra={"mode": "orchestrated", **(extra or {})})
    return _voice.VoiceAdapter(cfg)


@pytest.mark.asyncio
async def test_orchestrated_connect_subscribes_and_join_room_starts_call(
        fake_modules, monkeypatch):
    adapter = _orchestrated_adapter(monkeypatch, extra={"voice_id": "vox-1"})
    assert await adapter.connect() is True
    # control channel subscribed with the control-plane env creds
    assert len(fake_modules.control_channels) == 1
    control = fake_modules.control_channels[0]
    assert control.started is True
    assert control.base_url == "http://control.test"
    assert control.api_key == "sb_key_9"
    # no call yet — orchestrated mode waits for commands
    assert adapter._active_call is None

    await control.on_event(
        {"action": "join_room", "roomUrl": "https://x.daily.co/r1", "token": "t1"})
    assert adapter._active_call is not None
    assert fake_modules.stts[0].started is True
    assert fake_modules.stts[0].api_key == "g-test"
    assert fake_modules.transports[0].joined == ("https://x.daily.co/r1", "t1")
    # the transport fans caller audio out to the STT AND the turn loop's
    # local energy barge-in
    await fake_modules.transports[0].on_audio_in(b"\x01\x02")
    assert getattr(fake_modules.stts[0], "audio_chunks", []) == [b"\x01\x02"]
    assert getattr(fake_modules.turn_loops[0], "inbound_chunks", []) == [b"\x01\x02"]
    assert fake_modules.turn_loops[0].extra["mode"] == "orchestrated"

    # the tts_factory wired into the loop opens a per-turn Gradium socket
    turn = await fake_modules.turn_loops[0].tts_factory(lambda pcm: None)
    assert turn.opened is True
    assert turn.voice_id == "vox-1"
    assert turn.api_key == "g-test"

    await adapter.disconnect()


@pytest.mark.asyncio
async def test_second_join_room_ends_first_call(fake_modules, monkeypatch):
    adapter = _orchestrated_adapter(monkeypatch)
    await adapter.connect()
    control = fake_modules.control_channels[0]
    await control.on_event(
        {"action": "join_room", "roomUrl": "https://x.daily.co/r1", "token": "t1"})
    first_call = adapter._active_call
    await control.on_event(
        {"action": "join_room", "roomUrl": "https://x.daily.co/r2", "token": "t2"})
    # first call fully torn down
    assert fake_modules.turn_loops[0].stopped is True
    assert fake_modules.stts[0].stopped is True
    assert fake_modules.transports[0].left is True
    # second call live in the new room
    assert adapter._active_call is not None
    assert adapter._active_call is not first_call
    assert fake_modules.transports[1].joined == ("https://x.daily.co/r2", "t2")
    await adapter.disconnect()


@pytest.mark.asyncio
async def test_leave_room_tears_down_call(fake_modules, monkeypatch):
    adapter = _orchestrated_adapter(monkeypatch)
    await adapter.connect()
    control = fake_modules.control_channels[0]
    await control.on_event(
        {"action": "join_room", "roomUrl": "https://x.daily.co/r1", "token": "t1"})
    await control.on_event({"action": "leave_room"})
    assert adapter._active_call is None
    assert fake_modules.turn_loops[0].stopped is True
    assert fake_modules.stts[0].stopped is True
    assert fake_modules.transports[0].left is True
    # leave_room with no live call is a harmless no-op
    await control.on_event({"action": "leave_room"})
    await adapter.disconnect()


@pytest.mark.asyncio
async def test_disconnect_stops_control_channel_and_call(fake_modules, monkeypatch):
    adapter = _orchestrated_adapter(monkeypatch)
    await adapter.connect()
    control = fake_modules.control_channels[0]
    await control.on_event(
        {"action": "join_room", "roomUrl": "https://x.daily.co/r1", "token": "t1"})
    await adapter.disconnect()
    assert control.stopped is True
    assert adapter._active_call is None
    assert fake_modules.turn_loops[0].stopped is True
    assert fake_modules.stts[0].stopped is True
    assert fake_modules.transports[0].left is True


@pytest.mark.asyncio
async def test_send_without_call_fails_honestly(fake_modules, monkeypatch):
    adapter = _orchestrated_adapter(monkeypatch)
    await adapter.connect()
    result = await adapter.send("voice", "hello")
    assert result.success is False
    assert "no active voice call" in result.error
    await adapter.disconnect()


@pytest.mark.asyncio
async def test_send_speaks_when_agent_mid_utterance(fake_modules, monkeypatch):
    adapter = _orchestrated_adapter(monkeypatch)
    await adapter.connect()
    control = fake_modules.control_channels[0]
    await control.on_event(
        {"action": "join_room", "roomUrl": "https://x.daily.co/r1", "token": "t1"})
    vloop = fake_modules.turn_loops[0]
    # no in-flight TTS turn -> honest failure, no hidden queue
    result = await adapter.send("voice", "psst")
    assert result.success is False
    # mid-utterance -> text rides the live turn socket
    tts = FakeTTSTurn("g-test", "", lambda pcm: None)
    vloop._tts = tts
    result = await adapter.send("voice", "breaking news")
    assert result.success is True
    assert tts.sent == ["breaking news"]
    await adapter.disconnect()


@pytest.mark.asyncio
async def test_standalone_connect_creates_room_and_joins(fake_modules, monkeypatch):
    async def fake_create(self):
        return "https://x.daily.co/solo", "tok-solo"

    monkeypatch.setattr(
        _voice.VoiceAdapter, "_create_standalone_room", fake_create)
    cfg = PlatformConfig(enabled=True, extra={"mode": "standalone"})
    adapter = _voice.VoiceAdapter(cfg)
    assert await adapter.connect() is True
    assert adapter._active_call is not None
    assert fake_modules.transports[0].joined == ("https://x.daily.co/solo", "tok-solo")
    assert fake_modules.stts[0].started is True
    await adapter.disconnect()
    assert adapter._active_call is None


@pytest.mark.asyncio
async def test_create_standalone_room_daily_rest_shapes(monkeypatch):
    posts = []

    class FakeResponse:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return self

        def json(self):
            return self._payload

    class FakeAsyncClient:
        def __init__(self, timeout=None):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def post(self, url, headers=None, json=None):
            posts.append({"url": url, "headers": headers, "json": json})
            if url.endswith("/rooms"):
                return FakeResponse(
                    {"url": "https://x.daily.co/r9", "name": "r9"})
            return FakeResponse({"token": "tok-agent"})

    fake_httpx = types.ModuleType("httpx")
    fake_httpx.AsyncClient = FakeAsyncClient
    monkeypatch.setitem(sys.modules, "httpx", fake_httpx)
    monkeypatch.setenv("DAILY_API_KEY", "dk-test")
    monkeypatch.setenv("GRADIUM_API_KEY", "g-test")

    cfg = PlatformConfig(enabled=True, extra={"mode": "standalone"})
    adapter = _voice.VoiceAdapter(cfg)
    room_url, token = await adapter._create_standalone_room()
    assert room_url == "https://x.daily.co/r9"
    assert token == "tok-agent"

    assert posts[0]["url"] == "https://api.daily.co/v1/rooms"
    assert posts[0]["headers"]["Authorization"] == "Bearer dk-test"
    assert posts[0]["json"]["privacy"] == "private"
    assert isinstance(posts[0]["json"]["properties"]["exp"], int)

    assert posts[1]["url"] == "https://api.daily.co/v1/meeting-tokens"
    assert posts[1]["json"]["properties"]["room_name"] == "r9"
    assert posts[1]["json"]["properties"]["is_owner"] is False


# ---------------------------------------------------------------------------
# Vamp lifecycle (adapter-level)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_call_start_creates_and_starts_vamp_cache(
        fake_modules, monkeypatch):
    adapter = _orchestrated_adapter(
        monkeypatch, extra={"voice_id": "vox-2",
                            "vamp_texts": ["Right.", "One sec."]})
    await adapter.connect()
    await fake_modules.control_channels[0].on_event(
        {"action": "join_room", "roomUrl": "https://x.daily.co/r1",
         "token": "t1"})
    assert len(fake_modules.vamps) == 1
    vamp = fake_modules.vamps[0]
    # background synthesis kicked off, in the agent's voice, custom texts
    assert vamp.started is True
    assert vamp.api_key == "g-test"
    assert vamp.voice_id == "vox-2"
    assert vamp.texts == ["Right.", "One sec."]
    # the cache is handed to the turn loop
    assert fake_modules.turn_loops[0].vamp is vamp
    await adapter.disconnect()
    assert vamp.stopped is True


@pytest.mark.asyncio
async def test_vamp_disabled_by_config(fake_modules, monkeypatch):
    adapter = _orchestrated_adapter(monkeypatch,
                                    extra={"vamp_enabled": False})
    await adapter.connect()
    await fake_modules.control_channels[0].on_event(
        {"action": "join_room", "roomUrl": "https://x.daily.co/r1",
         "token": "t1"})
    assert fake_modules.vamps == []
    assert fake_modules.turn_loops[0].vamp is None
    await adapter.disconnect()


# ---------------------------------------------------------------------------
# Call watchdog (ENG-555: abandoned calls must never keep billing ASR)
# ---------------------------------------------------------------------------


async def _eventually(cond, timeout=2.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while not cond():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("condition not met within %.1fs" % timeout)
        await asyncio.sleep(0.01)


def _teardown_summary(caplog):
    """Parse the voice_call_teardown telemetry JSON line from the logs."""
    for record in caplog.records:
        msg = record.getMessage()
        if "voice_call_teardown" in msg:
            return json.loads(msg.split("voice/telemetry ", 1)[1])
    return None


async def _joined_adapter(fake_modules, monkeypatch, extra):
    monkeypatch.setattr(_voice, "WATCHDOG_POLL_S", 0.01)
    adapter = _orchestrated_adapter(monkeypatch, extra=extra)
    await adapter.connect()
    await fake_modules.control_channels[0].on_event(
        {"action": "join_room", "roomUrl": "https://x.daily.co/r1",
         "token": "t1"})
    return adapter


def _call_torn_down(adapter, fake_modules):
    return (adapter._active_call is None
            and fake_modules.turn_loops[0].stopped
            and fake_modules.stts[0].stopped
            and fake_modules.transports[0].left)


@pytest.mark.asyncio
async def test_watchdog_tears_down_after_humans_leave(
        fake_modules, monkeypatch, caplog):
    """Tab-abandon: the caller's participant drops to 0 and never comes
    back -> full teardown once idle_teardown_s elapses."""
    caplog.set_level("INFO")
    adapter = await _joined_adapter(
        fake_modules, monkeypatch,
        extra={"idle_teardown_s": 0.2, "max_call_s": 60})
    transport = fake_modules.transports[0]
    transport.remote_participant_count = 1
    # A present human holds the call open well past the idle window.
    await asyncio.sleep(0.4)
    assert adapter._active_call is not None
    # Tab closed: participants drop to 0 and stay there.
    transport.remote_participant_count = 0
    await _eventually(lambda: _call_torn_down(adapter, fake_modules))
    assert transport.teardown_begun is True
    summary = _teardown_summary(caplog)
    assert summary["reason"] == "no-human-participants"
    await adapter.disconnect()


@pytest.mark.asyncio
async def test_watchdog_idle_clock_resets_when_human_returns(
        fake_modules, monkeypatch):
    """A brief participant gap (reconnect blip) must not end the call."""
    adapter = await _joined_adapter(
        fake_modules, monkeypatch,
        extra={"idle_teardown_s": 0.3, "max_call_s": 60})
    transport = fake_modules.transports[0]
    transport.remote_participant_count = 1
    await asyncio.sleep(0.05)
    transport.remote_participant_count = 0      # blip starts
    await asyncio.sleep(0.1)                    # under the idle window
    transport.remote_participant_count = 1      # human came back
    await asyncio.sleep(0.4)                    # idle clock must have reset
    assert adapter._active_call is not None
    await adapter.disconnect()


@pytest.mark.asyncio
async def test_watchdog_remote_end_tears_down_immediately(
        fake_modules, monkeypatch, caplog):
    """Ejection (room expiry / kicked): the transport flags abnormal_end
    and the watchdog ends the call on the next poll — no idle wait."""
    caplog.set_level("INFO")
    adapter = await _joined_adapter(
        fake_modules, monkeypatch,
        extra={"idle_teardown_s": 30, "max_call_s": 60})
    transport = fake_modules.transports[0]
    transport.remote_participant_count = 1
    transport.abnormal_end = "left"
    await _eventually(lambda: _call_torn_down(adapter, fake_modules),
                      timeout=1.0)   # far below idle_teardown_s
    summary = _teardown_summary(caplog)
    assert summary["reason"] == "remote-end"
    await adapter.disconnect()


@pytest.mark.asyncio
async def test_watchdog_enforces_max_call_duration(
        fake_modules, monkeypatch, caplog):
    """The hard age cap fires even with a human still on the call."""
    caplog.set_level("INFO")
    adapter = await _joined_adapter(
        fake_modules, monkeypatch,
        extra={"idle_teardown_s": 30, "max_call_s": 0.1})
    fake_modules.transports[0].remote_participant_count = 1
    await _eventually(lambda: _call_torn_down(adapter, fake_modules))
    summary = _teardown_summary(caplog)
    assert summary["reason"] == "max-call-duration"
    await adapter.disconnect()


@pytest.mark.asyncio
async def test_teardown_summary_reports_asr_cost(
        fake_modules, monkeypatch, caplog):
    """Every teardown emits an auditable per-call summary with the
    estimated billable ASR wall-clock."""
    caplog.set_level("INFO")
    adapter = await _joined_adapter(
        fake_modules, monkeypatch, extra={})
    fake_modules.stts[0].asr_seconds_est = 123.4
    await fake_modules.control_channels[0].on_event({"action": "leave_room"})
    summary = _teardown_summary(caplog)
    assert summary["reason"] == "control-leave"
    assert summary["asr_seconds_est"] == 123.4
    assert summary["room_url"] == "https://x.daily.co/r1"
    assert summary["call_s"] >= 0
    await adapter.disconnect()


@pytest.mark.asyncio
async def test_teardown_summary_appended_to_durable_sink(
        fake_modules, monkeypatch):
    """ENG-555 durable telemetry: the per-call teardown record (the one
    carrying asr_seconds_est) must land in the JSONL sink so the call is
    auditable independent of the container log level."""
    adapter = await _joined_adapter(fake_modules, monkeypatch, extra={})
    fake_modules.stts[0].asr_seconds_est = 42.5
    await fake_modules.control_channels[0].on_event({"action": "leave_room"})
    await adapter.disconnect()
    lines = fake_modules.telemetry_path.read_text(
        encoding="utf-8").splitlines()
    teardowns = [json.loads(line) for line in lines
                 if json.loads(line)["event"] == "voice_call_teardown"]
    assert len(teardowns) == 1
    assert teardowns[0]["reason"] == "control-leave"
    assert teardowns[0]["asr_seconds_est"] == 42.5


@pytest.mark.asyncio
async def test_teardown_stops_keepalive_before_pipeline_winddown(
        fake_modules, monkeypatch):
    """Ordering (ENG-555): begin_teardown (kills the billable keep-alive
    feed) must run FIRST — before the turn loop, STT, and transport leave,
    each of which can await for real time."""
    adapter = await _joined_adapter(fake_modules, monkeypatch, extra={})
    await fake_modules.control_channels[0].on_event({"action": "leave_room"})
    events = fake_modules.teardown_events
    assert events[0] == "transport.begin_teardown"
    assert events.index("loop.stop") < events.index("stt.stop")
    assert events.index("stt.stop") < events.index("transport.leave")
    await adapter.disconnect()
