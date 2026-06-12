"""Tests for the voice platform-plugin: registration shape + requirement gates."""

from __future__ import annotations

from unittest.mock import MagicMock

from gateway.config import PlatformConfig
from tests.gateway._plugin_adapter_loader import load_plugin_adapter

_voice = load_plugin_adapter("voice")


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
