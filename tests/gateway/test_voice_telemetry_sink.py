"""Durable telemetry sink (ENG-555): every voice/telemetry JSON record is
appended to a JSONL file on the agent volume so calls stay auditable
independent of the container's log level, and the log lines themselves are
emitted at WARNING so the default container log level shows them too.

Fail-soft contract: an unwritable sink path logs ONE warning and never
raises into the call path.
"""

from __future__ import annotations

import asyncio
import json
import logging

import pytest

from tests.gateway._voice_module_loader import load_voice_module
from tests.gateway.test_voice_turn_loop import (
    GREETING_REPLY,
    FakeSTT,
    FakeTTSFactory,
    FakeTransport,
    _eventually,
    install_agents,
    say,
)

turn_loop = load_voice_module("turn_loop")


@pytest.fixture()
def sink_path(tmp_path, monkeypatch):
    path = tmp_path / "voice-telemetry.jsonl"
    monkeypatch.setattr(turn_loop, "TELEMETRY_SINK_PATH", str(path))
    monkeypatch.setattr(turn_loop, "_sink_warned", False)
    return path


def _file_records(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


# ---------------------------------------------------------------------------
# Unit: emit_telemetry
# ---------------------------------------------------------------------------


def test_emit_appends_one_json_line_per_record(sink_path):
    rec_a = {"event": "voice_turn", "turn": 1, "status": "ok"}
    rec_b = {"event": "voice_call_summary", "turns": 1}
    turn_loop.emit_telemetry(rec_a)
    turn_loop.emit_telemetry(rec_b)
    assert _file_records(sink_path) == [rec_a, rec_b]


def test_emit_logs_at_warning_level(sink_path, caplog):
    caplog.set_level(logging.WARNING, logger=turn_loop.__name__)
    turn_loop.emit_telemetry({"event": "voice_turn", "turn": 3})
    telemetry = [r for r in caplog.records
                 if r.getMessage().startswith("voice/telemetry {")]
    assert len(telemetry) == 1
    assert telemetry[0].levelno == logging.WARNING
    assert json.loads(
        telemetry[0].getMessage()[len("voice/telemetry "):])["turn"] == 3


def test_unwritable_sink_fails_soft_and_warns_once(
        tmp_path, monkeypatch, caplog):
    missing = tmp_path / "no-such-dir" / "voice-telemetry.jsonl"
    monkeypatch.setattr(turn_loop, "TELEMETRY_SINK_PATH", str(missing))
    monkeypatch.setattr(turn_loop, "_sink_warned", False)
    caplog.set_level(logging.WARNING, logger=turn_loop.__name__)
    turn_loop.emit_telemetry({"event": "voice_turn", "turn": 1})
    turn_loop.emit_telemetry({"event": "voice_turn", "turn": 2})
    # never raised; both records still hit the log stream
    telemetry = [r for r in caplog.records
                 if r.getMessage().startswith("voice/telemetry {")]
    assert len(telemetry) == 2
    # exactly ONE warning about the unwritable sink
    sink_warnings = [r for r in caplog.records
                     if "sink unwritable" in r.getMessage()]
    assert len(sink_warnings) == 1
    assert not missing.exists()


# ---------------------------------------------------------------------------
# Integration: a real loop run lands per-turn + per-call records in the file
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_turn_and_call_summary_records_reach_the_file(
        sink_path, monkeypatch):
    install_agents(monkeypatch, [
        say(GREETING_REPLY), say("It is quarter past three.")])
    stt, factory, transport = FakeSTT(), FakeTTSFactory(), FakeTransport()
    # Drives end-of-turn via Gradium VAD steps (the telemetry record shape is
    # detector-agnostic); pin the legacy detector so the step path is active.
    vloop = turn_loop.VoiceTurnLoop(
        stt, factory, transport, extra={"turn_detector": "gradium"})
    task = asyncio.create_task(vloop.run())
    try:
        await _eventually(lambda: vloop._state == turn_loop.LISTENING
                          and len(vloop._history) == 1)
        stt.push_text("what time is it?")
        stt.push_end_of_turn_step()
        await _eventually(lambda: len(vloop._history) == 3)
    finally:
        await vloop.stop()
        await task
    records = _file_records(sink_path)
    turns = [r for r in records if r["event"] == "voice_turn"
             and r["vad_end_ms"] is not None]
    assert len(turns) == 1
    assert turns[0]["status"] == "ok"
    summaries = [r for r in records if r["event"] == "voice_call_summary"]
    assert len(summaries) == 1
    assert summaries[0]["turns"] == 1
