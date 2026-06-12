"""Tests for the synthetic-call probe's reply-end classifier.

Verify run 2026-06-12: the fixed REPLY_END_SILENCE_S = 4.0 mistook a vamp
clip followed by the (3.9-4.3s) vamp->substantive gap for a completed
reply, so the probe's next utterance collided with the late substantive
audio — the prime suspect for the run's missed vamp and turn-8 false
fire. The wait is now dynamic: vamp-length audio followed by silence is
"still waiting" until substantive audio arrives or a hard timeout passes.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_probe():
    mod_name = "voice_probe_under_test"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    path = _REPO_ROOT / "scripts" / "voice_probe.py"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


probe = _load_probe()

CHUNK_S = probe.CHUNK_S
SPEECH = probe.RMS_SPEECH + 700          # clearly agent speech
QUIET = 50                               # below the speech floor


def _samples(*runs):
    """Build (t, rms) speaker samples from (t_start, duration_s, rms)."""
    out = []
    for t0, dur, rms in runs:
        n = max(1, int(round(dur / CHUNK_S)))
        out.extend((t0 + i * CHUNK_S, rms) for i in range(n))
    out.sort(key=lambda x: x[0])
    return out


# ---------------------------------------------------------------------------
# reply_end_state
# ---------------------------------------------------------------------------


def test_no_speech_is_waiting():
    assert probe.reply_end_state([], 0.0, 5.0) == ("waiting", None)
    silence_only = _samples((0.0, 2.0, QUIET))
    assert probe.reply_end_state(silence_only, 0.0, 5.0) == ("waiting", None)


def test_substantive_reply_ends_after_end_silence():
    # 4s of speech = clearly substantive
    samples = _samples((1.0, 4.0, SPEECH))
    last = samples[-1][0]
    state, t_last = probe.reply_end_state(samples, 0.0, last + 1.0)
    assert state == "waiting"
    state, t_last = probe.reply_end_state(
        samples, 0.0, last + probe.REPLY_END_SILENCE_S + 0.1)
    assert state == "done"
    assert t_last == last


def test_vamp_clip_plus_old_bug_gap_is_not_done():
    """The exact verify-run failure: a ~1s ack clip followed by a 4s gap
    must NOT read as a completed reply."""
    samples = _samples((1.0, 1.0, SPEECH))           # vamp-length clip
    last = samples[-1][0]
    state, _ = probe.reply_end_state(samples, 0.0, last + 4.0)
    assert state == "vamp-gap"                       # old code said done
    # substantive audio finally arrives -> normal end-silence rule applies
    samples += _samples((last + 4.2, 3.0, SPEECH))
    new_last = samples[-1][0]
    state, _ = probe.reply_end_state(samples, 0.0, new_last + 1.0)
    assert state == "waiting"
    state, t_last = probe.reply_end_state(
        samples, 0.0, new_last + probe.REPLY_END_SILENCE_S + 0.1)
    assert state == "done"
    assert t_last == new_last


def test_vamp_only_reply_done_after_hard_timeout():
    """A short clip with NOTHING behind it eventually counts as the whole
    reply (hard timeout) — the probe must not hang forever."""
    samples = _samples((1.0, 1.0, SPEECH))
    last = samples[-1][0]
    state, _ = probe.reply_end_state(
        samples, 0.0, last + probe.VAMP_GAP_TIMEOUT_S - 0.5)
    assert state == "vamp-gap"
    state, t_last = probe.reply_end_state(
        samples, 0.0, last + probe.VAMP_GAP_TIMEOUT_S + 0.1)
    assert state == "done"
    assert t_last == last


def test_speech_before_t_start_is_ignored():
    samples = _samples((0.0, 5.0, SPEECH))           # the PREVIOUS reply
    assert probe.reply_end_state(samples, 10.0, 20.0) == ("waiting", None)


# ---------------------------------------------------------------------------
# first_substantive_start (perceived vs substantive split)
# ---------------------------------------------------------------------------


def test_substantive_start_after_vamp_clip():
    samples = _samples((1.0, 1.0, SPEECH),           # vamp clip
                       (5.0, 3.0, SPEECH))           # substantive reply
    assert probe.first_substantive_start(samples, 0.0) == 5.0


def test_substantive_start_without_vamp():
    samples = _samples((2.0, 4.0, SPEECH))           # one long reply
    assert probe.first_substantive_start(samples, 0.0) == 2.0


def test_substantive_start_vamp_only_is_none():
    samples = _samples((1.0, 1.0, SPEECH))
    assert probe.first_substantive_start(samples, 0.0) is None
    assert probe.first_substantive_start([], 0.0) is None
