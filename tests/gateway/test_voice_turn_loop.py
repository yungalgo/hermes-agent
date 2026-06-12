"""Tests for the voice turn orchestrator: greeting-first, VAD end-of-turn,
sentence buffering, filler-on-tool, barge-in, and history bookkeeping.

The agent factory is monkeypatched with fakes whose ``run_conversation``
executes in the real default executor thread (via the loop's own
``run_in_executor``), so the ``loop.call_soon_threadsafe`` delta marshal
is exercised across an actual thread boundary.
"""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from tests.gateway._voice_module_loader import load_voice_module

turn_loop = load_voice_module("turn_loop")


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeSTT:
    """Scripted STT event source. flush() optionally auto-acks by pushing a
    ``flushed`` event back onto its own stream (like the live server)."""

    def __init__(self, *, flush_returns_none=False, auto_ack=True):
        self.queue: "asyncio.Queue[dict]" = asyncio.Queue()
        self.flush_calls = 0
        self.rotate_msgs = []
        self.flush_returns_none = flush_returns_none
        self.auto_ack = auto_ack

    async def events(self):
        while True:
            msg = await self.queue.get()
            yield msg

    async def maybe_rotate(self, msg):
        self.rotate_msgs.append(msg)

    async def flush(self):
        self.flush_calls += 1
        if self.flush_returns_none:
            return None
        if self.auto_ack:
            self.queue.put_nowait(
                {"type": "flushed", "flush_id": self.flush_calls})
        return self.flush_calls

    def push(self, msg):
        self.queue.put_nowait(msg)

    def push_text(self, text):
        self.push({"type": "text", "text": text, "start_s": 0.0})

    def push_end_of_turn_step(self, prob=0.9):
        # Live shape: FOUR horizons (notes §15) — looked up by value.
        self.push({"type": "step", "total_duration_s": 1.0, "vad": [
            {"horizon_s": 0.5, "inactivity_prob": prob},
            {"horizon_s": 1.0, "inactivity_prob": prob},
            {"horizon_s": 2.0, "inactivity_prob": prob},
            {"horizon_s": 3.0, "inactivity_prob": prob},
        ]})


class FakeTTS:
    """Mirrors the GradiumTTSTurn surface the loop touches."""

    def __init__(self, on_audio):
        self.on_audio = on_audio
        self.sent = []
        self.ended = False
        self.aborted = False

    async def send_text(self, text):
        if self.aborted or not text:
            return
        self.sent.append(text)

    async def send_filler(self, text):
        await self.send_text(text + " <flush>")

    async def end(self):
        self.ended = True

    async def abort(self):
        self.aborted = True


class FakeTTSFactory:
    def __init__(self):
        self.instances = []

    async def __call__(self, on_audio):
        tts = FakeTTS(on_audio)
        self.instances.append(tts)
        return tts


class FakeTransport:
    def __init__(self):
        self.chunks = []
        self.clear_calls = 0

    async def send_audio(self, pcm):
        self.chunks.append(pcm)

    def clear_output(self):
        self.clear_calls += 1


class FakeAgent:
    """run_conversation runs in the executor thread; *behavior* drives the
    delta / tool-progress callbacks from there."""

    def __init__(self, behavior, delta_cb, tool_cb):
        self._behavior = behavior
        self.delta_cb = delta_cb
        self.tool_cb = tool_cb
        self.interrupt_event = threading.Event()
        self.interrupt_reason = None
        self.run_kwargs = None

    def run_conversation(self, user_message, conversation_history, task_id):
        self.run_kwargs = {
            "user_message": user_message,
            "conversation_history": conversation_history,
            "task_id": task_id,
        }
        return self._behavior(self)

    def interrupt(self, reason):
        self.interrupt_reason = reason
        self.interrupt_event.set()


def say(text):
    """Behavior: stream *text* as one delta, finish normally."""
    def _behavior(agent):
        agent.delta_cb(text)
        return {"final_response": text}
    return _behavior


def speak_then_block(text, timeout=10.0):
    """Behavior: stream a delta, then hang until interrupt() (barge-in)."""
    def _behavior(agent):
        agent.delta_cb(text)
        agent.interrupt_event.wait(timeout)
        return {"final_response": text}
    return _behavior


def install_agents(monkeypatch, behaviors):
    """Replace _create_voice_agent; each turn consumes the next behavior."""
    agents = []
    behaviors = list(behaviors)

    def _fake_create(session_id, stream_delta_callback, tool_progress_callback,
                     *, extra=None):
        behavior = behaviors.pop(0)
        agent = FakeAgent(behavior, stream_delta_callback,
                          tool_progress_callback)
        agents.append(agent)
        return agent

    monkeypatch.setattr(turn_loop, "_create_voice_agent", _fake_create)
    return agents


async def _eventually(cond, timeout=5.0, msg="condition not met"):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return
        await asyncio.sleep(0.01)
    raise AssertionError(msg)


def _make_loop(stt, factory, transport, extra=None):
    return turn_loop.VoiceTurnLoop(
        stt, factory, transport, extra=extra or {})


# ---------------------------------------------------------------------------
# max_turns config resolution (pure)
# ---------------------------------------------------------------------------


def test_max_turns_from_extra(monkeypatch):
    monkeypatch.delenv("HERMES_MAX_ITERATIONS", raising=False)
    assert turn_loop._resolve_max_iterations({"max_turns": 5}) == 5
    assert turn_loop._resolve_max_iterations({"max_turns": "7"}) == 7
    assert turn_loop._resolve_max_iterations({}) == 90
    assert turn_loop._resolve_max_iterations({"max_turns": "bogus"}) == 90
    monkeypatch.setenv("HERMES_MAX_ITERATIONS", "12")
    assert turn_loop._resolve_max_iterations({}) == 12


# ---------------------------------------------------------------------------
# Turn-loop behavior
# ---------------------------------------------------------------------------


GREETING_REPLY = "Hello there, this is your agent speaking. How can I help?"


@pytest.mark.asyncio
async def test_greeting_turn_runs_first_and_speaks(monkeypatch):
    agents = install_agents(monkeypatch, [say(GREETING_REPLY)])
    stt, factory, transport = FakeSTT(), FakeTTSFactory(), FakeTransport()
    vloop = _make_loop(stt, factory, transport)
    task = asyncio.create_task(vloop.run())
    try:
        await _eventually(lambda: factory.instances and factory.instances[0].ended,
                          msg="greeting turn never completed")
        # Agent spoke first — without any STT input at all.
        assert stt.flush_calls == 0
        assert len(agents) == 1
        assert agents[0].run_kwargs["user_message"] == \
            turn_loop.DEFAULT_GREETING_PROMPT
        sent = factory.instances[0].sent
        assert sent and "".join(sent) == GREETING_REPLY
        # Greeting is not a user message: history has only the assistant turn.
        assert vloop._history == [
            {"role": "assistant", "content": GREETING_REPLY}]
    finally:
        await vloop.stop()
        await task


@pytest.mark.asyncio
async def test_custom_greeting_prompt_used(monkeypatch):
    agents = install_agents(monkeypatch, [say(GREETING_REPLY)])
    stt, factory, transport = FakeSTT(), FakeTTSFactory(), FakeTransport()
    vloop = _make_loop(stt, factory, transport,
                       extra={"greeting_prompt": "(Say hi in Latin.)"})
    task = asyncio.create_task(vloop.run())
    try:
        await _eventually(lambda: agents and agents[0].run_kwargs)
        assert agents[0].run_kwargs["user_message"] == "(Say hi in Latin.)"
    finally:
        await vloop.stop()
        await task


@pytest.mark.asyncio
async def test_vad_end_of_turn_triggers_one_agent_turn(monkeypatch):
    user_reply = "The weather is sunny today, with a gentle breeze."
    agents = install_agents(monkeypatch, [say(GREETING_REPLY), say(user_reply)])
    stt, factory, transport = FakeSTT(), FakeTTSFactory(), FakeTransport()
    vloop = _make_loop(stt, factory, transport)
    task = asyncio.create_task(vloop.run())
    try:
        await _eventually(lambda: vloop._state == turn_loop.LISTENING
                          and len(vloop._history) == 1)
        stt.push_text("What is")
        stt.push_text("the weather?")
        stt.push_end_of_turn_step()
        # extra steps must NOT spawn extra turns while finalize is pending
        stt.push_end_of_turn_step()
        await _eventually(lambda: len(agents) == 2 and agents[1].run_kwargs)
        assert agents[1].run_kwargs["user_message"] == "What is the weather?"
        assert stt.flush_calls == 1
        # flushed ack consumed (no flush-timeout stall): the second turn's
        # TTS finished well within the test timeout
        await _eventually(lambda: len(factory.instances) == 2
                          and factory.instances[1].ended)
        # history: greeting assistant + user + assistant
        assert vloop._history == [
            {"role": "assistant", "content": GREETING_REPLY},
            {"role": "user", "content": "What is the weather?"},
            {"role": "assistant", "content": user_reply},
        ]
        # the second turn saw the greeting in its conversation history
        assert agents[1].run_kwargs["conversation_history"] == [
            {"role": "assistant", "content": GREETING_REPLY}]
        # semantic VAD steps were offered to the rotation check
        assert len(stt.rotate_msgs) == 2
        assert len(agents) == 2
    finally:
        await vloop.stop()
        await task


@pytest.mark.asyncio
async def test_flush_none_skips_flushed_wait(monkeypatch):
    """GradiumSTT.flush() -> None means no live session: the loop must NOT
    burn the flushed-wait timeout (notes §15 addendum)."""
    agents = install_agents(
        monkeypatch, [say(GREETING_REPLY), say("Right away, doing it now.")])
    stt = FakeSTT(flush_returns_none=True)
    factory, transport = FakeTTSFactory(), FakeTransport()
    vloop = _make_loop(stt, factory, transport)
    task = asyncio.create_task(vloop.run())
    try:
        await _eventually(lambda: vloop._state == turn_loop.LISTENING
                          and len(vloop._history) == 1)
        stt.push_text("Do the thing please")
        start = time.monotonic()
        stt.push_end_of_turn_step()
        await _eventually(lambda: len(agents) == 2)
        # Well under FLUSHED_WAIT_TIMEOUT_S (2.0s): the wait was skipped.
        assert time.monotonic() - start < 1.0
        assert agents[1].run_kwargs["user_message"] == "Do the thing please"
    finally:
        await vloop.stop()
        await task


@pytest.mark.asyncio
async def test_short_fragments_merged_into_next_sentence(monkeypatch):
    def behavior(agent):
        # "Yes. " alone is < 20 chars — must be merged forward, not spoken.
        agent.delta_cb("Yes. ")
        agent.delta_cb("I can absolutely help with that request.")
        return {"final_response": "x"}

    install_agents(monkeypatch, [behavior])
    stt, factory, transport = FakeSTT(), FakeTTSFactory(), FakeTransport()
    vloop = _make_loop(stt, factory, transport)
    task = asyncio.create_task(vloop.run())
    try:
        await _eventually(lambda: factory.instances and factory.instances[0].ended)
        sent = factory.instances[0].sent
        # The short fragment was never spoken alone (tts_tool.py:2637-2640
        # mirror): it rides along with the rest at the final flush.
        assert sent == ["Yes. I can absolutely help with that request."]
        assert all(len(s.strip()) >= turn_loop.MIN_SENTENCE_LEN for s in sent)
    finally:
        await vloop.stop()
        await task


@pytest.mark.asyncio
async def test_long_buffer_flushes_without_boundary(monkeypatch):
    blocker = threading.Event()
    no_boundary = "word " * 30  # 150 chars, no sentence punctuation

    def behavior(agent):
        agent.delta_cb(no_boundary)
        blocker.wait(10.0)
        return {"final_response": "x"}

    install_agents(monkeypatch, [behavior])
    stt, factory, transport = FakeSTT(), FakeTTSFactory(), FakeTransport()
    vloop = _make_loop(stt, factory, transport)
    task = asyncio.create_task(vloop.run())
    try:
        # The 0.5s queue-timeout path must flush the >100-char buffer even
        # though no boundary ever arrives (tts_tool.py:2604-2607 mirror).
        await _eventually(lambda: factory.instances
                          and factory.instances[0].sent == [no_boundary],
                          timeout=5.0)
    finally:
        blocker.set()
        await vloop.stop()
        await task


@pytest.mark.asyncio
async def test_tool_started_speaks_filler_once_with_flush(monkeypatch):
    def behavior(agent):
        agent.tool_cb("tool.started", tool_name="create_note")
        agent.tool_cb("tool.completed", tool_name="create_note")
        agent.tool_cb("tool.started", tool_name="search_notes")
        agent.delta_cb("All done, your note has been created successfully. ")
        return {"final_response": "x"}

    install_agents(monkeypatch, [behavior])
    stt, factory, transport = FakeSTT(), FakeTTSFactory(), FakeTransport()
    vloop = _make_loop(stt, factory, transport,
                       extra={"filler_text": "Hang on."})
    task = asyncio.create_task(vloop.run())
    try:
        await _eventually(lambda: factory.instances and factory.instances[0].ended)
        sent = factory.instances[0].sent
        fillers = [s for s in sent if s.endswith("<flush>")]
        assert fillers == ["Hang on. <flush>"]
        assert sent[0] == "Hang on. <flush>"  # spoken before the reply text
    finally:
        await vloop.stop()
        await task


@pytest.mark.asyncio
async def test_barge_in_interrupts_agent_aborts_tts_clears_output(monkeypatch):
    agents = install_agents(monkeypatch, [
        speak_then_block("This is a very long story that goes on and on. "
                         "It keeps going for a while longer than anyone wants. "),
    ])
    stt, factory, transport = FakeSTT(), FakeTTSFactory(), FakeTransport()
    vloop = _make_loop(stt, factory, transport)
    task = asyncio.create_task(vloop.run())
    try:
        await _eventually(lambda: vloop._state == turn_loop.SPEAKING
                          and factory.instances
                          and factory.instances[0].sent)
        # caller speech while the agent is speaking -> barge-in
        stt.push_text("stop right there")
        await _eventually(lambda: vloop._state == turn_loop.LISTENING,
                          msg="barge-in never returned to LISTENING")
        assert agents[0].interrupt_reason == "user barge-in (voice)"
        assert factory.instances[0].aborted is True
        assert factory.instances[0].ended is False
        assert transport.clear_calls >= 1
        # interrupted turn leaves NO trace in history
        assert vloop._history == []
        # the barged-in utterance is kept for the next turn
        assert vloop._pending_text == ["stop right there"]
    finally:
        await vloop.stop()
        await task


@pytest.mark.asyncio
async def test_barge_in_during_thinking_before_tts(monkeypatch):
    started = threading.Event()

    def thinking_behavior(agent):
        started.set()
        agent.interrupt_event.wait(10.0)
        return {"final_response": "too late"}

    agents = install_agents(monkeypatch, [thinking_behavior])
    stt, factory, transport = FakeSTT(), FakeTTSFactory(), FakeTransport()
    vloop = _make_loop(stt, factory, transport)
    task = asyncio.create_task(vloop.run())
    try:
        await _eventually(lambda: started.is_set())
        stt.push_text("hold on")
        await _eventually(lambda: agents[0].interrupt_event.is_set())
        await _eventually(lambda: vloop._state == turn_loop.LISTENING)
        assert vloop._history == []
    finally:
        await vloop.stop()
        await task


@pytest.mark.asyncio
async def test_completed_turn_after_barge_in_still_works(monkeypatch):
    """After a barge-in, the next end-of-turn still produces a clean turn."""
    agents = install_agents(monkeypatch, [
        speak_then_block("A very long opening sentence that will be cut off. "),
        say("Second answer, delivered without any interruption at all."),
    ])
    stt, factory, transport = FakeSTT(), FakeTTSFactory(), FakeTransport()
    vloop = _make_loop(stt, factory, transport)
    task = asyncio.create_task(vloop.run())
    try:
        await _eventually(lambda: vloop._state == turn_loop.SPEAKING)
        stt.push_text("actually, never mind that")
        await _eventually(lambda: vloop._state == turn_loop.LISTENING)
        stt.push_end_of_turn_step()
        await _eventually(lambda: len(agents) == 2 and agents[1].run_kwargs)
        assert agents[1].run_kwargs["user_message"] == "actually, never mind that"
        await _eventually(lambda: len(vloop._history) == 2)
        assert vloop._history[0]["role"] == "user"
        assert vloop._history[1]["role"] == "assistant"
    finally:
        await vloop.stop()
        await task


@pytest.mark.asyncio
async def test_stop_cancels_inflight_turn(monkeypatch):
    agents = install_agents(monkeypatch, [
        speak_then_block("An answer that never gets to finish speaking. ")])
    stt, factory, transport = FakeSTT(), FakeTTSFactory(), FakeTransport()
    vloop = _make_loop(stt, factory, transport)
    task = asyncio.create_task(vloop.run())
    await _eventually(lambda: vloop._state == turn_loop.SPEAKING)
    await vloop.stop()
    await asyncio.wait_for(task, timeout=5.0)
    assert agents[0].interrupt_event.is_set()
    assert factory.instances[0].aborted is True


@pytest.mark.asyncio
async def test_stale_pre_rotation_ack_does_not_satisfy_new_flush_wait(monkeypatch):
    """A ``flushed`` ack carrying an OLD flush id (e.g. replayed by the
    pre-rotation socket — both sessions share one events queue) must not
    satisfy a wait keyed on a NEWER id: the loop waits out the timeout
    instead of being falsely released."""
    monkeypatch.setattr(turn_loop, "FLUSHED_WAIT_TIMEOUT_S", 0.3)

    class StaleAckSTT(FakeSTT):
        async def flush(self):
            self.flush_calls += 1
            # First flush acked correctly; every later flush only ever
            # sees a REPLAY of the old ack (id 1).
            self.queue.put_nowait({"type": "flushed", "flush_id": 1})
            return self.flush_calls

    agents = install_agents(monkeypatch, [
        say(GREETING_REPLY),
        say("First answer, complete and properly delivered."),
        say("Second answer, complete and properly delivered."),
    ])
    stt = StaleAckSTT()
    factory, transport = FakeTTSFactory(), FakeTransport()
    vloop = _make_loop(stt, factory, transport)
    task = asyncio.create_task(vloop.run())
    try:
        await _eventually(lambda: vloop._state == turn_loop.LISTENING
                          and len(vloop._history) == 1)
        stt.push_text("first question for you")
        stt.push_end_of_turn_step()
        await _eventually(lambda: len(agents) == 2 and len(vloop._history) == 3)
        # Second user turn: flush id 2, but only the stale ack (id 1) arrives.
        stt.push_text("second question for you")
        start = time.monotonic()
        stt.push_end_of_turn_step()
        await _eventually(lambda: len(agents) == 3 and agents[2].run_kwargs)
        elapsed = time.monotonic() - start
        # The stale ack did NOT release the wait: the full (reduced)
        # flush timeout was burned before proceeding.
        assert elapsed >= 0.3
        assert agents[2].run_kwargs["user_message"] == "second question for you"
        assert not vloop._flushed.is_set()
    finally:
        await vloop.stop()
        await task


@pytest.mark.asyncio
async def test_barge_in_before_agent_constructed_still_interrupts(monkeypatch):
    """A barge-in landing while the executor thread is still CONSTRUCTING
    the agent (agent_ref[0] not yet set) must still interrupt the
    generation: the interrupt latch is checked right after construction,
    before the first iteration."""
    construction_entered = threading.Event()
    construction_gate = threading.Event()
    agents = []

    def _gated_create(session_id, stream_delta_callback,
                      tool_progress_callback, *, extra=None):
        construction_entered.set()
        construction_gate.wait(10.0)     # hold agent_ref[0] unset
        agent = FakeAgent(
            speak_then_block("A story that should never be told. "),
            stream_delta_callback, tool_progress_callback)
        agents.append(agent)
        return agent

    monkeypatch.setattr(turn_loop, "_create_voice_agent", _gated_create)
    stt, factory, transport = FakeSTT(), FakeTTSFactory(), FakeTransport()
    vloop = _make_loop(stt, factory, transport)
    task = asyncio.create_task(vloop.run())
    try:
        await _eventually(lambda: construction_entered.is_set())
        # Barge-in NOW: no agent exists yet, so the direct interrupt path
        # cannot fire — only the latch can carry it.
        stt.push_text("stop")
        await _eventually(
            lambda: vloop._interrupt_latch is not None
            and vloop._interrupt_latch.is_set(),
            msg="barge-in did not latch before agent construction")
        assert agents == []              # constructed AFTER the barge-in
        construction_gate.set()
        await _eventually(
            lambda: agents and agents[0].interrupt_event.is_set(),
            msg="latched barge-in never interrupted the agent")
        assert agents[0].interrupt_reason == "user barge-in (voice)"
        await _eventually(lambda: vloop._state == turn_loop.LISTENING)
        assert vloop._history == []
    finally:
        construction_gate.set()
        await vloop.stop()
        await task


@pytest.mark.asyncio
async def test_interrupted_turn_await_is_bounded(monkeypatch, caplog):
    """An agent that IGNORES interrupt() cannot block the consumer/teardown
    forever: the post-interrupt await on the executor future is bounded by
    RUN_FUTURE_TIMEOUT_S and aborts with a loud log."""
    assert turn_loop.RUN_FUTURE_TIMEOUT_S == 60.0   # generous default bound
    monkeypatch.setattr(turn_loop, "RUN_FUTURE_TIMEOUT_S", 0.3)
    release = threading.Event()

    def deaf_behavior(agent):
        # Ignores agent.interrupt_event entirely — a worst-case generation.
        agent.delta_cb("I am going to keep talking no matter what you say. ")
        release.wait(10.0)
        return {"final_response": "too late"}

    agents = install_agents(monkeypatch, [deaf_behavior])
    stt, factory, transport = FakeSTT(), FakeTTSFactory(), FakeTransport()
    vloop = _make_loop(stt, factory, transport)
    task = asyncio.create_task(vloop.run())
    try:
        await _eventually(lambda: vloop._state == turn_loop.SPEAKING
                          and factory.instances
                          and factory.instances[0].sent)
        with caplog.at_level("ERROR", logger=turn_loop.__name__):
            start = time.monotonic()
            stt.push_text("please stop now")
            # Returns to LISTENING well before the 10s the deaf agent
            # blocks for — the await was bounded, not orphaned silently.
            await _eventually(lambda: vloop._state == turn_loop.LISTENING,
                              msg="bounded await never released the loop")
            assert time.monotonic() - start < 5.0
        assert agents[0].interrupt_event.is_set()   # interrupt WAS issued
        assert "abandoning executor thread" in caplog.text
        assert vloop._history == []                  # nothing recorded
    finally:
        release.set()
        await vloop.stop()
        await task


@pytest.mark.asyncio
async def test_stop_cancels_pending_finalize_before_barge_in(monkeypatch):
    """stop() must cancel a pending finalize BEFORE barge-in: otherwise the
    finalize can _start_turn in the window and orphan a fresh turn."""
    agents = install_agents(monkeypatch, [say(GREETING_REPLY)])
    stt = FakeSTT(auto_ack=False)    # finalize blocks awaiting the flush ack
    factory, transport = FakeTTSFactory(), FakeTransport()
    vloop = _make_loop(stt, factory, transport)
    task = asyncio.create_task(vloop.run())
    await _eventually(lambda: vloop._state == turn_loop.LISTENING
                      and len(vloop._history) == 1)
    stt.push_text("one last thing before you go")
    stt.push_end_of_turn_step()
    await _eventually(lambda: vloop._finalize_task is not None,
                      msg="finalize never became pending")
    # Spy: by the time stop() reaches _barge_in, finalize must be gone.
    orig_barge_in = vloop._barge_in
    finalize_gone_at_barge_in = []

    async def spy_barge_in():
        t = vloop._finalize_task
        finalize_gone_at_barge_in.append(t is None or t.done())
        await orig_barge_in()

    vloop._barge_in = spy_barge_in
    await vloop.stop()
    await asyncio.wait_for(task, timeout=5.0)
    assert finalize_gone_at_barge_in == [True]
    # The pending finalize never spawned a fresh (orphaned) turn.
    assert len(agents) == 1
    assert len(factory.instances) == 1
    assert vloop._history == [{"role": "assistant", "content": GREETING_REPLY}]
