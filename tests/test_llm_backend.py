"""A dead backend must fail loudly, never score zero.

Before this, `CliBackend._call` returned "" on any failure — an unauthenticated
CLI, a timeout, a provider fault — and the empty response scored 0.0 while the
run exited 0 and staged a report. These tests pin the distinction between "the
candidate genuinely didn't help" and "nothing was measured".
"""

import pytest
from skillopt_sleep.types import TaskRecord

from hermes_skillopt.llm_backend import (
    BackendCallError,
    StrictBackend,
    _extract_content,
    _usage_tokens,
    build_validating_backend,
    load_hermes_env,
)


def _task(tid="t1"):
    return TaskRecord(
        id=tid, project="", intent="do the thing", context_excerpt="",
        attempted_solution="", outcome="success", reference_kind="none",
        reference="", judge={}, tags=[], source_sessions=[], split="train",
        origin="real",
    )


class FakeBackend:
    """Stand-in with the CliBackend shape."""

    name = "fake"

    def __init__(self, call_result="a real answer", judge_reason="ok"):
        self.call_result = call_result
        self.judge_reason = judge_reason
        self.calls = 0

    def _call(self, prompt, *, max_tokens=1024):
        self.calls += 1
        return self.call_result

    def attempt(self, task, skill, memory):
        return self._call(f"{skill}|{task.intent}")

    def judge(self, task, response):
        return 1.0, 1.0, self.judge_reason

    def reflect(self, *a, **k):
        return ["edit"]

    def tokens_used(self):
        return 42


# ── The core P1 guarantee ────────────────────────────────────────────────────

def test_empty_call_raises_instead_of_scoring_zero():
    strict = StrictBackend(FakeBackend(call_result=""))
    with pytest.raises(BackendCallError) as excinfo:
        strict.attempt(_task(), "skill text", "")
    assert "empty response" in str(excinfo.value)


def test_whitespace_only_response_is_also_a_failure():
    strict = StrictBackend(FakeBackend(call_result="   \n  "))
    with pytest.raises(BackendCallError):
        strict.attempt(_task(), "skill", "")


def test_unparseable_judge_raises():
    """0.0 from a broken judge is indistinguishable from a genuinely bad answer."""
    strict = StrictBackend(FakeBackend(judge_reason="judge-parse-failed"))
    with pytest.raises(BackendCallError) as excinfo:
        strict.judge(_task(), "some response")
    assert "unparseable" in str(excinfo.value)


def test_healthy_backend_passes_through_untouched():
    inner = FakeBackend()
    strict = StrictBackend(inner)
    assert strict.attempt(_task(), "skill", "") == "a real answer"
    assert strict.judge(_task(), "resp") == (1.0, 1.0, "ok")
    assert strict.reflect() == ["edit"]      # delegated via __getattr__
    assert strict.tokens_used() == 42
    assert strict.name == "fake"


def test_probe_uses_the_guarded_path():
    dead = StrictBackend(FakeBackend(call_result=""))
    with pytest.raises(BackendCallError):
        dead.probe()
    assert StrictBackend(FakeBackend()).probe() == "a real answer"


# ── Response parsing ─────────────────────────────────────────────────────────

class _Msg:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content):
        self.message = _Msg(content)


class _Resp:
    def __init__(self, content, usage=None):
        self.choices = [_Choice(content)]
        self.usage = usage


class _Usage:
    def __init__(self, p, c, total=None):
        self.prompt_tokens = p
        self.completion_tokens = c
        if total is not None:
            self.total_tokens = total


def test_extracts_content_and_survives_odd_shapes():
    assert _extract_content(_Resp("hello")) == "hello"
    assert _extract_content(_Resp(None)) == ""     # provider fault shape
    assert _extract_content(object()) == ""


def test_usage_prefers_total_then_sums():
    assert _usage_tokens(_Resp("x", _Usage(10, 5, total=15))) == 15
    assert _usage_tokens(_Resp("x", _Usage(10, 5))) == 15
    assert _usage_tokens(_Resp("x")) == 0


# ── Wiring ───────────────────────────────────────────────────────────────────

def test_mock_backend_is_not_wrapped():
    """Mock never calls out; wrapping it would only add a misleading guard."""
    be = build_validating_backend("mock")
    assert not isinstance(be, StrictBackend)
    assert be.name == "hermes"          # HermesBackend's own name


def test_env_loader_does_not_clobber_existing(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text(
        '# comment\nNEW_KEY="from-file"\nEXISTING=should-not-win\n\nBAD LINE\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("EXISTING", "already-set")
    monkeypatch.delenv("NEW_KEY", raising=False)

    applied = load_hermes_env(str(tmp_path))

    import os
    assert applied == 1
    assert os.environ["NEW_KEY"] == "from-file"
    assert os.environ["EXISTING"] == "already-set"


def test_env_loader_tolerates_missing_file(tmp_path):
    assert load_hermes_env(str(tmp_path / "nope")) == 0
