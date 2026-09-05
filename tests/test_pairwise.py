"""Pairwise judging: does a response beat the baseline, rather than what does it score.

The absolute judge collapsed to a coarse switch — measured live it emitted only
{0.0, 0.1, 0.9, 1.0} and rated a four-character response 1.00. These tests pin
the properties that make the comparison judge a usable replacement: a baseline
anchored at 0.5 so the gate compares like with like, agreement across both
presentation orders so a verdict is about the answers and not their position,
and a loud failure when the judge says nothing intelligible.
"""

import json

import pytest
from skillopt_sleep.types import TaskRecord

from hermes_skillopt.llm_backend import (
    BackendCallError,
    PairwiseJudge,
    build_validating_backend,
)


def _task(task_id="t1", rubric="Answer the question about tides completely."):
    return TaskRecord(
        id=task_id,
        project="hermes",
        intent="why are there tides",
        context_excerpt="",
        reference_kind="rubric",
        reference=rubric,
    )


class FakeBackend:
    """Records prompts and replies from a scripted queue of winners."""

    name = "fake"

    def __init__(self, winners=(), absolute=(1.0, 1.0, "looks fine")):
        self.prompts = []
        self.winners = list(winners)
        self.absolute = absolute
        self.absolute_calls = 0

    def _call(self, prompt, *, max_tokens=1024):
        self.prompts.append(prompt)
        if not self.winners:
            raise AssertionError("more comparisons than the test scripted")
        return json.dumps({"winner": self.winners.pop(0), "reason": "because"})

    def judge(self, task, response):
        self.absolute_calls += 1
        return self.absolute

    def attempt(self, task, skill, memory):
        return "delegated"

    def tokens_used(self):
        return 42


def test_baseline_is_a_tie_with_itself():
    """The gate's reference has to be 0.5, or a win rate is not comparable to it."""
    fake = FakeBackend()
    judge = PairwiseJudge(fake)

    hard, soft, reason = judge.judge(_task(), "the first response")

    assert (hard, soft) == (0.5, 0.5)
    assert reason.startswith("baseline:")
    assert fake.prompts == []  # a baseline is not compared against anything


def test_baseline_rationale_carries_the_absolute_diagnosis():
    """reflect reads fail_reason; every baseline is a 0.5, so it needs prose."""
    fake = FakeBackend(absolute=(0.0, 0.1, "never answered the question"))
    judge = PairwiseJudge(fake)

    _hard, _soft, reason = judge.judge(_task(), "I would need tool access.")

    assert "never answered the question" in reason
    assert fake.absolute_calls == 1


def test_baseline_diagnosis_can_be_switched_off():
    fake = FakeBackend()
    judge = PairwiseJudge(fake, diagnose_baseline=False)

    _hard, _soft, reason = judge.judge(_task(), "first")

    assert fake.absolute_calls == 0
    assert reason == "baseline (no comparison yet)"


def test_a_broken_absolute_judge_cannot_fail_the_run():
    """The diagnosis is a comment, not a score; it must never take the run down."""
    fake = FakeBackend()
    fake.judge = lambda task, response: (_ for _ in ()).throw(BackendCallError("dead"))
    judge = PairwiseJudge(fake)

    hard, soft, reason = judge.judge(_task(), "first")

    assert (hard, soft) == (0.5, 0.5)
    assert reason == "baseline (no comparison yet)"


def test_candidate_winning_both_orders_scores_one():
    # baseline shown as A -> "B" wins; candidate shown as A -> "A" wins.
    fake = FakeBackend(winners=["B", "A"])
    judge = PairwiseJudge(fake)
    task = _task()
    judge.judge(task, "baseline answer")

    hard, soft, reason = judge.judge(task, "much better answer")

    assert (hard, soft) == (1.0, 1.0)
    assert "wins both orders" in reason
    assert len(fake.prompts) == 2


def test_candidate_losing_both_orders_scores_zero():
    fake = FakeBackend(winners=["A", "B"])
    judge = PairwiseJudge(fake)
    task = _task()
    judge.judge(task, "baseline answer")

    hard, soft, reason = judge.judge(task, "worse answer")

    assert (hard, soft) == (0.0, 0.0)
    assert "loses both orders" in reason


def test_position_dependent_verdict_is_a_tie():
    """Whichever answer is shown first wins: that is bias, not a result."""
    fake = FakeBackend(winners=["A", "A"])  # first-shown wins in both runs
    judge = PairwiseJudge(fake)
    task = _task()
    judge.judge(task, "baseline answer")

    hard, soft, reason = judge.judge(task, "candidate answer")

    assert (hard, soft) == (0.5, 0.5)
    assert "order-dependent" in reason


def test_declared_tie_in_both_orders_is_a_tie():
    fake = FakeBackend(winners=["tie", "tie"])
    judge = PairwiseJudge(fake)
    task = _task()
    judge.judge(task, "baseline answer")

    assert judge.judge(task, "different but equal")[:2] == (0.5, 0.5)


def test_an_unchanged_response_costs_nothing():
    """reflect often proposes an edit the model ignores; equality settles that."""
    fake = FakeBackend()
    judge = PairwiseJudge(fake)
    task = _task()
    judge.judge(task, "same text")

    hard, soft, reason = judge.judge(task, "same text")

    assert (hard, soft) == (0.5, 0.5)
    assert reason == "identical to baseline"
    assert fake.prompts == []


def test_baselines_are_tracked_per_task():
    fake = FakeBackend(winners=["B", "A"])
    judge = PairwiseJudge(fake)
    judge.judge(_task("t1"), "answer one")
    judge.judge(_task("t2"), "answer two")

    # t2's own baseline is "answer two", so "answer one" is a genuine candidate.
    assert judge.judge(_task("t2"), "answer one")[:2] == (1.0, 1.0)


def test_an_unintelligible_verdict_raises():
    """Scoring a broken judge as a tie is the silent no-change this repo exists to stop."""
    fake = FakeBackend()
    fake._call = lambda prompt, *, max_tokens=1024: "I cannot decide."
    judge = PairwiseJudge(fake)
    task = _task()
    judge.judge(task, "baseline")

    with pytest.raises(BackendCallError, match="names no winner"):
        judge.judge(task, "candidate")


def test_the_rubric_and_both_answers_reach_the_judge():
    fake = FakeBackend(winners=["B", "A"])
    judge = PairwiseJudge(fake)
    task = _task(rubric="Must cite a date.")
    judge.judge(task, "baseline text")
    judge.judge(task, "candidate text")

    first, second = fake.prompts
    assert "Must cite a date." in first
    assert "baseline text" in first and "candidate text" in first
    # ...and the swapped run really is swapped
    assert first.index("baseline text") < first.index("candidate text")
    assert second.index("candidate text") < second.index("baseline text")


def test_everything_else_delegates():
    fake = FakeBackend()
    judge = PairwiseJudge(fake)

    assert judge.attempt(_task(), "skill", "memory") == "delegated"
    assert judge.tokens_used() == 42
    assert judge.name == "fake+pairwise"


def test_mock_backend_refuses_to_judge_pairwise():
    """Its scores come from recorded outcomes, so both sides would tie forever."""
    with pytest.raises(ValueError, match="cannot judge pairwise"):
        build_validating_backend("mock", judge_mode="pairwise")


def test_unknown_judge_mode_is_rejected():
    with pytest.raises(ValueError, match="unknown judge mode"):
        build_validating_backend("claude", judge_mode="elo")


def test_responses_containing_percent_and_braces_do_not_break_the_prompt():
    """A news digest says "up 40%"; a code answer contains {}. Neither is a format spec."""
    fake = FakeBackend(winners=["B", "A"])
    judge = PairwiseJudge(fake)
    task = _task(rubric="Report the change, 100% precisely.")
    judge.judge(task, "revenue fell 12% {stale}")

    hard, _soft, _reason = judge.judge(task, "revenue rose 40% -> {'q': 3}")

    assert hard == 1.0
    assert "revenue rose 40% -> {'q': 3}" in fake.prompts[0]
    assert "100% precisely" in fake.prompts[0]
