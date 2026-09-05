"""The gate, driven by pairwise scores, through the real consolidate() loop.

The unit tests pin what PairwiseJudge returns. This pins what that does to the
decision: that a baseline anchors at exactly 0.5 inside upstream's own
arithmetic, and that `candidate > baseline` therefore means "wins more
head-to-heads than it loses" rather than "two noisy absolute means differ".

Nothing here calls out. The fake backend answers well only when the skill
carries the rule, and the fake judge prefers the better answer — so the loop
exercised is upstream's, and only the verdicts are scripted.
"""

import json

from skillopt_sleep.consolidate import consolidate
from skillopt_sleep.types import EditRecord, TaskRecord

from hermes_skillopt.llm_backend import PairwiseJudge

RULE = "Answer from your own knowledge; never decline for lack of tools."
GOOD = "Tides are raised by the Moon's differential gravity, roughly twice daily."
POOR = "I would need tool access to answer that."


def _tasks(n=4):
    return [
        TaskRecord(
            id=f"t{i}",
            project="hermes",
            intent="why are there tides",
            reference_kind="rubric",
            reference="Explain tides. Do not decline.",
            split="train" if i % 2 else "val",
        )
        for i in range(n)
    ]


class ScriptedBackend:
    """Answers well iff the skill carries RULE; judges the better answer better."""

    name = "scripted"

    def __init__(self, edits=(RULE,)):
        self.edits = list(edits)
        self.comparisons = 0

    def attempt(self, task, skill, memory):
        return GOOD if RULE in (skill or "") else POOR

    def judge(self, task, response):
        # The absolute judge, used only for the baseline's why-wrong note.
        return (0.0, 0.1, "declined instead of answering")

    def _call(self, prompt, *, max_tokens=1024):
        self.comparisons += 1
        a_start = prompt.index("# Answer A")
        b_start = prompt.index("# Answer B")
        a_is_good = GOOD in prompt[a_start:b_start]
        return json.dumps({"winner": "A" if a_is_good else "B", "reason": "answers"})

    def reflect(self, failures, successes, skill, memory, *, edit_budget,
                evolve_skill, evolve_memory):
        return [EditRecord(target="skill", op="add", content=e, rationale="fix")
                for e in self.edits]

    def tokens_used(self):
        return 0


def _run(backend, skill):
    return consolidate(
        PairwiseJudge(backend), _tasks(), skill, "",
        edit_budget=2, gate_metric="soft", gate_mode="on",
        evolve_skill=True, evolve_memory=False, night=1,
    )


def test_a_real_improvement_clears_the_gate():
    result = _run(ScriptedBackend(), "# Digest skill\n")

    assert result.baseline_score == 0.5, "the baseline must anchor at a tie with itself"
    assert result.candidate_score == 1.0, "the candidate won every head-to-head"
    assert result.accepted
    assert any(RULE in e.content for e in result.applied_edits)


def test_an_edit_that_changes_nothing_is_rejected():
    """The candidate's responses are identical, so it ties itself and cannot pass 0.5."""
    backend = ScriptedBackend(edits=("Prefer clarity.",))  # never reaches attempt()
    result = _run(backend, "# Digest skill\n")

    assert result.baseline_score == 0.5
    assert result.candidate_score == 0.5
    assert not result.accepted
    assert backend.comparisons == 0, "identical text is settled without a judge call"


def test_a_regression_is_rejected():
    """Start from a skill that already works; the edit cannot help, so it must not pass."""
    result = _run(ScriptedBackend(edits=("Be brief.",)), f"# Digest skill\n{RULE}\n")

    assert result.baseline_score == 0.5
    assert result.candidate_score <= 0.5
    assert not result.accepted


def test_the_absolute_judge_misses_the_same_improvement():
    """Why this exists, as a test: the identical real improvement, judged absolutely.

    ScriptedBackend.judge returns the same score for any response, which is the
    degenerate form of the failure measured live — an absolute judge whose output
    does not track the answer. The gate then sees no movement and rejects an edit
    that demonstrably fixed every response. Pairwise accepts it (above).
    """
    backend = ScriptedBackend()
    result = consolidate(
        backend, _tasks(), "# Digest skill\n", "",
        edit_budget=2, gate_metric="soft", gate_mode="on",
        evolve_skill=True, evolve_memory=False, night=1,
    )

    assert result.baseline_score == result.candidate_score
    assert not result.accepted
