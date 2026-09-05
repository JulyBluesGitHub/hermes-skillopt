"""Mined tasks must carry a rubric the judge can discriminate on.

With no reference, the mock judge scores `1.0 if outcome == "success"` — fixed
history that ignores the response entirely, which is why every skill scored the
same constant. The LLM judge fared little better, receiving a 200-character
truncation of the request with the model name prefixed to it.
"""

from hermes_skillopt.sleep import (
    HermesSession,
    PromptRecord,
    build_task_rubric,
    mine_hermes_tasks,
)


def _record(prompt="Summarise the release notes", **kw):
    return PromptRecord(prompt=prompt, **kw)


def test_rubric_carries_the_request():
    rubric = build_task_rubric(_record("Explain the retry policy"))
    assert "Explain the retry policy" in rubric
    assert "A good response must:" in rubric


def test_rubric_requires_a_claim_to_rest_on_something():
    """The defect these two lines close: nothing required an answer to be right.

    Asked "can we push?", a wrong four-character "Yes." beat a baseline that
    correctly reported it could not verify that from the session — in both
    judges, in both orders. "Yes." satisfied "directly address the request" and
    the honest answer read as the hedging the findings requirement penalizes.
    """
    rubric = build_task_rubric(_record("can we push?"))
    assert "rest each claim on something it can point to" in rubric
    assert "cannot be settled from what it has" in rubric


def test_reporting_a_limit_is_not_penalized_as_a_plan():
    """The findings requirement names plans, not every answer that stops short.

    "not a description of how it would proceed" also caught "I checked and this
    session cannot tell me" — a finding, phrased as a limit. Naming the target
    is what lets both requirements hold at once.
    """
    rubric = build_task_rubric(_record())
    assert "not a plan for how it would arrive at them" in rubric
    assert "how it would proceed" not in rubric


def test_tool_failure_adds_an_honesty_requirement():
    clean = build_task_rubric(_record(response="done"))
    failed = build_task_rubric(_record(response="done", tool_errors=["shell: boom"]))
    assert "instead of presenting" not in clean
    assert "instead of presenting" in failed


def test_unanswered_turn_adds_an_answer_requirement():
    rubric = build_task_rubric(_record(outcome="fail", tool_errors=["x: boom"]))
    assert "previously went unanswered" in rubric


def test_long_requests_are_bounded_but_generous():
    rubric = build_task_rubric(_record("word " * 500))
    assert 800 < len(rubric) < 1400   # full request would be 2500


def test_mined_tasks_carry_a_rubric_and_a_clean_intent():
    # tools_used carries skill_view because harvesting only ever sets
    # skills_loaded from one: a record attributed to a skill with no skill_view
    # anywhere is not a state the miner can see.
    record = PromptRecord(prompt="Ship the changelog", response="done",
                          outcome="success", skills_loaded=["demo"],
                          tools_used=["skill_view"])
    session = HermesSession(
        session_id="s1", title="t", cwd="", model="gpt-5.6-sol",
        started_at=0.0, ended_at=1.0, completed=True,
        message_count=2, tool_call_count=0,
        skills_loaded=["demo"], prompt_records=[record],
    )
    tasks = mine_hermes_tasks([session], skill_name="demo", max_tasks=4)

    assert tasks, "expected at least one mined task"
    task = tasks[0]
    assert task.reference_kind == "rubric"
    assert "Ship the changelog" in task.reference
    # the model name is noise to a judge; it already lives in tags
    assert not task.intent.startswith("[")
    assert task.intent.startswith("Ship the changelog")
    assert any("gpt" in t or t == "hermes" for t in task.tags)
