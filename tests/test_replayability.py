"""Only mine turns replay can fairly reproduce.

Replay is a single-shot text call with no tools. A turn that ran `terminal` to
answer "can we push?" cannot be replayed: the model cannot check, correctly says
so, and the rubric marks that down as describing how it would proceed. The
optimizer's way out is a rule like "do not decline, answer from what you know" —
which scores well in replay and is actively harmful in production, where the
agent has tools and the skill may have a fail-closed rule saying not to guess.

Measured on real history, this dropped the pathological task that let a
four-character "Yes." — the wrong answer — beat an honest "I cannot verify that".
"""

from hermes_skillopt.sleep import (
    REPLAYABLE_TOOLS,
    PromptRecord,
    harvest_hermes_sessions,
    mine_hermes_tasks,
    turn_needs_tools,
)
from tests.test_session_mining import _make_db, _skill_result  # noqa: F401


def _record(tools):
    return PromptRecord(prompt="do a thing", response="done", tools_used=list(tools))


def test_a_toolless_turn_is_replayable():
    assert turn_needs_tools(_record([])) == []


def test_skill_view_alone_stays_replayable():
    """Replay puts the skill in the prompt, so loading it is reproduced.

    It also cannot be treated as an external dependency: attribution requires a
    successful skill_view, so counting it would exclude every task.
    """
    assert "skill_view" in REPLAYABLE_TOOLS
    assert turn_needs_tools(_record(["skill_view"])) == []


def test_a_turn_that_shelled_out_is_not_replayable():
    assert turn_needs_tools(_record(["skill_view", "terminal"])) == ["terminal"]


def test_every_blocking_tool_is_reported():
    needed = turn_needs_tools(_record(["skill_view", "terminal", "browser_navigate"]))
    assert needed == ["terminal", "browser_navigate"]


def test_mining_drops_unreplayable_turns_and_says_which_tool(tmp_path):
    home = _make_db(
        tmp_path,
        [
            {"role": "user", "content": "can we push?"},
            {"role": "tool", "tool_name": "skill_view", "content": _skill_result("s")},
            {"role": "tool", "tool_name": "terminal", "content": '{"exit_code": 0}'},
            {"role": "assistant", "content": "Not yet, I cannot verify that."},
            {"role": "user", "content": "explain this repo"},
            {"role": "assistant", "content": "It is a skill optimizer."},
        ],
    )
    session = harvest_hermes_sessions(str(home))[0]

    skipped = {}
    tasks = mine_hermes_tasks([session], skipped=skipped)

    assert [t.intent for t in tasks] == ["explain this repo"]
    assert skipped == {"terminal": 1}


def test_the_filter_can_be_switched_off(tmp_path):
    home = _make_db(
        tmp_path,
        [
            {"role": "user", "content": "can we push?"},
            {"role": "tool", "tool_name": "terminal", "content": '{"exit_code": 0}'},
            {"role": "assistant", "content": "Not yet."},
        ],
    )
    session = harvest_hermes_sessions(str(home))[0]

    assert mine_hermes_tasks([session]) == []
    assert len(mine_hermes_tasks([session], require_replayable=False)) == 1


def test_task_ids_stay_dense_when_turns_are_skipped(tmp_path):
    """A skipped turn must not burn an id; reports index tasks by it."""
    home = _make_db(
        tmp_path,
        [
            {"role": "user", "content": "shell first"},
            {"role": "tool", "tool_name": "terminal", "content": '{"exit_code": 0}'},
            {"role": "assistant", "content": "ran it"},
            {"role": "user", "content": "then talk"},
            {"role": "assistant", "content": "talking"},
        ],
    )
    session = harvest_hermes_sessions(str(home))[0]

    tasks = mine_hermes_tasks([session])

    assert [t.id for t in tasks] == ["hermes-1"]
