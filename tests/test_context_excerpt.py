"""Give the replay the conversation the original turn actually had.

`context_excerpt` is pasted straight into the attempt prompt, right after the
intent. It used to hold the session title, the model name and a list of skill
names — no conversation at all. Every mined turn sits after some earlier turn
that ran a tool (measured: 0% have a tool-free prefix), so the replayed model
was routinely asked about a repository it had never been shown. It correctly
answered that nothing had been provided, and lost to a baseline that guessed.

These tests pin what the excerpt must carry, what it must not, and what it
must never leak.
"""

import json

from hermes_skillopt.sleep import (
    HermesSession,
    PromptRecord,
    build_context_excerpt,
    harvest_hermes_sessions,
    mine_hermes_tasks,
    redact_secrets,
)
from tests.test_session_mining import _make_db, _skill_result  # noqa: F401


def _session(*records: PromptRecord) -> HermesSession:
    return HermesSession(
        session_id="sess-abcdef123456",
        title="Daily digest",
        cwd="/repo",
        model="test-model",
        started_at=0.0,
        ended_at=1.0,
        completed=True,
        message_count=0,
        tool_call_count=0,
        prompt_records=list(records),
    )


def _turn(prompt, response="", tools=(), skills=("digest",)):
    return PromptRecord(
        prompt=prompt,
        response=response,
        skills_loaded=list(skills),
        tools_used=[name for name, _ in tools],
        tool_results=[(name, result) for name, result in tools],
    )


# ── what the excerpt carries ─────────────────────────────────────────────────

def test_the_first_turn_has_no_conversation_to_report():
    session = _session(_turn("start here", "started"))

    excerpt = build_context_excerpt(session, 0)

    assert "Earlier in this conversation" not in excerpt
    assert excerpt.startswith("Session: Daily digest")


def test_a_later_turn_carries_the_earlier_request_tools_and_reply():
    session = _session(
        _turn("what does this repo do?", "It optimizes skills.",
              tools=[("terminal", "README.md: a skill optimizer")]),
        _turn("explain the changes", "..."),
    )

    excerpt = build_context_excerpt(session, 1)

    assert "[user] what does this repo do?" in excerpt
    assert "[tool:terminal] README.md: a skill optimizer" in excerpt
    assert "[assistant] It optimizes skills." in excerpt


def test_the_turn_being_replayed_is_not_in_its_own_excerpt():
    """Its answer is the thing replay is supposed to produce."""
    session = _session(
        _turn("earlier question", "earlier answer"),
        _turn("the mined request", "the answer replay must reproduce"),
    )

    excerpt = build_context_excerpt(session, 1)

    assert "the answer replay must reproduce" not in excerpt
    assert "the mined request" not in excerpt


def test_turns_are_rendered_oldest_first():
    session = _session(
        _turn("first", "one"), _turn("second", "two"), _turn("third", "three"),
    )

    excerpt = build_context_excerpt(session, 2)

    assert excerpt.index("first") < excerpt.index("second")


def test_a_turn_with_no_reply_still_contributes_its_request():
    session = _session(_turn("did this work?"), _turn("and now?"))

    excerpt = build_context_excerpt(session, 1)

    assert "[user] did this work?" in excerpt
    assert "[assistant]" not in excerpt


# ── what the excerpt spends ──────────────────────────────────────────────────

def test_the_budget_keeps_the_turns_nearest_the_task():
    """Overflow drops the oldest turns, because the request refers back nearby."""
    session = _session(
        *[_turn(f"turn {i} " + "padding " * 60, f"reply {i} " + "filler " * 60)
          for i in range(8)],
        _turn("the mined request"),
    )

    excerpt = build_context_excerpt(session, 8)

    assert "turn 7" in excerpt
    assert "turn 0" not in excerpt
    assert len(excerpt) < 3000


def test_a_single_oversized_turn_is_kept_rather_than_dropped():
    """Trimming it would cost the reply at its end; dropping it empties the excerpt."""
    session = _session(
        _turn("q " + "x " * 500, "a " + "y " * 500,
              tools=[("terminal", "z " * 500)]),
        _turn("the mined request"),
    )

    excerpt = build_context_excerpt(session, 1)

    assert "[user]" in excerpt and "[assistant]" in excerpt


def test_only_the_last_tool_results_of_a_turn_are_kept():
    """A turn can call a tool twenty times; the calls behind the reply are late."""
    session = _session(
        _turn("do it", "done",
              tools=[("terminal", f"result {i}") for i in range(9)]),
        _turn("and now?"),
    )

    excerpt = build_context_excerpt(session, 1)

    assert excerpt.count("[tool:terminal]") == 4
    assert "result 8" in excerpt
    assert "result 0" not in excerpt


def test_a_long_tool_dump_is_condensed_to_one_line():
    session = _session(
        _turn("list it", "listed", tools=[("terminal", "a\n\n   b\n\tc")]),
        _turn("and now?"),
    )

    excerpt = build_context_excerpt(session, 1)

    assert "[tool:terminal] a b c" in excerpt


# ── what the excerpt must never leak ─────────────────────────────────────────

def test_credentials_in_tool_output_do_not_reach_the_prompt():
    session = _session(
        _turn("cat the env", "here it is",
              tools=[("terminal", "DEEPSEEK_API_KEY=sk-9f8e7d6c5b4a3210 PORT=8080")]),
        _turn("and now?"),
    )

    excerpt = build_context_excerpt(session, 1)

    assert "sk-9f8e7d6c5b4a3210" not in excerpt
    assert "DEEPSEEK_API_KEY=<redacted>" in excerpt
    assert "PORT=8080" in excerpt, "redaction must not eat ordinary configuration"


def test_redaction_keeps_the_name_that_labelled_the_secret():
    """The model still needs to know a key was there, just not what it was."""
    redacted = redact_secrets('{"PINECONE_KEY": "abcd1234efgh", "region": "us-east"}')

    assert "abcd1234efgh" not in redacted
    assert "PINECONE_KEY" in redacted
    assert "us-east" in redacted


def test_bare_provider_keys_are_redacted_without_a_name():
    assert "ghp_" not in redact_secrets("pushed with ghp_AbCdEf0123456789")
    assert "<redacted>" in redact_secrets("Authorization: Bearer eyJhbGciOiJIUzI1NiJ9")


def test_a_value_ends_at_a_json_escape_not_just_at_whitespace():
    """Tool results arrive JSON-encoded, where the line break is a literal `\\n`."""
    redacted = redact_secrets(r"api_key: 'abcd1234'\n10| context_length: 900000")

    assert "abcd1234" not in redacted
    assert "context_length: 900000" in redacted


def test_ordinary_prose_survives_redaction():
    text = "the tokens are fine and the key idea is simple"

    assert redact_secrets(text) == text


# ── end to end, from the session DB ──────────────────────────────────────────

def test_a_mined_task_carries_the_conversation_that_preceded_it(tmp_path):
    home = _make_db(
        tmp_path,
        [
            {"role": "user", "content": "can we push?"},
            {"role": "tool", "tool_name": "skill_view", "content": _skill_result("digest")},
            {
                "role": "tool",
                "tool_name": "terminal",
                "content": json.dumps({"output": "HEAD matches origin/main", "exit_code": 0}),
            },
            {"role": "assistant", "content": "Yes, already pushed."},
            {"role": "user", "content": "explain this repo to me"},
            {"role": "assistant", "content": "It is a skill optimizer."},
        ],
    )

    sessions = harvest_hermes_sessions(str(home))
    tasks = mine_hermes_tasks(sessions, skill_name="digest")

    # The `terminal` turn is unreplayable and dropped; the turn after it is mined.
    assert [task.intent for task in tasks] == ["explain this repo to me"]
    excerpt = tasks[0].context_excerpt
    assert "[user] can we push?" in excerpt
    assert "HEAD matches origin/main" in excerpt
    assert "[assistant] Yes, already pushed." in excerpt
    assert "test-model" not in excerpt, "the replaying model is not told which model answered"
