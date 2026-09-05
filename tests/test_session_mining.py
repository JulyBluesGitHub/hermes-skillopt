import json
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

from hermes_skillopt.sleep import harvest_hermes_sessions, mine_hermes_tasks


def _make_db(tmp_path: Path, messages: list[dict], *, ended: bool = True) -> Path:
    home = tmp_path / "hermes"
    home.mkdir(parents=True)
    db = sqlite3.connect(home / "state.db")
    db.executescript(
        """
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY,
            title TEXT,
            cwd TEXT,
            model TEXT,
            started_at REAL,
            ended_at REAL,
            message_count INTEGER,
            tool_call_count INTEGER,
            archived INTEGER DEFAULT 0,
            system_prompt TEXT
        );
        CREATE TABLE messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            role TEXT,
            content TEXT,
            tool_name TEXT,
            tool_calls TEXT,
            timestamp REAL,
            active INTEGER DEFAULT 1
        );
        """
    )
    now = time.time()
    db.execute(
        "INSERT INTO sessions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("s1", "test", str(tmp_path), "test-model", now, now if ended else None, len(messages), 1, 0, None),
    )
    for message in messages:
        db.execute(
            """INSERT INTO messages
               (session_id, role, content, tool_name, tool_calls, timestamp, active)
               VALUES (?, ?, ?, ?, ?, ?, 1)""",
            (
                "s1",
                message["role"],
                message.get("content", ""),
                message.get("tool_name"),
                message.get("tool_calls"),
                now,
            ),
        )
    db.commit()
    db.close()
    return home


def _skill_result(name: str) -> str:
    return json.dumps({"success": True, "name": name, "content": "# Skill"})


def test_harvest_detects_loaded_skill_from_skill_view_result(tmp_path):
    home = _make_db(
        tmp_path,
        [
            {"role": "user", "content": "Make today's digest"},
            {"role": "tool", "tool_name": "skill_view", "content": _skill_result("daily-ai-news-digest")},
            {"role": "assistant", "content": "Digest created."},
        ],
    )

    sessions = harvest_hermes_sessions(str(home))

    assert sessions[0].skills_loaded == ["daily-ai-news-digest"]


def test_successful_tool_payload_with_null_error_is_not_a_failure(tmp_path):
    home = _make_db(
        tmp_path,
        [
            {"role": "user", "content": "Run the check"},
            {
                "role": "tool",
                "tool_name": "terminal",
                "content": json.dumps({"output": "ok", "exit_code": 0, "error": None}),
            },
            {"role": "assistant", "content": "The check passed."},
        ],
    )

    session = harvest_hermes_sessions(str(home))[0]
    tasks = mine_hermes_tasks([session])

    assert tasks[0].outcome == "success"
    assert tasks[0].attempted_solution == "The check passed."


def test_recovered_tool_error_is_mixed_and_keeps_final_response(tmp_path):
    home = _make_db(
        tmp_path,
        [
            {"role": "user", "content": "Run the check"},
            {
                "role": "tool",
                "tool_name": "terminal",
                "content": json.dumps({"output": "boom", "exit_code": 1, "error": "failed"}),
            },
            {"role": "assistant", "content": "I used a fallback and completed the check."},
        ],
    )

    session = harvest_hermes_sessions(str(home))[0]
    tasks = mine_hermes_tasks([session])

    assert tasks[0].outcome == "mixed"
    assert tasks[0].attempted_solution == "I used a fallback and completed the check."


def test_mining_for_skill_excludes_sessions_that_did_not_load_it(tmp_path):
    matching_home = _make_db(
        tmp_path / "matching",
        [
            {"role": "user", "content": "Matching task"},
            {"role": "tool", "tool_name": "skill_view", "content": _skill_result("target-skill")},
            {"role": "assistant", "content": "Matching response"},
        ],
    )
    other_home = _make_db(
        tmp_path / "other",
        [
            {"role": "user", "content": "Unrelated task"},
            {"role": "tool", "tool_name": "skill_view", "content": _skill_result("other-skill")},
            {"role": "assistant", "content": "Unrelated response"},
        ],
    )
    sessions = harvest_hermes_sessions(str(matching_home)) + harvest_hermes_sessions(str(other_home))

    tasks = mine_hermes_tasks(sessions, skill_name="target-skill")

    assert [task.intent for task in tasks] == ["[test-model] Matching task"]
    assert tasks[0].attempted_solution == "Matching response"


def test_mining_ignores_unfinished_sessions(tmp_path):
    home = _make_db(
        tmp_path,
        [
            {"role": "user", "content": "Task still running"},
            {"role": "tool", "tool_name": "skill_view", "content": _skill_result("target-skill")},
            {
                "role": "tool",
                "tool_name": "terminal",
                "content": json.dumps({"output": "boom", "exit_code": 1, "error": "failed"}),
            },
        ],
        ended=False,
    )

    session = harvest_hermes_sessions(str(home))[0]

    assert mine_hermes_tasks([session], skill_name="target-skill") == []


def test_sleep_module_help_exits_cleanly():
    result = subprocess.run(
        [sys.executable, "-m", "hermes_skillopt.sleep", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "On-demand SkillOpt-Sleep for Hermes Agent" in result.stdout
    assert "Nightly" not in result.stdout
