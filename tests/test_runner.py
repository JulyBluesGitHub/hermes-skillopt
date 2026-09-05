import json
import sqlite3
import time
from pathlib import Path

from hermes_skillopt.run_nightly import (
    adopt_all_staged,
    adopt_staging_dirs,
    recently_used_skills,
    write_staging_report,
)


def _create_history(home: Path) -> None:
    home.mkdir()
    db = sqlite3.connect(home / "state.db")
    db.executescript(
        """
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY,
            started_at REAL,
            archived INTEGER DEFAULT 0,
            system_prompt TEXT
        );
        CREATE TABLE messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            role TEXT,
            content TEXT,
            tool_name TEXT,
            active INTEGER DEFAULT 1
        );
        """
    )
    db.execute("INSERT INTO sessions VALUES (?, ?, 0, NULL)", ("s1", time.time()))
    for name in ("daily-ai-news-digest", "github"):
        db.execute(
            "INSERT INTO messages (session_id, role, content, tool_name, active) VALUES (?, 'tool', ?, 'skill_view', 1)",
            ("s1", json.dumps({"success": True, "name": name})),
        )
    db.commit()
    db.close()


def _stage(root: Path, skill_name: str, live: Path, proposed: str) -> Path:
    report = {
        "accepted": True,
        "edits": [{"target": "skill", "op": "add", "content": proposed}],
        "rejected_edits": [],
    }
    return Path(write_staging_report(skill_name, report, proposed, str(live), str(root)))


def test_recently_used_skills_reads_skill_view_results(tmp_path):
    home = tmp_path / "hermes"
    _create_history(home)

    skills = recently_used_skills(str(home), lookback_hours=24)

    assert skills == ["daily-ai-news-digest", "github"]


def test_cmd_adopt_all_handles_hyphenated_skill_names(tmp_path):
    staging = tmp_path / "staging"
    staging.mkdir()
    live = tmp_path / "claude-code.md"
    live.write_text("old", encoding="utf-8")
    _stage(staging, "claude-code", live, "new")

    adopted = adopt_all_staged(str(staging))

    assert adopted == 1
    assert live.read_text(encoding="utf-8") == "new"


def test_adopt_staging_dirs_only_adopts_the_new_run(tmp_path):
    staging = tmp_path / "staging"
    staging.mkdir()
    old_live = tmp_path / "old.md"
    new_live = tmp_path / "new.md"
    old_live.write_text("old original", encoding="utf-8")
    new_live.write_text("new original", encoding="utf-8")
    old_dir = _stage(staging, "old-skill", old_live, "old proposal")
    new_dir = _stage(staging, "new-skill", new_live, "new proposal")

    adopted = adopt_staging_dirs([str(new_dir)])

    assert adopted == 1
    assert old_live.read_text(encoding="utf-8") == "old original"
    assert new_live.read_text(encoding="utf-8") == "new proposal"
    assert old_dir.exists()
