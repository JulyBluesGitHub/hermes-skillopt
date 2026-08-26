"""
Hermes SkillOpt Plugin — give your Hermes agent a skill optimization cycle.

Harvests Hermes session transcripts from the SQLite state DB, mines recurring
tasks, replays them offline, and consolidates validated improvements into
Hermes skill (.md) files — behind a held-out validation gate.

Usage:
    python -m hermes_skillopt.sleep dry-run --skill codex
    python -m hermes_skillopt.sleep status
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List

from skillopt_sleep.backend import get_backend
from skillopt_sleep.consolidate import consolidate
from skillopt_sleep.mine import assign_splits
from skillopt_sleep.types import TaskRecord

# ── Configuration ────────────────────────────────────────────────────────────

DEFAULT_HERMES_HOME = os.path.expanduser("~/AppData/Local/hermes")
DEFAULT_SKILLS_DIR = os.path.expanduser("~/AppData/Local/hermes/skills")


def _resolve_hermes_home() -> str:
    """Find the Hermes state directory."""
    for candidate in [
        os.environ.get("HERMES_HOME", ""),
        DEFAULT_HERMES_HOME,
        os.path.expanduser("~/.hermes"),
    ]:
        if candidate and os.path.isdir(candidate):
            return candidate
    return DEFAULT_HERMES_HOME


# ── Stage 1: Harvest Hermes Sessions ─────────────────────────────────────────

@dataclass
class PromptRecord:
    """One user turn and the evidence used to label its outcome."""

    prompt: str
    response: str = ""
    outcome: str = "unknown"
    tool_errors: List[str] = field(default_factory=list)
    skills_loaded: List[str] = field(default_factory=list)


@dataclass
class HermesSession:
    """A single Hermes session extracted from the state DB."""
    session_id: str
    title: str
    cwd: str
    model: str
    started_at: float
    ended_at: float
    completed: bool
    message_count: int
    tool_call_count: int
    user_prompts: List[str] = field(default_factory=list)
    assistant_responses: List[str] = field(default_factory=list)
    tool_errors: List[str] = field(default_factory=list)
    skills_loaded: List[str] = field(default_factory=list)
    prompt_records: List[PromptRecord] = field(default_factory=list)


def _skill_name_from_result(content: str) -> str:
    """Return the exact skill loaded by a successful skill_view result."""
    try:
        payload = json.loads(content or "{}")
    except (TypeError, json.JSONDecodeError):
        return ""
    if not isinstance(payload, dict) or payload.get("success") is False:
        return ""
    name = payload.get("name")
    return name.strip() if isinstance(name, str) else ""


def _tool_result_failed(content: str) -> bool:
    """Classify tool output without treating ``error: null`` as failure."""
    text = content or ""
    try:
        payload = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        lowered = text.lower()
        return (
            "traceback (most recent call last)" in lowered
            or lowered.lstrip().startswith("error:")
            or '"success": false' in lowered
        )

    if not isinstance(payload, dict):
        return False
    if payload.get("success") is False:
        return True
    exit_code = payload.get("exit_code")
    if isinstance(exit_code, int) and exit_code != 0:
        return True
    return bool(payload.get("error"))


def _finish_prompt(record: PromptRecord) -> None:
    if record.response and record.tool_errors:
        record.outcome = "mixed"
    elif record.response:
        record.outcome = "success"
    elif record.tool_errors:
        record.outcome = "fail"
    else:
        record.outcome = "unknown"


def harvest_hermes_sessions(
    hermes_home: str = "",
    lookback_hours: int = 72,
    max_sessions: int = 20,
) -> List[HermesSession]:
    """Extract recent Hermes sessions with their messages."""
    hermes_home = hermes_home or _resolve_hermes_home()
    db_path = os.path.join(hermes_home, "state.db")

    if not os.path.exists(db_path):
        print(f"[hermes-sleep] No Hermes DB found at {db_path}")
        return []

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Get recent sessions
    import time
    cutoff = time.time() - (lookback_hours * 3600)

    rows = conn.execute("""
        SELECT id, title, cwd, model, started_at, ended_at,
               message_count, tool_call_count
        FROM sessions
        WHERE started_at > ? AND archived = 0
        ORDER BY started_at DESC
        LIMIT ?
    """, (cutoff, max_sessions)).fetchall()

    sessions = []
    for row in rows:
        s = HermesSession(
            session_id=row["id"],
            title=row["title"] or "",
            cwd=row["cwd"] or "",
            model=row["model"] or "",
            started_at=row["started_at"],
            ended_at=row["ended_at"] or row["started_at"],
            completed=row["ended_at"] is not None,
            message_count=row["message_count"],
            tool_call_count=row["tool_call_count"],
        )

        # Get messages for this session (user + assistant + tool, in order)
        msgs = conn.execute("""
            SELECT role, content, tool_name, timestamp
            FROM messages
            WHERE session_id = ? AND active = 1
            ORDER BY id
        """, (s.session_id,)).fetchall()

        current: PromptRecord | None = None
        active_skills: List[str] = []

        for msg in msgs:
            role = msg["role"]
            content = msg["content"] or ""
            if role == "user" and content.strip():
                if current is not None:
                    _finish_prompt(current)
                    s.prompt_records.append(current)
                current = PromptRecord(
                    prompt=content.strip(),
                    skills_loaded=list(active_skills),
                )
            elif role == "assistant" and content.strip():
                if current is not None:
                    current.response = content.strip()[:1000]
                s.assistant_responses.append(content[:1000])
            elif role == "tool" and msg["tool_name"]:
                tool_name = msg["tool_name"]
                if tool_name == "skill_view":
                    skill_name = _skill_name_from_result(content)
                    if skill_name and skill_name not in active_skills:
                        active_skills.append(skill_name)
                    if current is not None and skill_name and skill_name not in current.skills_loaded:
                        current.skills_loaded.append(skill_name)
                    if skill_name and skill_name not in s.skills_loaded:
                        s.skills_loaded.append(skill_name)
                if _tool_result_failed(content):
                    error = f"{tool_name}: {content[:300]}"
                    s.tool_errors.append(error)
                    if current is not None:
                        current.tool_errors.append(error)

        if current is not None:
            _finish_prompt(current)
            s.prompt_records.append(current)

        s.user_prompts = [record.prompt[:500] for record in s.prompt_records]

        sessions.append(s)

    conn.close()
    return sessions


# ── Stage 2: Mine Tasks from Hermes Sessions ─────────────────────────────────

def mine_hermes_tasks(
    sessions: List[HermesSession],
    skill_name: str = "",
    max_tasks: int = 30,
) -> List[TaskRecord]:
    """Convert Hermes sessions into SkillOpt TaskRecords.

    Each "task" is derived from a user prompt paired with the assistant's
    response and any tool errors that followed.
    """
    tasks = []
    task_id = 0

    for session in sessions:
        if not session.completed:
            continue
        if skill_name and skill_name not in session.skills_loaded:
            continue

        for record in session.prompt_records:
            if skill_name and skill_name not in record.skills_loaded:
                continue
            task_id += 1

            task = TaskRecord(
                id=f"hermes-{task_id}",
                project=session.cwd,
                intent=f"[{session.model}] {record.prompt[:200]}",
                context_excerpt=f"Session: {session.title or session.session_id[:12]}\n"
                               f"Model: {session.model}\n"
                               f"Skills loaded: {', '.join(record.skills_loaded[:5])}",
                attempted_solution=record.response[:500],
                outcome=record.outcome,
                reference_kind="none",
                reference="",
                judge={},
                tags=["hermes", session.model or "unknown"] +
                     ([skill_name] if skill_name else []),
                source_sessions=[session.session_id],
                split="train",
                origin="real",
            )
            tasks.append(task)

        # Cap at max_tasks
        if len(tasks) >= max_tasks:
            break

    return tasks[:max_tasks]


# ── Stage 3+4: Load skill and run consolidate ────────────────────────────────

def load_hermes_skill(skill_name: str, skills_dir: str = "") -> str:
    """Load a Hermes skill .md file. Searches recursively for category/skill/SKILL.md."""
    import glob
    skills_dir = skills_dir or DEFAULT_SKILLS_DIR

    # Search for the skill in nested category folders
    pattern = os.path.join(skills_dir, "**", skill_name, "SKILL.md")
    matches = glob.glob(pattern, recursive=True)
    if matches:
        with open(matches[0], encoding="utf-8") as f:
            return f.read()

    # Try flat: skills_dir/skill_name/SKILL.md
    skill_path = os.path.join(skills_dir, skill_name, "SKILL.md")
    if os.path.exists(skill_path):
        with open(skill_path, encoding="utf-8") as f:
            return f.read()

    # Try flat file
    skill_path = os.path.join(skills_dir, f"{skill_name}.md")
    if os.path.exists(skill_path):
        with open(skill_path, encoding="utf-8") as f:
            return f.read()

    return ""


def run_hermes_sleep(
    skill_name: str,
    hermes_home: str = "",
    skills_dir: str = "",
    lookback_hours: int = 72,
    max_sessions: int = 15,
    max_tasks: int = 30,
    edit_budget: int = 4,
    backend: str = "mock",
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Run one sleep cycle for a Hermes skill.

    Returns a report dict with proposed edits, scores, and staging info.
    """
    # Harvest
    sessions = harvest_hermes_sessions(
        hermes_home=hermes_home,
        lookback_hours=lookback_hours,
        max_sessions=max_sessions,
    )

    if not sessions:
        return {"error": "no_sessions", "message": "No recent Hermes sessions found."}

    # Mine
    tasks = mine_hermes_tasks(sessions, skill_name=skill_name, max_tasks=max_tasks)

    if not tasks:
        return {"error": "no_tasks", "message": "No tasks could be mined from sessions."}

    # Assign train/val splits
    tasks = assign_splits(tasks, holdout_fraction=0.30, seed=42)

    # Load current skill
    current_skill = load_hermes_skill(skill_name, skills_dir=skills_dir)
    if not current_skill:
        return {"error": "skill_not_found",
                "message": f"Skill '{skill_name}' not found in {skills_dir}"}

    # Get backend - use Hermes-specific backend for real failure analysis
    if backend == "mock":
        from hermes_skillopt.backend import HermesBackend
        be = HermesBackend()
    else:
        be = get_backend(backend)

    # Consolidate
    # With mock backend, scores can't change (outcome-derived), so use greedy mode.
    # With real backends (claude/codex), the gate validates actual improvement.
    use_gate = backend not in ("mock", "") and not dry_run
    result = consolidate(
        be, tasks, current_skill, "",
        edit_budget=edit_budget,
        gate_metric="mixed",
        gate_mixed_weight=0.5,
        gate_mode="on" if use_gate else "off",
        evolve_skill=True,
        evolve_memory=False,
        night=1,
    )

    # Build report
    n_train = sum(1 for t in tasks if t.split == "train")
    n_val = sum(1 for t in tasks if t.split in ("val", "holdout"))

    report = {
        "skill_name": skill_name,
        "n_sessions": len(sessions),
        "n_tasks": len(tasks),
        "n_train": n_train,
        "n_val": n_val,
        "baseline_score": round(result.baseline_score, 4),
        "candidate_score": round(result.candidate_score, 4),
        "accepted": result.accepted,
        "gate_action": result.gate_action,
        "edits": [
            {"target": e.target, "op": e.op, "content": e.content, "rationale": e.rationale}
            for e in result.applied_edits
        ],
        "rejected_edits": [
            {"target": e.target, "op": e.op, "content": e.content}
            for e in result.rejected_edits
        ],
        "proposed_skill": result.new_skill if result.accepted else None,
        "tokens_used": be.tokens_used(),
        "session_sample": [
            {"title": s.title, "model": s.model, "prompts": len(s.user_prompts)}
            for s in sessions[:5]
        ],
    }

    return report


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="hermes-sleep",
        description="On-demand SkillOpt-Sleep for Hermes Agent"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # run / dry-run
    for cmd_name in ("run", "dry-run"):
        p = sub.add_parser(cmd_name, help="Run a sleep cycle" if cmd_name == "run" else "Preview only")
        p.add_argument("--skill", required=True, help="Hermes skill name to optimize")
        p.add_argument("--hermes-home", default="", help=f"Hermes config dir (default: {DEFAULT_HERMES_HOME})")
        p.add_argument("--skills-dir", default="", help=f"Hermes skills dir (default: {DEFAULT_SKILLS_DIR})")
        p.add_argument("--lookback-hours", type=int, default=72, help="Hours of session history to harvest")
        p.add_argument("--max-sessions", type=int, default=15)
        p.add_argument("--max-tasks", type=int, default=30)
        p.add_argument("--edit-budget", type=int, default=4, help="Max edits per night (learning rate)")
        p.add_argument("--backend", default="mock", choices=["mock", "claude", "codex"])
        p.add_argument("--json", action="store_true", help="Machine-readable output")

    # status
    p_status = sub.add_parser("status", help="Show available skills and recent sessions")
    p_status.add_argument("--hermes-home", default="")
    p_status.add_argument("--skills-dir", default="")
    p_status.add_argument("--json", action="store_true")

    args = parser.parse_args()

    if args.command == "status":
        return cmd_status(args)

    return cmd_run(args)


def cmd_run(args) -> int:
    dry = (args.command == "dry-run")
    report = run_hermes_sleep(
        skill_name=args.skill,
        hermes_home=args.hermes_home,
        skills_dir=args.skills_dir,
        lookback_hours=args.lookback_hours,
        max_sessions=args.max_sessions,
        max_tasks=args.max_tasks,
        edit_budget=args.edit_budget,
        backend=args.backend,
        dry_run=dry,
    )

    if "error" in report:
        print(f"[hermes-sleep] ERROR: {report['message']}")
        return 1

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0

    print(f"\n{'='*60}")
    print(f"  Hermes SkillOpt-Sleep — {'DRY RUN' if dry else 'NIGHT 1'}")
    print(f"  Skill: {report['skill_name']}")
    print(f"{'='*60}")
    print(f"\n  Sessions harvested: {report['n_sessions']}")
    print(f"  Tasks mined: {report['n_tasks']} ({report['n_train']} train / {report['n_val']} val)")

    if report['session_sample']:
        print("\n  Sampled sessions:")
        for s in report['session_sample']:
            print(f"    • {s['title'][:60]:60s} [{s['model']}] ({s['prompts']} prompts)")

    print(f"\n  Held-out score: {report['baseline_score']:.3f} → {report['candidate_score']:.3f}")
    print(f"  Gate: {report['gate_action'].upper()} (accepted={report['accepted']})")
    print(f"  Tokens used: {report['tokens_used']}")

    if report['edits']:
        print(f"\n  Accepted edits ({len(report['edits'])}):")
        for e in report['edits']:
            print(f"    + [{e['target']}/{e['op']}] {e['content'][:100]}")
            if e['rationale']:
                print(f"      why: {e['rationale'][:120]}")

    if report['rejected_edits']:
        print(f"\n  Rejected by gate ({len(report['rejected_edits'])}):")
        for e in report['rejected_edits']:
            print(f"    - [{e['target']}/{e['op']}] {e['content'][:100]}")

    if report['accepted'] and not dry:
        print("\n  Proposed skill written. Review with:")
        print(f"    diff <(cat hermes skills/{report['skill_name']}/SKILL.md) <(echo 'proposed')")
        print("  To adopt: manually review and patch the skill file.")

    print()
    return 0


def cmd_status(args) -> int:
    hermes_home = args.hermes_home or _resolve_hermes_home()
    skills_dir = args.skills_dir or DEFAULT_SKILLS_DIR

    # List skills
    import glob
    skill_files = glob.glob(os.path.join(skills_dir, "**", "SKILL.md"), recursive=True)
    skills = []
    for sf in skill_files:
        name = os.path.basename(os.path.dirname(sf))
        if name and name != os.path.basename(skills_dir):
            size = os.path.getsize(sf)
            skills.append({"name": name, "path": sf, "size": size})

    # Recent sessions
    sessions = harvest_hermes_sessions(
        hermes_home=hermes_home, lookback_hours=168, max_sessions=10
    )

    if args.json:
        print(json.dumps({
            "hermes_home": hermes_home,
            "skills_dir": skills_dir,
            "skills": skills,
            "recent_sessions": [
                {"id": s.session_id[:12], "title": s.title, "model": s.model,
                 "prompts": len(s.user_prompts), "skills": s.skills_loaded[:10]}
                for s in sessions
            ],
        }, indent=2, ensure_ascii=False))
        return 0

    print("\n  Hermes SkillOpt-Sleep Status")
    print(f"  Hermes home: {hermes_home}")
    print(f"  Skills dir: {skills_dir}")
    print(f"\n  Skills available ({len(skills)}):")
    for sk in sorted(skills, key=lambda x: x['name']):
        print(f"    • {sk['name']:40s} ({sk['size']:>6d} bytes)")

    print(f"\n  Recent sessions ({len(sessions)}):")
    for s in sessions:
        skills_str = ", ".join(s.skills_loaded[:5]) or "(none detected)"
        print(f"    • {s.title[:55]:55s} [{s.model}] skills: {skills_str}")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
