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
import time
from contextlib import closing
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

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


class AmbiguousSkillError(RuntimeError):
    """A skill name resolves to more than one SKILL.md."""


def enable_utf8_output() -> None:
    """Make the CLI's status glyphs printable on legacy consoles.

    Windows terminals default to cp1252, where printing '✓' raises
    UnicodeEncodeError and takes the whole command down.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):  # detached or already-wrapped stream
            pass


def connect_readonly(db_path: str) -> sqlite3.Connection:
    """Open Hermes's live state DB read-only.

    This tool runs against a database a live agent may be writing. A read-only
    URI connection means a bug here can never mutate session history, and it
    keeps the reader out of the way of the writer.
    """
    uri = "file:" + os.path.abspath(db_path).replace("?", "%3f").replace("#", "%23") + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def resolve_skills_dir(skills_dir: str = "", hermes_home: str = "") -> str:
    """Skills directory: explicit override, else ``<resolved hermes home>/skills``.

    Derived from the resolved home rather than a frozen constant so ``HERMES_HOME``
    and ``--hermes-home`` are honored on every platform, not just the Windows default.
    """
    if skills_dir:
        return skills_dir
    return os.path.join(hermes_home or _resolve_hermes_home(), "skills")


def find_skill_path(skill_name: str, skills_dir: str = "", hermes_home: str = "") -> str:
    """Absolute path of ``skill_name``'s markdown file, or "" when it does not exist.

    Single source of truth for skill location. The optimizer (which *reads* the
    skill) and the stager (which records the adoption target and its hash) must
    agree on one path: if they resolve differently, a proposal derived from file
    A can be adopted onto file B while the safety hashes still verify, because
    each was computed against a different file.

    ``**`` already matches zero directories, so the nested-category glob covers
    the flat ``<skills_dir>/<name>/SKILL.md`` layout too; the bare ``<name>.md``
    file is the only separate fallback.
    """
    import glob

    base = resolve_skills_dir(skills_dir, hermes_home)
    matches = sorted(set(glob.glob(os.path.join(base, "**", skill_name, "SKILL.md"), recursive=True)))
    if len(matches) > 1:
        raise AmbiguousSkillError(
            f"Skill '{skill_name}' matches {len(matches)} files under {base}: "
            + ", ".join(matches)
            + ". Pass --skills-dir to disambiguate; refusing to guess which one to edit."
        )
    if matches:
        return matches[0]
    flat_file = os.path.join(base, f"{skill_name}.md")
    return flat_file if os.path.isfile(flat_file) else ""


# ── Stage 1: Harvest Hermes Sessions ─────────────────────────────────────────

@dataclass
class PromptRecord:
    """One user turn and the evidence used to label its outcome."""

    prompt: str
    response: str = ""
    outcome: str = "unknown"
    tool_errors: List[str] = field(default_factory=list)
    skills_loaded: List[str] = field(default_factory=list)
    #: Every tool this turn called, in first-use order. Replay is a bare text
    #: call, so this is what says whether replaying the turn is a fair test.
    tools_used: List[str] = field(default_factory=list)


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

    cutoff = time.time() - (lookback_hours * 3600)
    with closing(connect_readonly(db_path)) as conn:
        rows = conn.execute("""
            SELECT id, title, cwd, model, started_at, ended_at,
                   message_count, tool_call_count
            FROM sessions
            WHERE started_at > ? AND archived = 0
            ORDER BY started_at DESC
            LIMIT ?
        """, (cutoff, max_sessions)).fetchall()
        return [_build_session(conn, row) for row in rows]


def _build_session(conn: sqlite3.Connection, row: sqlite3.Row) -> HermesSession:
    """Assemble one session and its per-turn outcome records from the message log."""
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

    # Messages for this session (user + assistant + tool, in order)
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
            if current is not None and tool_name not in current.tools_used:
                current.tools_used.append(tool_name)
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
    return s


# ── Stage 2: Mine Tasks from Hermes Sessions ─────────────────────────────────

#: The user's request is the task; 200 chars routinely cut mid-sentence.
_INTENT_CHARS = 300
#: Rubrics are read by a judge, so they can afford the whole request.
_RUBRIC_CHARS = 800


def build_task_rubric(record: "PromptRecord") -> str:
    """Compose what a good answer to this turn must do.

    Without a rubric the judge has nothing to discriminate on: the mock judge
    falls back to ``1.0 if outcome == "success"``, which is fixed history and
    ignores the response entirely, and the LLM judge is handed a truncated
    prompt with the model name glued to the front.

    Requirements are derived from what actually happened in the turn, not
    boilerplate. The original response is deliberately NOT used as a reference —
    scoring similarity to history would reward reproducing the old answer,
    including when the old answer was the problem.
    """
    lines = [
        "The user asked:",
        record.prompt[:_RUBRIC_CHARS].strip(),
        "",
        "A good response must:",
        "- directly and completely address that request",
        "- state its actual findings, not a description of how it would proceed",
    ]
    if record.tool_errors:
        lines.append(
            "- report any step that fails, with the error, instead of presenting "
            "a result as though it had succeeded"
        )
    if record.outcome == "fail":
        lines.append("- produce a real answer; this request previously went unanswered")
    return "\n".join(lines)

#: Tools whose effect a bare text replay actually reproduces.
#:
#: ``skill_view`` loads a skill, and replay puts the skill straight into the
#: prompt — so a turn that used it is still fairly replayable. It also has to be
#: here for a second reason: skill attribution *requires* a successful
#: ``skill_view``, so counting it as an external dependency would exclude every
#: task this tool can see.
#:
#: Everything else reaches outside the prompt — the filesystem, a shell, a
#: browser, the network, session-local todo state, the user. Replay has none of
#: it, so replaying such a turn asks the model a different question than the one
#: that was actually answered.
REPLAYABLE_TOOLS = frozenset({"skill_view"})


def turn_needs_tools(record: "PromptRecord") -> List[str]:
    """Tools this turn used that replay cannot supply; empty means fairly replayable."""
    return [t for t in record.tools_used if t not in REPLAYABLE_TOOLS]


def mine_hermes_tasks(
    sessions: List[HermesSession],
    skill_name: str = "",
    max_tasks: int = 30,
    *,
    require_replayable: bool = True,
    skipped: Optional[Dict[str, int]] = None,
) -> List[TaskRecord]:
    """Convert Hermes sessions into SkillOpt TaskRecords.

    Each "task" is derived from a user prompt paired with the assistant's
    response and any tool errors that followed.

    Turns that used tools replay cannot supply are skipped by default. Replay is
    a single-shot text call with no tools, so replaying such a turn scores the
    model on a different task than the one that happened: it cannot fetch, read
    or run anything, correctly says so, and the rubric marks that down as
    "describing how it would proceed". The optimizer's way out is to propose
    rules like "do not decline, answer from what you know" — which score well
    in replay and are actively harmful in production, where the agent does have
    tools and the skill may have a fail-closed rule saying not to guess.

    Pass ``require_replayable=False`` to mine them anyway, and ``skipped`` to
    receive a tool-name histogram of what was dropped.
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
            needed = turn_needs_tools(record)
            if require_replayable and needed:
                if skipped is not None:
                    for tool in needed:
                        skipped[tool] = skipped.get(tool, 0) + 1
                continue
            task_id += 1

            task = TaskRecord(
                id=f"hermes-{task_id}",
                project=session.cwd,
                intent=record.prompt[:_INTENT_CHARS],
                context_excerpt=f"Session: {session.title or session.session_id[:12]}\n"
                               f"Model: {session.model}\n"
                               f"Skills loaded: {', '.join(record.skills_loaded[:5])}",
                attempted_solution=record.response[:500],
                outcome=record.outcome,
                reference_kind="rubric",
                reference=build_task_rubric(record),
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

def load_hermes_skill(skill_name: str, skills_dir: str = "", hermes_home: str = "") -> str:
    """Load a Hermes skill .md file, or "" when it does not exist."""
    path = find_skill_path(skill_name, skills_dir=skills_dir, hermes_home=hermes_home)
    if not path:
        return ""
    with open(path, encoding="utf-8") as f:
        return f.read()


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
    agent_path: str = "",
    model: str = "",
    judge_mode: str = "absolute",
    require_replayable: bool = True,
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

    # Mine. Turns that used tools replay cannot supply are dropped by default —
    # see mine_hermes_tasks for why scoring them rewards confident guessing.
    skipped: Dict[str, int] = {}
    tasks = mine_hermes_tasks(
        sessions, skill_name=skill_name, max_tasks=max_tasks,
        require_replayable=require_replayable, skipped=skipped,
    )

    if not tasks:
        n_skipped = sum(skipped.values())
        if n_skipped:
            top = ", ".join(f"{k} ({v})" for k, v in
                            sorted(skipped.items(), key=lambda kv: -kv[1])[:5])
            return {"error": "no_replayable_tasks",
                    "message": f"Every mined turn for '{skill_name}' used tools replay "
                               f"cannot supply (most often: {top}). Replaying them would "
                               f"score the model on a task it cannot perform. Pass "
                               f"--allow-unreplayable to mine them anyway, knowing the "
                               f"scores reward guessing."}
        return {"error": "no_tasks", "message": "No tasks could be mined from sessions."}

    # Assign train/val splits
    tasks = assign_splits(tasks, holdout_fraction=0.30, seed=42)

    # Load current skill. The resolved path travels in the report so the stager
    # records the hash of the very file that was optimized (never a re-resolved one).
    resolved_dir = resolve_skills_dir(skills_dir, hermes_home)
    live_skill_path = find_skill_path(skill_name, skills_dir=skills_dir, hermes_home=hermes_home)
    if not live_skill_path:
        return {"error": "skill_not_found",
                "message": f"Skill '{skill_name}' not found in {resolved_dir}"}
    with open(live_skill_path, encoding="utf-8") as handle:
        current_skill = handle.read()
    if not current_skill:
        return {"error": "skill_empty",
                "message": f"Skill '{skill_name}' is empty at {live_skill_path}"}

    # Get backend. Real backends are wrapped so an empty response raises instead
    # of silently scoring 0.0 — a dead CLI used to be indistinguishable from a
    # candidate that genuinely didn't help.
    from hermes_skillopt.llm_backend import BackendCallError, build_validating_backend

    be = build_validating_backend(
        backend, hermes_home=hermes_home, agent_path=agent_path, model=model,
        judge_mode=judge_mode,
    )
    if backend in ("mock", ""):
        print("[hermes-sleep] WARNING: the mock backend derives scores from recorded "
              "outcomes, not from the skill. It cannot show that an edit helps — "
              "baseline and candidate will always match. Use --backend hermes to validate.",
              file=sys.stderr)
    else:
        # One cheap round-trip: fail in seconds if the backend is dead, rather
        # than after ~30 calls that all return nothing.
        try:
            probe = getattr(be, "probe", None)
            if callable(probe):
                probe()
        except BackendCallError as exc:
            return {"error": "backend_unavailable", "message": str(exc)}

    # Consolidate
    # With mock backend, scores can't change (outcome-derived), so use greedy mode.
    # With real backends, the gate validates actual improvement.
    use_gate = backend not in ("mock", "") and not dry_run
    try:
        result = consolidate(
            be, tasks, current_skill, "",
            edit_budget=edit_budget,
            # A pairwise score is already the comparison the gate wants, and it
            # reports the same value as hard and soft, so every metric agrees
            # today. Naming "soft" pins that: the absolute judge's hard bit is a
            # >=0.8 threshold, and if one is ever reintroduced here the gate must
            # still read the comparison rather than a blend with a threshold.
            gate_metric="soft" if judge_mode == "pairwise" else "mixed",
            gate_mixed_weight=0.5,
            gate_mode="on" if use_gate else "off",
            evolve_skill=True,
            evolve_memory=False,
            night=1,
        )
    except BackendCallError as exc:
        # Nothing is staged: a partial replay cannot support a proposal.
        return {"error": "replay_failed", "message": str(exc)}

    # Build report
    n_train = sum(1 for t in tasks if t.split == "train")
    n_val = sum(1 for t in tasks if t.split in ("val", "holdout"))

    report = {
        "skill_name": skill_name,
        "live_skill_path": live_skill_path,
        "n_sessions": len(sessions),
        "n_tasks": len(tasks),
        "n_train": n_train,
        "n_val": n_val,
        "baseline_score": round(result.baseline_score, 4),
        "candidate_score": round(result.candidate_score, 4),
        "accepted": result.accepted,
        "gate_action": result.gate_action,
        "judge_mode": judge_mode,
        "n_skipped_unreplayable": sum(skipped.values()),
        "skipped_tools": dict(sorted(skipped.items(), key=lambda kv: -kv[1])[:10]),
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
    enable_utf8_output()
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
        p.add_argument("--backend", default="mock", choices=["mock", "hermes", "claude", "codex"],
                       help="mock = offline heuristics (cannot validate); hermes = replay via "
                            "Hermes's configured providers")
        p.add_argument("--model", default="", help="Override the replay model")
        p.add_argument("--hermes-agent-path", default="",
                       help="hermes-agent checkout providing agent.auxiliary_client")
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
        agent_path=args.hermes_agent_path,
        model=args.model,
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
