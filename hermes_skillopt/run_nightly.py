"""
Hermes SkillOpt-Sleep — On-Demand Runner

Runs the sleep cycle for recently-used Hermes skills and stages
reviewable improvements. Invoke it explicitly when optimization is wanted.

Usage:
    python -m hermes_skillopt.run_nightly           # all recently-used skills
    python -m hermes_skillopt.run_nightly --dry-run # preview only
    python -m hermes_skillopt.run_nightly --skill ai-mentor  # single skill
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from contextlib import closing
from typing import Dict, List, Optional

from hermes_skillopt.sleep import _resolve_hermes_home, connect_readonly, enable_utf8_output

# Resolved from HERMES_HOME / the platform default rather than a frozen Windows
# path, so the staging root follows the same install the sessions come from.
HERMES_HOME = _resolve_hermes_home()
STAGING_ROOT = os.path.join(HERMES_HOME, "skillopt-staging")


def recently_used_skills(hermes_home: str = "", lookback_hours: int = 72, max_skills: int = 20) -> List[str]:
    """Find skills actually loaded through successful ``skill_view`` calls."""
    db_path = os.path.join(hermes_home or _resolve_hermes_home(), "state.db")
    if not os.path.exists(db_path):
        return []

    cutoff = time.time() - (lookback_hours * 3600)
    with closing(connect_readonly(db_path)) as conn:
        rows = conn.execute(
            """
            SELECT m.content
            FROM sessions AS s
            JOIN messages AS m ON m.session_id = s.id
            WHERE s.started_at > ?
              AND COALESCE(s.archived, 0) = 0
              AND m.active = 1
              AND m.role = 'tool'
              AND m.tool_name = 'skill_view'
            ORDER BY s.started_at DESC, m.id
            """,
            (cutoff,),
        ).fetchall()

    skill_counts: Dict[str, int] = {}
    first_seen: Dict[str, int] = {}
    for index, (content,) in enumerate(rows):
        try:
            payload = json.loads(content or "{}")
        except (TypeError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict) or payload.get("success") is False:
            continue
        skill = payload.get("name")
        if not isinstance(skill, str) or not skill.strip():
            continue
        skill = skill.strip()
        skill_counts[skill] = skill_counts.get(skill, 0) + 1
        first_seen.setdefault(skill, index)

    sorted_skills = sorted(skill_counts, key=lambda name: (-skill_counts[name], first_seen[name]))
    return sorted_skills[:max_skills]


def ensure_staging_dir() -> str:
    os.makedirs(STAGING_ROOT, exist_ok=True)
    return STAGING_ROOT


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_staging_report(
    skill_name: str,
    report: dict,
    proposed_skill: Optional[str],
    live_skill_path: str,
    staging_root: str = STAGING_ROOT,
) -> str:
    """Write one skill's sleep report to staging. Returns staging path."""
    ts = time.strftime("%Y%m%d-%H%M%S")
    out = os.path.join(staging_root, f"{ts}-{skill_name}")
    os.makedirs(out, exist_ok=True)

    # Machine-readable report
    with open(os.path.join(out, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # Human-readable summary
    accepted = report.get("accepted", False)
    edits = report.get("edits", [])
    rejected = report.get("rejected_edits", [])

    lines = [
        f"# SkillOpt-Sleep: {skill_name}",
        "",
        f"- **Night:** {report.get('night', 1)}",
        f"- **Sessions:** {report.get('n_sessions', 0)}",
        f"- **Tasks:** {report.get('n_tasks', 0)} ({report.get('n_train', 0)} train / {report.get('n_val', 0)} val)",
        f"- **Score:** {report.get('baseline_score', 0):.3f} → {report.get('candidate_score', 0):.3f}",
        f"- **Gate:** {report.get('gate_action', 'unknown')}",
        f"- **Accepted:** {accepted}",
        "",
    ]

    if edits:
        lines.append("## Accepted Edits")
        for e in edits:
            lines.append(f"- **[{e['target']}/{e['op']}]** {e['content'][:120]}")
            if e.get('rationale'):
                lines.append(f"  > {e['rationale'][:150]}")
        lines.append("")

    if rejected:
        lines.append("## Rejected by Gate")
        for e in rejected:
            lines.append(f"- ~~[{e['target']}/{e['op']}]~~ {e['content'][:120]}")
        lines.append("")

    if report.get("session_sample"):
        lines.append("## Sessions Analyzed")
        for s in report["session_sample"][:5]:
            lines.append(f"- {s['title'][:70]} ({s.get('prompts', '?')} prompts)")
        lines.append("")

    lines.extend([
        "---",
        "",
        f"**To adopt:** run `python -m hermes_skillopt.run_nightly --adopt {skill_name}`",
        f"**To discard:** delete `{out}`",
        f"**Live skill:** `{live_skill_path}`",
    ])

    with open(os.path.join(out, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # Proposed skill (if accepted)
    if proposed_skill and accepted:
        with open(os.path.join(out, "proposed_SKILL.md"), "w", encoding="utf-8") as f:
            f.write(proposed_skill)

    # Manifest
    manifest = {
        "skill_name": skill_name,
        "live_skill_path": live_skill_path,
        "has_proposed": bool(proposed_skill and accepted),
        "accepted": accepted,
        "status": "staged",
        "live_skill_sha256": _sha256_file(live_skill_path) if os.path.isfile(live_skill_path) else "",
        "proposed_skill_sha256": hashlib.sha256((proposed_skill or "").encode("utf-8")).hexdigest(),
        "staged_at": ts,
    }
    with open(os.path.join(out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    return out


def adopt_skill(skill_name: str, staging_dir: str = "", staging_root: str = STAGING_ROOT) -> bool:
    """Safely apply the newest exact staged proposal for ``skill_name``."""
    if not staging_dir:
        candidates = []
        if os.path.isdir(staging_root):
            for name in os.listdir(staging_root):
                candidate = os.path.join(staging_root, name)
                manifest_path = os.path.join(candidate, "manifest.json")
                if not os.path.isfile(manifest_path):
                    continue
                try:
                    with open(manifest_path, encoding="utf-8") as handle:
                        manifest = json.load(handle)
                except (OSError, json.JSONDecodeError):
                    continue
                if manifest.get("skill_name") == skill_name and manifest.get("status", "staged") == "staged":
                    candidates.append(candidate)
        if not candidates:
            print(f"[skillopt-sleep] No staged proposals for {skill_name}")
            return False
        staging_dir = max(candidates, key=os.path.getmtime)

    manifest_path = os.path.join(staging_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        print(f"[skillopt-sleep] No manifest in {staging_dir}")
        return False

    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)

    if manifest.get("skill_name") != skill_name:
        print(f"[skillopt-sleep] Manifest is for {manifest.get('skill_name')}, not {skill_name}")
        return False
    if manifest.get("status", "staged") != "staged":
        print(f"[skillopt-sleep] Proposal for {skill_name} is already {manifest.get('status')}")
        return False
    if not manifest.get("has_proposed"):
        print(f"[skillopt-sleep] No proposed skill to adopt for {skill_name}")
        return False

    live_path = manifest["live_skill_path"]
    proposed_path = os.path.join(staging_dir, "proposed_SKILL.md")
    expected_live_hash = manifest.get("live_skill_sha256", "")
    expected_proposed_hash = manifest.get("proposed_skill_sha256", "")

    if not os.path.isfile(live_path) or not os.path.isfile(proposed_path):
        print(f"[skillopt-sleep] Live or proposed skill file is missing for {skill_name}")
        return False
    if not expected_live_hash or not expected_proposed_hash:
        print(f"[skillopt-sleep] Legacy proposal for {skill_name} lacks safety hashes; rerun SkillOpt")
        return False
    if _sha256_file(live_path) != expected_live_hash:
        print(f"[skillopt-sleep] Live skill changed after staging; refusing to overwrite {live_path}")
        return False
    if _sha256_file(proposed_path) != expected_proposed_hash:
        print(f"[skillopt-sleep] Proposed skill hash mismatch; refusing {proposed_path}")
        return False

    import shutil

    backup_dir = os.path.join(staging_dir, "backup")
    os.makedirs(backup_dir, exist_ok=True)
    shutil.copy2(live_path, os.path.join(backup_dir, os.path.basename(live_path)))
    shutil.copy2(proposed_path, live_path)

    manifest["status"] = "adopted"
    manifest["adopted_at"] = time.strftime("%Y%m%d-%H%M%S")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    print(f"[skillopt-sleep] Adopted: {proposed_path} → {live_path}")
    return True


def adopt_staging_dirs(staging_dirs: List[str]) -> int:
    """Adopt exactly the supplied staging directories, never older proposals."""
    adopted = 0
    for staging_dir in staging_dirs:
        manifest_path = os.path.join(staging_dir, "manifest.json")
        if not os.path.isfile(manifest_path):
            continue
        try:
            with open(manifest_path, encoding="utf-8") as handle:
                manifest = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        skill_name = manifest.get("skill_name")
        if isinstance(skill_name, str) and adopt_skill(skill_name, staging_dir=staging_dir):
            adopted += 1
    return adopted


def adopt_all_staged(staging_root: str = STAGING_ROOT) -> int:
    """Adopt every valid, still-staged proposal under ``staging_root``."""
    if not os.path.isdir(staging_root):
        return 0
    staging_dirs = [os.path.join(staging_root, name) for name in os.listdir(staging_root)]
    return adopt_staging_dirs(staging_dirs)


def run_nightly(
    hermes_home: str = "",
    skills: Optional[List[str]] = None,
    lookback_hours: int = 72,
    max_sessions: int = 15,
    max_tasks: int = 30,
    edit_budget: int = 4,
    backend: str = "mock",
    dry_run: bool = False,
    skills_dir: str = "",
    staging_root: str = "",
    agent_path: str = "",
    model: str = "",
    judge_mode: str = "absolute",
) -> List[dict]:
    """Run sleep cycle for specified skills (or auto-detect recently used ones)."""
    from hermes_skillopt.sleep import run_hermes_sleep

    # Resolved per call, not bound at import, so a caller can redirect staging.
    staging_root = staging_root or STAGING_ROOT

    if skills is None:
        skills = recently_used_skills(hermes_home, lookback_hours)
        print(f"[skillopt-sleep] Auto-detected {len(skills)} recently-used skills")
    else:
        print(f"[skillopt-sleep] Running for {len(skills)} specified skills")

    results = []
    for skill_name in skills:
        print(f"\n  [{skill_name}] ", end="", flush=True)

        try:
            report = run_hermes_sleep(
                skill_name=skill_name,
                hermes_home=hermes_home,
                skills_dir=skills_dir,
                lookback_hours=lookback_hours,
                max_sessions=max_sessions,
                max_tasks=max_tasks,
                edit_budget=edit_budget,
                backend=backend,
                dry_run=dry_run,
                agent_path=agent_path,
                model=model,
                judge_mode=judge_mode,
            )
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({"skill": skill_name, "error": str(e)})
            continue

        if "error" in report:
            print(f"SKIP: {report['message']}")
            results.append({"skill": skill_name, "skipped": report["message"]})
            continue

        accepted = report.get("accepted", False)
        n_edits = len(report.get("edits", []))
        score = f"{report.get('baseline_score', 0):.2f}→{report.get('candidate_score', 0):.2f}"

        if n_edits:
            print(f"{n_edits} edit(s) [{score}] {'ACCEPTED' if accepted else 'REJECTED'}")
        else:
            print(f"no changes [{score}]")

        # Stage the proposal
        if not dry_run and report.get("edits") and report.get("accepted"):
            # Use the path the optimizer actually read. Re-resolving it here once
            # allowed the proposal to be staged against a different file than the
            # one it was derived from — the hashes would still verify, because each
            # was taken from a different file.
            live_path = report.get("live_skill_path", "")
            if not live_path:
                print("no live path; not staged")
                results.append({"skill": skill_name, "skipped": "unresolved live skill path"})
                continue

            staging_path = write_staging_report(
                skill_name, report,
                report.get("proposed_skill"),
                live_path,
                staging_root,
            )
            print(f"         → staged: {staging_path}")

        results.append(report)

    return results


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    enable_utf8_output()
    parser = argparse.ArgumentParser(
        prog="hermes-skillopt",
        description="On-demand SkillOpt-Sleep for Hermes Agent"
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview without staging")
    parser.add_argument("--skill", action="append", dest="skills",
                        help="Specific skill(s) to optimize (repeatable)")
    parser.add_argument("--lookback-hours", type=int, default=72)
    parser.add_argument("--max-sessions", type=int, default=15)
    parser.add_argument("--max-tasks", type=int, default=30)
    parser.add_argument("--edit-budget", type=int, default=4)
    parser.add_argument("--backend", default="mock", choices=["mock", "hermes", "claude", "codex"],
                        help="mock = offline heuristics (cannot validate); hermes = replay via "
                             "Hermes's configured providers")
    parser.add_argument("--model", default="", help="Override the replay model")
    parser.add_argument("--judge", default="absolute", choices=["absolute", "pairwise"],
                        dest="judge_mode",
                        help="absolute: rate each response 0..1. pairwise: compare each "
                             "response against the baseline (needs a real backend)")
    parser.add_argument("--hermes-agent-path", default="",
                        help="hermes-agent checkout providing agent.auxiliary_client")
    parser.add_argument("--hermes-home", default="")
    parser.add_argument("--skills-dir", default="",
                        help="Skills directory (default: <hermes-home>/skills)")
    parser.add_argument("--adopt", action="append", dest="adopt_skills",
                        help="Adopt staged proposals for skill(s)")
    parser.add_argument("--list-staged", action="store_true", help="List staged proposals")
    parser.add_argument("--adopt-all", action="store_true", help="Adopt all staged proposals")
    parser.add_argument("--run-and-adopt", action="store_true",
                        help="Run pipeline + auto-adopt all results (one-shot)")
    parser.add_argument("--json", action="store_true")

    args = parser.parse_args()

    if args.run_and_adopt:
        before = set(os.listdir(STAGING_ROOT)) if os.path.isdir(STAGING_ROOT) else set()
        ret = cmd_run(args)
        after = set(os.listdir(STAGING_ROOT)) if os.path.isdir(STAGING_ROOT) else set()
        new_dirs = [os.path.join(STAGING_ROOT, name) for name in sorted(after - before)]
        adopted = adopt_staging_dirs(new_dirs)
        print(f"\n[skillopt-sleep] Adopted {adopted} newly staged skill(s)")
        return ret
    if args.list_staged:
        return cmd_list_staged(args)
    if args.adopt_skills:
        return cmd_adopt(args.adopt_skills)
    if args.adopt_all:
        return cmd_adopt_all()

    return cmd_run(args)


def cmd_list_staged(args) -> int:
    """Show what's waiting to be adopted."""
    if not os.path.isdir(STAGING_ROOT):
        print("[skillopt-sleep] No staged proposals.")
        return 0

    staged = sorted(os.listdir(STAGING_ROOT), reverse=True)
    if not staged:
        print("[skillopt-sleep] No staged proposals.")
        return 0

    print("\n  Staged Skill Improvements\n")
    for d in staged:
        path = os.path.join(STAGING_ROOT, d)
        manifest_path = os.path.join(path, "manifest.json")
        report_path = os.path.join(path, "report.md")

        if not os.path.exists(manifest_path):
            continue

        with open(manifest_path, encoding="utf-8") as f:
            m = json.load(f)

        skill = m.get("skill_name", d)
        accepted = "✓" if m.get("accepted") else "✗"
        has_proposal = "ready" if m.get("has_proposed") else "empty"

        # Read score from report
        score_line = ""
        if os.path.exists(report_path):
            with open(report_path, encoding="utf-8") as f:
                for line in f:
                    if "Score:" in line:
                        score_line = line.strip()
                        break

        print(f"  [{accepted}] {skill:35s} {has_proposal:6s}  {score_line}")
        print(f"       adopt: python -m hermes_skillopt.run_nightly --adopt {skill}")

    print()
    return 0


def cmd_adopt(skills: List[str]) -> int:
    """Adopt staged proposals for specific skills."""
    for skill in skills:
        if adopt_skill(skill):
            print(f"  ✓ {skill}")
        else:
            print(f"  ✗ {skill} (no staged proposal or already adopted)")
    return 0


def cmd_adopt_all() -> int:
    """Adopt all valid staged proposals."""
    adopted = adopt_all_staged()
    print(f"\n[skillopt-sleep] Adopted {adopted} skill(s)")
    return 0


def cmd_run(args) -> int:
    results = run_nightly(
        hermes_home=args.hermes_home,
        skills_dir=args.skills_dir,
        skills=args.skills,
        lookback_hours=args.lookback_hours,
        max_sessions=args.max_sessions,
        max_tasks=args.max_tasks,
        edit_budget=args.edit_budget,
        backend=args.backend,
        judge_mode=args.judge_mode,
        agent_path=args.hermes_agent_path,
        model=args.model,
        dry_run=args.dry_run,
    )

    if args.json:
        print(json.dumps([r for r in results if not isinstance(r, dict) or "error" not in r],
                         indent=2, ensure_ascii=False))

    # Summary
    if not args.dry_run:
        staged = [r for r in results if isinstance(r, dict) and r.get("accepted")]
        if staged:
            print(f"\n[skillopt-sleep] {len(staged)} skill(s) staged for review.")
            print("[skillopt-sleep] Review: python -m hermes_skillopt.run_nightly --list-staged")
            print("[skillopt-sleep] Adopt:  python -m hermes_skillopt.run_nightly --adopt-all")

    return 0


if __name__ == "__main__":
    sys.exit(main())
