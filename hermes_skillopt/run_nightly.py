"""
Hermes SkillOpt-Sleep — Nightly Runner

Runs the sleep cycle for recently-used Hermes skills, staging any
validated improvements for morning review. Designed to be called from cron.

Usage:
    python -m hermes_skillopt.run_nightly           # all recently-used skills
    python -m hermes_skillopt.run_nightly --dry-run # preview only
    python -m hermes_skillopt.run_nightly --skill ai-mentor  # single skill
"""

from __future__ import annotations

import json
import os
import sys
import time
import argparse
from typing import Dict, List, Optional

# Add repo root
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO_ROOT)

HERMES_HOME = os.path.expanduser("~/AppData/Local/hermes")
STAGING_ROOT = os.path.join(HERMES_HOME, "skillopt-staging")


def recently_used_skills(hermes_home: str = "", lookback_hours: int = 72, max_skills: int = 20) -> List[str]:
    """Find which skills were loaded in recent Hermes sessions."""
    import sqlite3
    import re

    db_path = os.path.join(hermes_home or HERMES_HOME, "state.db")
    if not os.path.exists(db_path):
        return []

    conn = sqlite3.connect(db_path)
    cutoff = time.time() - (lookback_hours * 3600)

    rows = conn.execute("""
        SELECT system_prompt FROM sessions
        WHERE started_at > ? AND system_prompt IS NOT NULL
        ORDER BY started_at DESC
    """, (cutoff,)).fetchall()

    skill_counts: Dict[str, int] = {}
    for (sp,) in rows:
        if not sp:
            continue
        found = re.findall(r'- (\S+):', sp)
        for skill in found:
            skill_counts[skill] = skill_counts.get(skill, 0) + 1

    conn.close()

    # Sort by frequency, most-used first
    sorted_skills = sorted(skill_counts.items(), key=lambda x: -x[1])
    return [s for s, _ in sorted_skills[:max_skills]]


def ensure_staging_dir() -> str:
    os.makedirs(STAGING_ROOT, exist_ok=True)
    return STAGING_ROOT


def write_staging_report(
    skill_name: str,
    report: dict,
    proposed_skill: Optional[str],
    live_skill_path: str,
) -> str:
    """Write one skill's sleep report to staging. Returns staging path."""
    ts = time.strftime("%Y%m%d-%H%M%S")
    out = os.path.join(STAGING_ROOT, f"{ts}-{skill_name}")
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
        "staged_at": ts,
    }
    with open(os.path.join(out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    return out


def adopt_skill(skill_name: str, staging_dir: str = "") -> bool:
    """Apply a staged skill improvement to the live skill file.

    Backs up the original first, then copies the proposed version over it.
    """
    if not staging_dir:
        # Find the latest staging for this skill
        candidates = []
        for d in os.listdir(STAGING_ROOT):
            if d.endswith(f"-{skill_name}"):
                candidates.append(os.path.join(STAGING_ROOT, d))
        if not candidates:
            print(f"[skillopt-sleep] No staged proposals for {skill_name}")
            return False
        staging_dir = max(candidates, key=os.path.getmtime)

    manifest_path = os.path.join(staging_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        print(f"[skillopt-sleep] No manifest in {staging_dir}")
        return False

    with open(manifest_path) as f:
        manifest = json.load(f)

    if not manifest.get("has_proposed"):
        print(f"[skillopt-sleep] No proposed skill to adopt for {skill_name}")
        return False

    live_path = manifest["live_skill_path"]
    proposed_path = os.path.join(staging_dir, "proposed_SKILL.md")

    if not os.path.exists(proposed_path):
        print(f"[skillopt-sleep] Proposed skill file missing: {proposed_path}")
        return False

    # Backup
    import shutil
    backup_dir = os.path.join(staging_dir, "backup")
    os.makedirs(backup_dir, exist_ok=True)
    if os.path.exists(live_path):
        shutil.copy2(live_path, os.path.join(backup_dir, os.path.basename(live_path)))
        print(f"[skillopt-sleep] Backed up: {live_path} → {backup_dir}")

    # Apply
    shutil.copy2(proposed_path, live_path)
    print(f"[skillopt-sleep] Adopted: {proposed_path} → {live_path}")
    return True


def run_nightly(
    hermes_home: str = "",
    skills: Optional[List[str]] = None,
    lookback_hours: int = 72,
    max_sessions: int = 15,
    max_tasks: int = 30,
    edit_budget: int = 4,
    backend: str = "mock",
    dry_run: bool = False,
) -> List[dict]:
    """Run sleep cycle for specified skills (or auto-detect recently used ones)."""
    from hermes_skillopt.sleep import run_hermes_sleep, load_hermes_skill

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
                lookback_hours=lookback_hours,
                max_sessions=max_sessions,
                max_tasks=max_tasks,
                edit_budget=edit_budget,
                backend=backend,
                dry_run=dry_run,
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
            skill_path = load_hermes_skill(skill_name)  # to find the live path
            # Reconstruct the actual live path
            import glob
            pattern = os.path.join(hermes_home or HERMES_HOME, "skills", "**", skill_name, "SKILL.md")
            matches = glob.glob(pattern, recursive=True)
            live_path = matches[0] if matches else ""

            staging_path = write_staging_report(
                skill_name, report,
                report.get("proposed_skill"),
                live_path,
            )
            print(f"         → staged: {staging_path}")

        results.append(report)

    return results


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="hermes-skillopt-nightly",
        description="Nightly SkillOpt-Sleep for Hermes Agent"
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview without staging")
    parser.add_argument("--skill", action="append", dest="skills",
                        help="Specific skill(s) to optimize (repeatable)")
    parser.add_argument("--lookback-hours", type=int, default=72)
    parser.add_argument("--max-sessions", type=int, default=15)
    parser.add_argument("--max-tasks", type=int, default=30)
    parser.add_argument("--edit-budget", type=int, default=4)
    parser.add_argument("--backend", default="mock", choices=["mock", "claude", "codex"])
    parser.add_argument("--hermes-home", default="")
    parser.add_argument("--adopt", action="append", dest="adopt_skills",
                        help="Adopt staged proposals for skill(s)")
    parser.add_argument("--list-staged", action="store_true", help="List staged proposals")
    parser.add_argument("--adopt-all", action="store_true", help="Adopt all staged proposals")
    parser.add_argument("--run-and-adopt", action="store_true",
                        help="Run pipeline + auto-adopt all results (one-shot)")
    parser.add_argument("--json", action="store_true")

    args = parser.parse_args()

    if args.run_and_adopt:
        # Run pipeline first
        ret = cmd_run(args)
        # Then adopt everything that was staged
        cmd_adopt_all()
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

        with open(manifest_path) as f:
            m = json.load(f)

        skill = m.get("skill_name", d)
        accepted = "✓" if m.get("accepted") else "✗"
        has_proposal = "ready" if m.get("has_proposed") else "empty"

        # Read score from report
        score_line = ""
        if os.path.exists(report_path):
            with open(report_path) as f:
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
    """Adopt all staged proposals."""
    if not os.path.isdir(STAGING_ROOT):
        print("[skillopt-sleep] No staged proposals.")
        return 0

    adopted = 0
    for d in os.listdir(STAGING_ROOT):
        parts = d.rsplit("-", 1)
        if len(parts) == 2:
            skill = parts[1]
            if adopt_skill(skill):
                adopted += 1

    print(f"\n[skillopt-sleep] Adopted {adopted} skill(s)")
    return 0


def cmd_run(args) -> int:
    results = run_nightly(
        hermes_home=args.hermes_home,
        skills=args.skills,
        lookback_hours=args.lookback_hours,
        max_sessions=args.max_sessions,
        max_tasks=args.max_tasks,
        edit_budget=args.edit_budget,
        backend=args.backend,
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
            print(f"[skillopt-sleep] Review: python -m hermes_skillopt.run_nightly --list-staged")
            print(f"[skillopt-sleep] Adopt:  python -m hermes_skillopt.run_nightly --adopt-all")

    return 0


if __name__ == "__main__":
    sys.exit(main())
