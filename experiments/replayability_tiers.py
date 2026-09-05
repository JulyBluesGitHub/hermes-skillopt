"""How much of Hermes history can a toolless, historyless replay fairly reproduce?

Replay gets a skill, an intent and a short context excerpt. It gets no tools, and
no conversation. So a mined turn is only a fair test if neither is needed. This
counts three progressively honest definitions of that.

Free and read-only: it opens the session DB and calls no model.

    python experiments/replayability_tiers.py [skill-name]
"""

import sys

from hermes_skillopt.sleep import harvest_hermes_sessions, turn_needs_tools


def tiers(sessions, skill):
    total = own = prefix = first = 0
    for session in sessions:
        if not session.completed:
            continue
        if skill and skill not in session.skills_loaded:
            continue
        seen_blocking = False
        for index, record in enumerate(session.prompt_records):
            needed = turn_needs_tools(record)
            if (not skill) or skill in record.skills_loaded:
                total += 1
                if not needed:
                    own += 1
                    if not seen_blocking:
                        prefix += 1
                        if index == 0:
                            first += 1
            if needed:
                seen_blocking = True
    return total, own, prefix, first


def main() -> int:
    skill = sys.argv[1] if len(sys.argv) > 1 else "daily-ai-news-digest"
    sessions = harvest_hermes_sessions(lookback_hours=2880, max_sessions=200)

    for target in (skill, ""):
        total, own, prefix, first = tiers(sessions, target)
        pct = lambda n: f"{100 * n / max(1, total):.1f}%"  # noqa: E731
        print(f"\n=== {target or '<all skills>'} ===")
        print(f"  attributed turns                       : {total}")
        print(f"  1. own turn used no blocking tool      : {own:5d}  ({pct(own)})")
        print(f"  2. ...and no earlier turn did either   : {prefix:5d}  ({pct(prefix)})")
        print(f"  3. ...and it is the session's 1st turn : {first:5d}  ({pct(first)})")

    print("\nTier 1 is what mine_hermes_tasks enforces today. Tier 2 is what a replay")
    print("with no conversation history would actually need. The gap between them is")
    print("the inherited-context problem: a turn that called no tool itself, but whose")
    print("prerequisites were established by earlier turns that did.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
