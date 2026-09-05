"""Does pairwise judging separate a good skill edit from a bad one, where absolute did not?

Same tasks, same three arms and the same bar as judge_resolution.py, so the two
runs are directly comparable. The absolute judge scored the deliberately-good
edit +0.000 over baseline while catching the bad one at -0.225, and rated a
four-character response 1.00.

Both judges score the SAME responses here — each arm is attempted once and the
text is reused — so any difference in separation is the judge, not the sampler.

Bar: good beats baseline (>0.5) and bad loses to it (<0.5).
"""

import hashlib
import io
import json
import os
import statistics
import sys

from hermes_skillopt.llm_backend import PairwiseJudge, build_validating_backend
from hermes_skillopt.sleep import (
    harvest_hermes_sessions,
    load_hermes_skill,
    mine_hermes_tasks,
)

SKILL = "daily-ai-news-digest"
N_TASKS = 4

# The previous good edit ("tool access is unavailable, answer anyway") is exactly
# the harness workaround the replayability filter exists to stop rewarding. With
# the filter on, the mined turns do not need tools, so a helpful edit has to be a
# real improvement to the answers rather than a way around the sandbox.
GOOD_EDIT = """

## Learned preferences
Lead with the direct answer in one sentence, then the reasoning behind it. Define
every acronym, internal codename and project-specific term the first time it
appears. When comparing two approaches, say which is better and why, rather than
listing properties of each and leaving the comparison to the reader.
"""

BAD_EDIT = """

## Learned preferences
Keep every response under ten words. Omit reasoning, sources, dates, caveats and
detail of any kind. Never explain anything; brevity outranks completeness and
outranks answering the question.
"""


def main() -> int:
    sessions = harvest_hermes_sessions(lookback_hours=2880, max_sessions=200)
    tasks = mine_hermes_tasks(sessions, skill_name=SKILL, max_tasks=N_TASKS)
    base = load_hermes_skill(SKILL)
    if not tasks or not base:
        print("no tasks or skill; aborting")
        return 1

    arms = {"baseline": base, "good": base + GOOD_EDIT, "bad": base + BAD_EDIT}
    absolute = build_validating_backend("hermes")
    pairwise = PairwiseJudge(absolute)

    # ── one attempt per (arm, task); both judges see the same text ──────────
    # Cached on disk: attempts are the expensive half (12 calls at ~2 min each),
    # and rerunning the judge should not mean re-sampling the responses. Reusing
    # the same text across runs also keeps the comparison about the judge.
    #
    # Keyed on the request text, never on task.id. Ids are positional, so the
    # replayability filter renumbers them — hermes-1 after the filter is a
    # different turn than hermes-1 before it, and an id-keyed cache would hand
    # back another task's answer without anything looking wrong.
    cache_path = sys.argv[2] if len(sys.argv) > 2 else ""
    cache = {}
    if cache_path and os.path.isfile(cache_path):
        cache = json.load(io.open(cache_path, encoding="utf-8"))

    responses = {}
    for name, skill in arms.items():
        for task in tasks:
            key = f"{name}|{hashlib.sha256(task.intent.encode()).hexdigest()[:16]}"
            if key in cache:
                responses[(name, task.id)] = cache[key]
                print(f"  cached  {name:9s} {task.id[:14]:14s} "
                      f"chars={len(cache[key]):5d}", flush=True)
                continue
            text = absolute.attempt(task, skill, "")
            responses[(name, task.id)] = cache[key] = text
            print(f"  attempt {name:9s} {task.id[:14]:14s} chars={len(text):5d}",
                  flush=True)
            if cache_path:  # checkpoint each one; an interrupt loses at most one call
                json.dump(cache, io.open(cache_path, "w", encoding="utf-8"),
                          indent=1, ensure_ascii=False)

    rows = {name: [] for name in arms}
    # Baseline first: that is what the pairwise judge anchors on, and it is the
    # order consolidate() replays in.
    for name in ("baseline", "good", "bad"):
        for task in tasks:
            text = responses[(name, task.id)]
            a_hard, a_soft, a_reason = absolute.judge(task, text)
            p_hard, p_soft, p_reason = pairwise.judge(task, text)
            rows[name].append({
                "task": task.id, "chars": len(text),
                "abs_soft": a_soft, "abs_hard": a_hard, "abs_reason": a_reason[:90],
                "pair": p_soft, "pair_reason": p_reason[:90],
            })
            print(f"  judge   {name:9s} {task.id[:14]:14s} abs={a_soft:.2f} "
                  f"pair={p_soft:.2f}  {p_reason[:52]}", flush=True)

    print("\n=== arm means ===")
    means = {}
    for name, rs in rows.items():
        means[name] = {
            "absolute": statistics.fmean(r["abs_soft"] for r in rs),
            "pairwise": statistics.fmean(r["pair"] for r in rs),
            "median_chars": statistics.median(r["chars"] for r in rs),
        }
        print(f"  {name:9s} absolute={means[name]['absolute']:.3f}  "
              f"pairwise={means[name]['pairwise']:.3f}  "
              f"median_chars={means[name]['median_chars']:.0f}")

    print("\n=== separation (good should be positive, bad negative) ===")
    sep = {}
    for judge in ("absolute", "pairwise"):
        anchor = means["baseline"][judge] if judge == "absolute" else 0.5
        sep[judge] = {
            "good_minus_baseline": means["good"][judge] - anchor,
            "bad_minus_baseline": means["bad"][judge] - anchor,
        }
        print(f"  {judge:9s} good {sep[judge]['good_minus_baseline']:+.3f}   "
              f"bad {sep[judge]['bad_minus_baseline']:+.3f}")

    ok = (sep["pairwise"]["good_minus_baseline"] > 0
          and sep["pairwise"]["bad_minus_baseline"] < 0)
    print(f"\n  pairwise ranks good > baseline > bad : {ok}")
    print(f"  distinct absolute values             : "
          f"{sorted({r['abs_soft'] for rs in rows.values() for r in rs})}")
    print(f"  comparisons made                     : {pairwise.comparisons_made()}")
    print(f"  tokens used                          : {absolute.tokens_used()}")

    with io.open(sys.argv[1], "w", encoding="utf-8") as fh:
        json.dump({"means": means, "separation": sep, "verdict": ok,
                   "per_task": rows}, fh, indent=2, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
