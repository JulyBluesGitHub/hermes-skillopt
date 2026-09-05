"""Does an embedding tell a turn that is still about the skill from one that drifted?

`active_skills` in sleep.py accumulates and is never cleared, so one `skill_view`
at turn 1 attributes every later turn in that session to that skill. Measured on
`daily-ai-news-digest`, all five mined tasks were AIMentor engineering chat from
sessions that opened as a digest and wandered. The optimizer then wrote rules for
a news skill out of a conversation about retrieval routing, and the gate approved
them.

A turn is the skill's if it is still *about* the skill. This asks whether
all-MiniLM-L6-v2 can score that, and at what threshold.

Ground truth is the turn's own `skill_view`. A turn where the agent loaded the
skill to answer that turn is on-topic by construction, and those are the only
labels the transcripts carry for free. Inherited turns are the population in
question, so they are reported as a distribution rather than scored as errors:
some are honest follow-ups and some are drift, and separating those is the whole
task.

    python experiments/topic_drift.py out.json

CPU, no API calls. Roughly a minute for 200 sessions.
"""

import io
import json
import statistics
import sys
from collections import defaultdict

from hermes_skillopt.sleep import (
    AmbiguousSkillError,
    find_skill_path,
    harvest_hermes_sessions,
)

MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def skill_anchor(skill_name: str) -> str:
    """The text a turn is compared against: what this skill is for.

    Frontmatter `description` plus the Purpose and Trigger sections, if present.
    MiniLM truncates past 256 word pieces, so this stays short deliberately: the
    body of a SKILL.md is procedure, and procedure describes how the work is done
    rather than what the work is about.
    """
    try:
        path = find_skill_path(skill_name)
    except AmbiguousSkillError:
        # Two categories hold a skill of this name. Refusing to guess is correct
        # for an edit; for a measurement the name alone is a usable anchor.
        return skill_name.replace("-", " ")
    if not path:
        return skill_name.replace("-", " ")
    text = io.open(path, encoding="utf-8").read()
    parts = [skill_name.replace("-", " ")]
    for line in text.splitlines():
        if line.startswith("description:"):
            parts.append(line.split(":", 1)[1].strip().strip('"'))
            break
    for heading in ("## Purpose", "## Trigger"):
        if heading in text:
            body = text.split(heading, 1)[1].split("\n## ", 1)[0]
            parts.append(" ".join(body.split())[:400])
    return "\n".join(parts)


def main() -> int:
    from sentence_transformers import SentenceTransformer, util

    model = SentenceTransformer(MODEL)
    sessions = harvest_hermes_sessions(lookback_hours=2880, max_sessions=200)
    print(f"sessions: {len(sessions)}")

    # (skill, prompt, loaded_in_this_turn)
    rows = []
    for session in sessions:
        for record in session.prompt_records:
            for skill in record.skills_loaded:
                in_turn = "skill_view" in record.tools_used
                rows.append((skill, record.prompt, in_turn))
    print(f"attributed turns: {len(rows)}")

    skills = sorted({s for s, _p, _i in rows})
    anchors = {s: skill_anchor(s) for s in skills}
    anchor_vecs = dict(zip(skills, model.encode([anchors[s] for s in skills],
                                                normalize_embeddings=True)))
    prompt_vecs = model.encode([p[:2000] for _s, p, _i in rows],
                               normalize_embeddings=True, batch_size=64)

    scored = []
    for (skill, prompt, in_turn), vec in zip(rows, prompt_vecs):
        sim = float(util.cos_sim(vec, anchor_vecs[skill])[0][0])
        scored.append({"skill": skill, "prompt": prompt[:120], "in_turn": in_turn,
                       "sim": round(sim, 4)})

    pos = [r["sim"] for r in scored if r["in_turn"]]
    inh = [r["sim"] for r in scored if not r["in_turn"]]
    print(f"\nloaded in-turn (on-topic by construction): n={len(pos)} "
          f"median={statistics.median(pos):.3f} mean={statistics.fmean(pos):.3f}")
    print(f"inherited (the population in question):     n={len(inh)} "
          f"median={statistics.median(inh):.3f} mean={statistics.fmean(inh):.3f}")

    print("\n=== what a threshold would keep ===")
    print(f"{'thr':>5s} {'in-turn kept':>13s} {'inherited kept':>15s}")
    for thr in (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40):
        kp = sum(s >= thr for s in pos) / len(pos)
        ki = sum(s >= thr for s in inh) / len(inh)
        print(f"{thr:5.2f} {kp:12.1%} {ki:14.1%}")

    # The five tasks the real run actually optimized daily-ai-news-digest on.
    print("\n=== the mined daily-ai-news-digest tasks ===")
    from hermes_skillopt.sleep import mine_hermes_tasks
    tasks = mine_hermes_tasks(sessions, skill_name="daily-ai-news-digest", max_tasks=5)
    tvecs = model.encode([t.intent[:2000] for t in tasks], normalize_embeddings=True)
    anchor = anchor_vecs["daily-ai-news-digest"]
    for task, vec in zip(tasks, tvecs):
        sim = float(util.cos_sim(vec, anchor)[0][0])
        print(f"  {sim:.3f}  {task.intent[:78]!r}")

    print("\n=== per-skill in-turn medians (is the anchor sane per skill?) ===")
    by_skill = defaultdict(list)
    for r in scored:
        if r["in_turn"]:
            by_skill[r["skill"]].append(r["sim"])
    common = sorted(by_skill.items(), key=lambda kv: -len(kv[1]))[:12]
    for skill, sims in common:
        print(f"  {skill:34s} n={len(sims):4d} median={statistics.median(sims):.3f}")

    with io.open(sys.argv[1] if len(sys.argv) > 1 else "topic_drift.json",
                 "w", encoding="utf-8") as fh:
        json.dump(scored, fh, indent=1, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
