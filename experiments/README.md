# Judge experiments

Scripts that measure whether the judge can tell a good skill edit from a bad one.
They call real providers and cost real tokens. They are not part of the test
suite.

## `judge_pairwise.py`

```bash
python experiments/judge_pairwise.py results.json attempts-cache.json
```

Mines four tasks from one skill, builds three variants of it (unchanged, a
deliberately helpful edit, a deliberately harmful one), attempts each, then
scores the same responses with both judges. Holding the responses fixed is the
point, because any difference in separation is then the judge and not the
sampler.

The script checkpoints attempts to the cache file as they complete. They are the
expensive half at roughly two minutes per call, and re-running the judge should
never mean re-sampling the answers.

**Pass a fresh cache path after changing anything upstream of the prompt.** The
cache key is `sha256(task.intent)`, not the full prompt, so a changed excerpt,
budget or delimiter leaves the key identical and an old cache file quietly serves
responses sampled under the old code. Task ids are positional and renumber when
the mining filter changes, which is why they are not the key either.

A change to the **rubric** is the exception, and reusing the cache is then the
right call rather than a shortcut. Upstream's attempt prompt is built from the
skill, the memory, `task.intent` and `task.context_excerpt`, and never reads
`task.reference`, so a rubric edit cannot reach the answers. Serving them from
cache is what makes the judge the only variable, and it turns a 25-minute run
into a few minutes of scoring. Confirm the mined intents still hash-match the
cache keys before trusting a reused file.

The bar is `good > baseline > bad`. The run of 2026-09-05 clears it at +0.375 and
-0.500. See "What filling it measured" in the top-level README for the four runs
that got there, and for the one contaminated response that made the third look
like a regression.

## `judge_grounding.py`

```bash
python experiments/judge_grounding.py results.json
```

A regression probe for one defect: the rubric asked for an answer that is direct
and states its findings, and never asked for one that is supported. A wrong
four-character `"Yes."` beat an honest "Not yet, I can't verify that from this
session" in both judges and both orders.

Four hand-written pairs, judged in both orders under the requirements as they
were and as they are, so a single run is the before-and-after. Sixteen
comparisons at roughly 17k tokens, which is cheap enough to re-run after any edit
to the rubric or the comparison instructions.

The pairs are fixed rather than mined because this measures the instructions, and
mined answers vary too much to isolate them. Two of the four are over-correction
guards. Respecting "I can't verify that" is progress only while an answer still
beats a plan, and while an unnecessary limit still loses to the answer the
request already contains.

The bar is that the right answer wins all four. Before the grounding
requirement it won three; the confident guess took the fourth in both orders.

## `topic_drift.py`

```bash
python experiments/topic_drift.py out.json
```

Scores every attributed turn against its skill's anchor and reports what a
threshold would keep. CPU only, no API calls, about a minute for 200 sessions.

Ground truth is the turn's own `skill_view`: a turn the agent loaded the skill to
answer is on-topic by construction, and those are the only labels the transcripts
carry for free. Inherited turns are the population in question and are reported
as a distribution rather than scored as errors, since some are honest follow-ups
and some are drift.

Run it before changing `DEFAULT_FRACTION` or the anchor in
`hermes_skillopt/topic.py`. It is also what shows that a global threshold cannot
work: per-skill in-turn medians ran from 0.150 to 0.342 under one scorer.

Set `HERMES_SKILLOPT_TOPIC_SCORER=lexical` to compare the fallback scorer against
the embedding on the same turns.
