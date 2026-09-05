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

The bar is `good > baseline > bad`. The run of 2026-09-05 clears it at +0.375 and
-0.500. See "What filling it measured" in the top-level README for the four runs
that got there, and for the one contaminated response that made the third look
like a regression.
