# Judge experiments

Scripts that measure whether the judge can tell a good skill edit from a bad one.
They call real providers and cost real tokens; they are not part of the test suite.

## `judge_pairwise.py`

```bash
python experiments/judge_pairwise.py results.json attempts-cache.json
```

Mines four tasks from one skill, builds three variants of it (unchanged, a
deliberately helpful edit, a deliberately harmful one), attempts each, then scores
**the same responses with both judges**. Holding the responses fixed is the point:
any difference in separation is the judge and not the sampler.

Attempts are checkpointed to the cache file as they complete — they are the
expensive half at roughly two minutes per call, and re-running the judge should
never mean re-sampling the answers.

The bar is `good > baseline > bad`. See the "Measured" section of the top-level
README for the result, and the limitation it exposed in the rubric.
