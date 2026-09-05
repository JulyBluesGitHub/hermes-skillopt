# Hermes SkillOpt

On-demand skill optimization for Hermes Agent, backed by Microsoft's [SkillOpt](https://github.com/microsoft/SkillOpt) engine.

Hermes SkillOpt reads completed Hermes sessions, identifies the skills that were loaded through `skill_view`, mines task outcomes, and stages bounded additions to each skill's managed `LEARNED` block.

## Why this adapter exists

SkillOpt already provides the optimization loop. This repository supplies the Hermes-specific parts:

- reads Hermes's SQLite session history;
- associates user turns with final assistant responses and tool failures;
- attributes tasks only to the skills loaded in that session;
- stages proposed `SKILL.md` updates outside the live skills directory;
- verifies file hashes before adoption and creates a backup.

It runs on demand. It does not install a daemon or cron job.

## Install

Requires Python 3.10+ and a local Hermes installation with session history.

```bash
git clone https://github.com/JulyBluesGitHub/hermes-skillopt.git
cd hermes-skillopt
python -m pip install -e ".[dev]"
```

That also installs the upstream `skillopt` package as a dependency.

## Usage

### Inspect available history

```bash
hermes-skillopt-status status
# or
python -m hermes_skillopt.sleep status
```

### Preview changes

```bash
hermes-skillopt --dry-run --skill daily-ai-news-digest
```

Dry-run does not write staging files or modify live skills.

### Stage changes for review

```bash
hermes-skillopt --skill daily-ai-news-digest
hermes-skillopt --list-staged
```

### Adopt a reviewed proposal

```bash
hermes-skillopt --adopt daily-ai-news-digest
```

### One-shot run and adoption

```bash
hermes-skillopt --run-and-adopt
```

`--run-and-adopt` adopts only proposals created by that invocation. It does not sweep up older staged proposals.

## Locating skills

Skills resolve from `<hermes-home>/skills` by default, where `<hermes-home>` is
`$HERMES_HOME`, else the platform default. Override either level:

```bash
hermes-skillopt --skill my-skill --hermes-home /srv/hermes
hermes-skillopt --skill my-skill --skills-dir /srv/hermes/skills
```

Both the nested `<category>/<skill>/SKILL.md` and flat `<skill>/SKILL.md`
layouts resolve, plus a bare `<skill>.md` as a last resort.

Two categories can each hold a skill of the same name. If a name matches more
than one file, the run fails with `AmbiguousSkillError` rather than editing
whichever the filesystem returned first. Pass `--skills-dir` to point at the one
you mean.

## Safety model

The optimizer writes every proposal under:

```text
%LOCALAPPDATA%/hermes/skillopt-staging/<timestamp>-<skill>/
```

A staging directory contains:

- `report.json` and `report.md`;
- `proposed_SKILL.md`;
- `manifest.json` with hashes of the live and proposed files;
- `backup/` after adoption.

Adoption fails closed when:

- the manifest's skill name does not match the requested skill;
- the proposal was already adopted;
- the live skill changed after staging;
- the proposed file changed after staging;
- a legacy manifest lacks safety hashes.

Edits stay inside the managed block:

```markdown
<!-- SKILLOPT-SLEEP:LEARNED START -->
...
<!-- SKILLOPT-SLEEP:LEARNED END -->
```

SkillOpt's edit applicator preserves hand-written content outside this block.

The report carries the exact path the optimizer read, instead of re-deriving it
at staging time. Resolving it twice is what lets a proposal built from one file
land on another while both hashes still verify.

The miner opens session history through a read-only SQLite connection, so a bug
here cannot mutate Hermes's `state.db`.

## What gets mined

Replay is a single-shot text call with no tools. A turn that shelled out to
answer "can we push?" cannot be replayed fairly. The model cannot run
`git status` and can only report that it cannot verify. The rubric used to mark
that down as describing how it would proceed, and the optimizer's way out was a
rule like *"do not decline, answer from what you know"*, which scores well in
replay and is harmful in production, where the agent does have tools and the
skill may carry a fail-closed rule against guessing. A reported limit now counts
as a finding, which removes the pressure toward that rule but not the reason to
drop the turn: replaying it still asks a question the replay cannot answer.

So the miner drops turns that used tools replay cannot supply. `skill_view` is
the one exception, because replay puts the skill straight into the prompt and
reproduces that load. Counting it would also exclude every task, since skill
attribution requires a successful `skill_view`.

Measured over 200 real sessions, 47% of turns attributed to one skill survive the
filter, and 34% across all skills. The survivors are the explanatory turns,
"explain this repo" and "what is X?", the ones a toolless replay can answer
honestly. The filter also drops the task that let a four-character `"Yes."`,
which was wrong, beat an honest "Not yet, I can't verify that". That turn ran
`terminal`, so no replay of it could have known the answer.

`--allow-unreplayable` mines them anyway. The scores then reward guessing, and
the run says so.

### The filter is necessary and not sufficient

Tools are one of two things replay does not have. The other was the conversation.
A turn that called no tool itself can still depend on files read, or answers
given, three turns earlier, and replay gets a skill, an intent and a context
excerpt, nothing more.

`python experiments/replayability_tiers.py` counts what survives each definition,
over 200 real sessions:

| a turn is replayable if... | `daily-ai-news-digest` | all skills |
|---|---|---|
| 1. its own turn used no blocking tool *(enforced today)* | 125 (46.6%) | 777 (33.6%) |
| 2. ...and no **earlier** turn did either | **0 (0.0%)** | 81 (3.5%) |
| 3. ...and it is the session's first turn | 0 (0.0%) | 33 (1.4%) |

Zero. Every turn attributed to that skill sits after some turn in its session
used a tool, so every mined task inherits context replay cannot reproduce.
Supplying that context is what "What the excerpt carries" below is for. The miner
still enforces tier 1, and tiers 2 and 3 measure how much context it has to hand
over.

The replays showed it. Asked to explain a repo, the model answered *"no
repository, code, or file contents were actually provided in this session"*,
which was correct because none were, and lost to a baseline that answered anyway.
Same shape as the tool problem, one level up.

So tier 1 removes the confirmed pathology and is worth having, but it left the
real constraint on context rather than tools.

### What the excerpt carries

The attempt prompt carries `context_excerpt` directly beneath the intent. It used
to hold the session title, the model name, and a list of skill names, which is
why tier 2 measured zero. Nothing in it could tell a replay what an earlier turn
had found. It now carries the conversation itself:

```
Session: Daily AI news digest #3
Skills loaded: daily-ai-news-digest, github, skillopt

--- record of earlier turns in this conversation ---

[user] can we push?
[tool:terminal] {"output": "HEAD -> fix/safe-session-mining ... ", "exit_code": 0}
[assistant] Already pushed and verified. Local HEAD matches the remote branch...

--- end of record ---
```

**The record is closed as well as opened.** An excerpt that simply stops is a
transcript, and the likeliest continuation of a transcript is its next entry. One
replay answered a task with a raw `read_file` tool call instead of an answer, for
a file its own excerpt already quoted. The wording stays neutral about what to do
with the record. Telling the model to answer regardless of what it can verify is
the workaround the replayability filter exists to stop rewarding. The builder
strips provider tool-call markup on the way in for the same reason, though that
markup was never the cause. Only 1 of 692 excerpts carried any.

Across the same 200 sessions, 744 of 777 mined tasks (95.8%) now carry real
preceding turns. The remaining 4.2% are the first turn of their session, which
had none.

Three budgets govern the excerpt: 1200 characters per message, 8000 per excerpt,
8 tool results per turn. They were raised from 400/2000/4 after measuring how
hard the originals bound. The old numbers cut 63.5% of captured tool results,
dropped 71.3% of tool calls, and rendered 8.2% of available preceding turns.
`read_file` is the second most-used tool in this history at 3501 calls, so an
earlier turn read the code a mined turn asks about and this step then threw it
away. At the current budgets those three figures read 37.3%, 54.6% and 13.8%, and
the median excerpt is 6.8k characters rather than 1.8k. `_CONTEXT_MESSAGE_CHARS`
also applies at harvest, so it decides what is ever available to render.

The builder spends the budget newest-turn-first and renders oldest-first. When a
conversation does not fit, the turns nearest the task are the ones its request
refers back to. The replayed turn never appears in its own excerpt, because its
answer is what replay is supposed to produce.

**Redaction runs before tool output leaves the process.** Filling the excerpt
means shell output and file contents now reach a model provider, which they did
not when it held a session title. The redactor blanks credential-shaped
substrings and keeps the name that labelled them, so the model still knows a key
was there. Over real history it fires 265 times, mostly on `read_file` of a
config with `api_key:` in it, and leaves no credential-shaped string behind.

None of this makes an unreplayable turn replayable, and the tool filter still
runs. A turn that needed to *run* something is still dropped.

### What filling it measured

`experiments/judge_pairwise.py`, same four tasks and the same three arms, before
and after. The attempt cache keys on the request text, so an excerpt change
re-samples every response on its own.

| pairwise separation | good vs baseline | bad vs baseline | good > baseline > bad |
|---|---|---|---|
| before the tool filter | +0.375 | -0.125 | yes, but the run contained the `"Yes."` pathology |
| after it, empty excerpt | -0.125 | -0.125 | no |
| after it, filled excerpt, budgets 400/2000/4 | -0.250 | **-0.500** | no |
| budgets 1200/8000/8 | -0.125 | -0.250 | no, and one baseline reply was a tool call |
| ...with the record delimited | **+0.375** | **-0.500** | **yes** |

**The gate ranks all three arms correctly on the last row.** The helpful edit
wins three tasks in both orders and ties the fourth. The harmful edit, ten-word
answers with no reasoning, loses all four in both orders. Both directions hold at
once for the first time.

**Read the middle two rows together, because one sample separates them.** Raising
the budgets alone looked like it cost 0.250 on the harmful arm. It did not. One
baseline replay answered with a raw `read_file` tool call instead of an answer,
scored 0.00, and lost to a 65-character bad-arm response. Delimiting the record
fixed that reply. The same task now scores 1.00 and wins both orders, and the
harmful arm returned to -0.500 with nothing else changed. A four-task run moves
0.250 on one contaminated response, which is the honest size of n = 4.

**The absolute judge reads +0.000 on the same run, which is a ceiling rather than
a disagreement.** All four baselines now score 1.00, so the helpful edit has no
headroom left to occupy. Its earlier +0.250 was the contaminated baseline scoring
0.00, not a gain. Absolute still rates a 62-character answer 1.00 on one task
where pairwise scores it a loss in both orders. The rubric's grounding defect is
intact, and comparison is what contains it.

Position bias is still real and still worth its cost. One of the four helpful-arm
comparisons returned whichever answer came first and was neutralised to a tie, so
+0.375 is what survived order-swapping rather than what a single order reported.

## Outcome labels

Hermes SkillOpt uses conservative weak labels:

- `success`: a completed turn produced a final assistant response without a detected tool failure;
- `mixed`: a completed turn recovered from one or more detected tool failures and still produced a final response;
- `fail`: a completed turn had a detected failure and no final response;
- `unknown`: no reliable outcome evidence.

The miner parses structured tool results. A successful payload containing
`"error": null` does not count as a failure.

The miner never touches incomplete sessions.

## Backends and validation

### `mock` (default)

Free and deterministic. It mines failure patterns and proposes heuristics, but
its outcome-derived replay score cannot prove an edit improves real agent
performance. Reports often show a flat score with `greedy_flat`.

Treat mock proposals as reviewable suggestions, not validated regression fixes.
The CLI warns on every mock run for this reason.

### `hermes` (recommended)

Replays each task through Hermes's own configured providers, with the candidate
skill in the prompt, so the gate scores a real difference in model behaviour.
Credentials come from `<hermes-home>/.env`, which this backend loads the way
Hermes's own entrypoints do.

```bash
hermes-skillopt --backend hermes --skill my-skill
hermes-skillopt --backend hermes --model deepseek-v4-flash --skill my-skill
```

This backend pins the provider and model (`SKILLOPT_HERMES_PROVIDER`,
`SKILLOPT_HERMES_MODEL`, or `--model`) instead of leaving them to Hermes's
routing. An unpinned auxiliary call can fall through to a provider with a credit
fault and return empty content. `--hermes-agent-path` (or `HERMES_AGENT_PATH`)
points at the hermes-agent checkout when it is not under the Hermes home.

### `claude` and `codex`

These go to the upstream SkillOpt backend registry. Both shell out to an external
CLI and need that CLI installed, authenticated, and fast enough to answer inside
the per-call timeout.

```bash
hermes-skillopt --backend claude --skill my-skill
```

## Judging: absolute or pairwise

### `absolute` (default)

The judge rates each response against the task's rubric on a 0..1 scale, and the
gate compares the mean of those ratings before and after an edit.

Measured live on `daily-ai-news-digest`, that scale is not one. Across twelve
calls the judge emitted only `{0.0, 0.1, 0.9, 1.0}`. It gave a four-character
response 1.00 while a 2,564-character substantive answer scored 0.00. Against a
deliberately harmful skill edit it separated correctly (-0.225), but against a
deliberately helpful one it separated by +0.000. It is a coarse switch that
sometimes flips the wrong way. It can veto a disaster and cannot approve an
improvement.

The grounding requirement in the rubric spread that scale out to seven distinct
values with no ceiling. It also made the judge stricter with the helpful edit
rather than kinder, scoring it -0.350 below baseline. See "What the grounding
requirement changed".

### `pairwise` (recommended for real runs)

```bash
hermes-skillopt --backend hermes --judge pairwise --skill my-skill
```

Instead of rating one answer, the judge sees the baseline answer and the
candidate answer to the same task and picks which better satisfies the rubric.
Ranking two responses is a much easier question than scoring one, and it is the
standard remedy for exactly this binary collapse.

This also fixes what the gate means. A baseline scores `0.5`, tied with itself by
definition, so `candidate > baseline` stops being a comparison of two noisy
absolute means and becomes "the candidate wins more head-to-heads than it loses".

Every comparison runs twice with the order swapped. Judges have strong position
bias, so a verdict that flips when the answers are exchanged counts as a tie
rather than a win. That doubles judge calls, the cheap half of a replay at 200
tokens against an attempt's 512, and buys a verdict about the answers rather than
their placement.

Two shortcuts keep the cost down. A candidate whose text matches the baseline
scores a tie without any call. The gate metric is forced to `soft`, because a
pairwise score is already the comparison the gate wants, and blending it with a
threshold bit would let a candidate pass on the projection instead of on the
answers.

The choice changes the gate's decision, not only the reported separation.
`evaluate_gate` accepts on a strict `cand_score > current_score`, and on the last
run the two judges disagree about the same helpful edit:

| judge | current (baseline) | helpful candidate | gate |
|---|---|---|---|
| absolute | 1.000 | 1.000 | reject |
| pairwise | 0.500 | 0.875 | accept |

Absolute cannot accept it, because all four baselines already score 1.00 and
nothing sits above the ceiling. Pairwise anchors the baseline at 0.5 by
construction, so headroom exists in both directions.

`--judge pairwise` requires a real backend. The mock backend derives scores from
recorded outcomes rather than from the responses, so both sides of a comparison
would always tie. Asking for it raises rather than silently doing nothing.

### Measured

Four mined tasks, three skill variants (unchanged, a deliberately helpful edit, a
deliberately harmful one), `deepseek-v4-flash`. Each variant was attempted once
and both judges scored the same twelve responses, so the judge is the only
variable.

| separation vs baseline | helpful edit | harmful edit |
|---|---|---|
| absolute | +0.150 | -0.075 |
| pairwise | **+0.375** | **-0.125** |

Pairwise separates the helpful edit 2.5x wider. Approving a real improvement is
what the absolute judge could not do, so that is the number that matters.

Order-swapping earns its cost. Two of eight comparisons returned whichever answer
came first and counted as ties instead of results. One of those would otherwise
have scored the helpful edit a loss.

n = 4 tasks on one skill. Small, and a wider separation on one draw is not a
guarantee.

Re-run on the tool-filtered pool with the excerpt filled, both arms land. The
helpful edit separates +0.375 and the harmful one -0.500, and the old pathology
is gone: a 62-character answer that the absolute judge rates 1.00 loses both
orders under comparison. See "What filling it measured" for the four runs that
got there, and for the single response that made one of them look like a
regression.

### What the grounding requirement changed

`experiments/judge_grounding.py` judges four fixed pairs in both orders, under the
requirements as they were and as they are. Sixteen comparisons, ~17k tokens.

| pair | before | after |
|---|---|---|
| `"Yes."` vs "Not yet, I can't verify that from this session" | **guess wins both orders** | limit wins both |
| the same limit, elaborated | limit wins | limit wins |
| a plan offered in place of the work | answer wins | answer wins |
| an unnecessary limit vs. the answer the request contains | answer wins | answer wins |

The pathology reproduces only against the *terse* honest answer. Elaborate the
same refusal and it won before the change too, so what the confident guess beat
was brevity plus a rubric that never asked for support.

The last two pairs are the over-correction guard. Teaching a judge to respect "I
can't verify that" is progress only while an answer still beats a plan, and while
an unnecessary limit still loses to the answer the request already contains. Both
hold after the change.

n = 1 per cell. This is a regression probe for one known defect, not a measure of
judge quality.

**The full comparison run is unmoved.** Re-scoring the same twelve stored answers
leaves pairwise exactly where it was. Helpful +0.375, harmful -0.500, ranking
good > baseline > bad. Only the judge changed between the runs; the responses are
byte-identical, served from the attempt cache. The grounding requirement costs
the comparison judge nothing.

**The absolute judge moves, and not in its favour.** Its ceiling is gone:
baselines that all scored 1.00 now spread 0.20 to 1.00, and the scale emits seven
distinct values where it used to emit four. But it scores the helpful edit 0.275
against a 0.625 baseline, which is -0.350 where it used to be +0.000. Its own
reasons say why: "grounds no claims in actual evidence", "makes numerous claims".
The helpful edit's answers are the longest in the run, a median 2,842 characters
against the baseline's 1,611. A longer answer makes more claims, and a
requirement to support each of them hands a judge that never sees the record more
to mark down. No gate decision changes, since absolute rejected the helpful edit
before this too, for want of headroom rather than for cause. But the coarse judge
is now a harsh one, and the case for `--judge pairwise` is stronger than it was.

### Failures are loud

A wrapper around every real backend raises on an empty or unparseable result
instead of scoring `0.0`. Upstream's CLI backend returns `""` on any failure and
never checks the exit code, so an unauthenticated CLI used to produce a full run
of zeros, exit `0`, and a staged report that looked exactly like a candidate
which did not help.

Each real run now starts with one cheap probe, so a dead backend fails in seconds
instead of after every task, and a run with any failed replay stages nothing.

## Development

```bash
python -m pip install -e ".[dev]"
ruff check .
python -m pytest -q
```

CI runs lint and tests on Python 3.10, 3.11, and 3.12.

### Test against the published `skillopt`, not only a local checkout

A local editable checkout of upstream and a `pip install` from PyPI are not the
same package, and they can carry the same version number while shipping different
modules.

`pyproject.toml` asked for `skillopt>=0.1.0` until 2026-09-05. Published 0.1.0
ships `skillopt`, `skillopt_webui` and `scripts`, and no `skillopt_sleep`, which
is the module every file here imports. Anything that resolved to it installed
cleanly and then raised `ModuleNotFoundError` on the first import, so the floor
described a configuration that never worked. The workstation that wrote this code
did not notice, because its editable checkout of upstream `main` also calls itself
0.1.0 and does have `skillopt_sleep`. The floor is now `>=0.2.0`, and CI installs
it exactly so the claim is checked rather than asserted.

The versions also call this package differently. 0.2.0 passes `sample_id` through
`replay_one` so repeated rollouts stop sharing one cache slot, and a
fixed-signature wrapper here raised `TypeError` on every replay against it. The
wrappers now forward `*args`/`**kwargs` everywhere they delegate, and a test
covers it. `pytest` prints the resolved `skillopt` version and path in its header,
so a green run says which version it was green against.

Before changing a wrapper, check a clean install rather than only this machine:

```bash
python -m venv /tmp/v020 && /tmp/v020/bin/pip install "skillopt==0.2.0" pytest -e . --no-deps
/tmp/v020/bin/python -m pytest -q
```

## Project layout

| Path | Purpose |
|---|---|
| `hermes_skillopt/sleep.py` | Session harvesting, skill attribution, task mining |
| `hermes_skillopt/backend.py` | Hermes-specific pattern reflection |
| `hermes_skillopt/llm_backend.py` | Real replay transport, loud-failure guard, pairwise judge |
| `hermes_skillopt/run_nightly.py` | On-demand staging and safe adoption CLI |
| `tests/` | Session-mining and adoption regression tests |
| `experiments/` | Live judge measurements: separation, replayability tiers, grounding |

## Known limitations

- Outcome labels are inferred from transcripts; Hermes does not yet store explicit human quality labels.
- The default absolute judge is coarse. A subtle edit can improve a response
  without moving the score, so use `--judge pairwise` on real runs.
- Pairwise scores are ordinal. A win rate says the candidate is better, not by how much.
- **The rubric now asks for grounding; the judge still cannot check facts.**
  Asked "can we push?", a four-character `"Yes."`, which was wrong, beat a
  baseline that correctly answered "Not yet, I can't verify that from this
  session". Both judges scored it a win in both orders, which located the defect
  in `build_task_rubric` rather than in either judge. Nothing there required an
  answer to be *supported*. `"Yes."` is maximally direct, and naming a limit of
  the evidence read as the hedging the findings requirement penalizes. Two
  requirements now say so outright, the findings requirement names what it was
  aimed at (a plan offered in place of the work), and the comparison instructions
  no longer count every declining answer as a deferral. On the probe that
  reproduces the case the honest answer goes from losing both orders to winning
  both. What this does not buy is fact checking. The judge sees the request and
  two answers, never the record, so it can prefer an answer that rests on
  something over one that rests on nothing, and cannot tell you which is right.
  Comparison is still the stronger guard, so use `--judge pairwise`. See "What
  the grounding requirement changed".
- Skill attribution is session/turn based and depends on successful `skill_view` tool results.
- The miner drops roughly half of all turns as unreplayable. Optimizing a skill
  whose work is mostly tool-driven means optimizing the minority of it that is
  conversation.
- Replay still has no *tools*. `context_excerpt` now carries the preceding turns
  and what their tools returned, so a mined task no longer refers to facts the
  replay was never shown, but it is a transcript rather than a working directory.
  A turn that needed to run something is dropped, not replayed from context.
- Redaction works by pattern, not by parsing. It catches credential-shaped
  substrings and does not guarantee that nothing sensitive reaches the provider.
- The default mock backend cannot demonstrate score lift; use `--backend hermes` to validate.
- Staging directories created before v0.2.0 lack safety hashes and are refused; regenerate them.

## License

MIT. See [LICENSE](LICENSE). Microsoft SkillOpt remains governed by its own upstream license.
