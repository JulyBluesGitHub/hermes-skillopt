# Hermes SkillOpt

On-demand skill optimization for Hermes Agent, backed by Microsoft's [SkillOpt](https://github.com/microsoft/SkillOpt) engine.

Hermes SkillOpt reads completed Hermes sessions, identifies the skills that were actually loaded through `skill_view`, mines task outcomes, and stages bounded additions to each skill's managed `LEARNED` block.

## Why this adapter exists

SkillOpt already provides the optimization loop. This repository supplies the Hermes-specific seams:

- reads Hermes's SQLite session history;
- associates user turns with final assistant responses and tool failures;
- attributes tasks only to skills actually loaded in that session;
- stages proposed `SKILL.md` updates outside the live skills directory;
- verifies file hashes before adoption and creates a backup.

It is intentionally on-demand. It does not install a daemon or cron job.

## Install

Requires Python 3.10+ and a local Hermes installation with session history.

```bash
git clone https://github.com/JulyBluesGitHub/hermes-skillopt.git
cd hermes-skillopt
python -m pip install -e ".[dev]"
```

The upstream `skillopt` package is installed as a dependency.

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

If a name matches more than one file — a real possibility, since two categories
can each hold a skill of the same name — the run fails with `AmbiguousSkillError`
rather than editing whichever the filesystem happened to return first. Pass
`--skills-dir` to point at the one you mean.

## Safety model

Every proposal is written under:

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

Edits are confined to the managed block:

```markdown
<!-- SKILLOPT-SLEEP:LEARNED START -->
...
<!-- SKILLOPT-SLEEP:LEARNED END -->
```

Hand-written content outside this block is preserved by SkillOpt's edit applicator.

The path a proposal is staged against is the exact path the optimizer read, and
it travels in the report rather than being re-derived at staging time. Resolving
it twice is what allows a proposal built from one file to be adopted onto
another while both hashes still verify.

Session history is opened through a read-only SQLite connection, so a bug in
this tool cannot mutate Hermes's `state.db`.

## Outcome labels

Hermes SkillOpt uses conservative weak labels:

- `success`: a completed turn produced a final assistant response without a detected tool failure;
- `mixed`: a completed turn recovered from one or more detected tool failures and still produced a final response;
- `fail`: a completed turn had a detected failure and no final response;
- `unknown`: no reliable outcome evidence.

Structured tool results are parsed. A successful payload containing `"error": null` is not treated as a failure.

Incomplete sessions are never mined.

## Backends and validation

### `mock` (default)

Free and deterministic. It mines failure patterns and proposes heuristics, but its outcome-derived replay score cannot prove that an edit improves real agent performance. Reports may therefore show a flat score with `greedy_flat`.

Treat mock proposals as reviewable suggestions, not validated regressions fixes.

### `claude` and `codex`

These names are passed to the upstream SkillOpt backend registry. Availability depends on the installed SkillOpt version and local backend credentials. They are experimental in this Hermes adapter.

```bash
hermes-skillopt --backend claude --skill my-skill
```

## Development

```bash
python -m pip install -e ".[dev]"
ruff check .
python -m pytest -q
```

CI runs lint and tests on Python 3.10, 3.11, and 3.12.

## Project layout

| Path | Purpose |
|---|---|
| `hermes_skillopt/sleep.py` | Session harvesting, skill attribution, task mining |
| `hermes_skillopt/backend.py` | Hermes-specific pattern reflection |
| `hermes_skillopt/run_nightly.py` | On-demand staging and safe adoption CLI |
| `tests/` | Session-mining and adoption regression tests |

## Known limitations

- Outcome labels are inferred from transcripts; explicit human quality labels are not yet stored by Hermes.
- Skill attribution is session/turn based and depends on successful `skill_view` tool results.
- The default mock backend cannot demonstrate score lift.
- Staging directories created before v0.2.0 lack safety hashes and are refused; regenerate them.

## License

MIT. See [LICENSE](LICENSE). Microsoft SkillOpt remains governed by its own upstream license.
