# Hermes SkillOpt

**On-demand skill optimization for AI coding agents.** Analyzes your agent's session history to find failure patterns and propose targeted improvements to skill files — backed by Microsoft's [SkillOpt](https://github.com/microsoft/SkillOpt) framework.

> Built for Hermes Agent. Ports to Claude Code and Codex are straightforward (SkillOpt already has plugins for both).

## What it does

1. **Harvests** your recent agent sessions (from SQLite DB)
2. **Mines** every task for success/failure patterns
3. **Reflects** on failures and proposes bounded, targeted edits
4. **Gates** edits against a held-out validation set
5. **Stages** improvements in a LEARNED block at the bottom of your skill files

Your hand-written skill content is never touched. Improvements accumulate in a marked `<!-- SKILLOPT-SLEEP:LEARNED -->` section.

## Install

```bash
# Requires Python 3.10+ and the SkillOpt package
pip install skillopt

# Clone this repo
git clone https://github.com/july-ai/hermes-skillopt.git
cd hermes-skillopt
```

## Usage

### One command (recommended)
```bash
python -m hermes_skillopt.run_nightly --run-and-adopt
```
This analyzes all recently-used skills and applies improvements in one shot.

### Step by step
```bash
# See what's available
python -m hermes_skillopt.sleep status

# Dry-run: preview proposed edits
python -m hermes_skillopt.run_nightly --dry-run

# Run for a specific skill
python -m hermes_skillopt.run_nightly --skill my-skill-name

# See staged improvements
python -m hermes_skillopt.run_nightly --list-staged

# Adopt everything
python -m hermes_skillopt.run_nightly --adopt-all
```

### Alias (optional)
```bash
alias skillopt="cd ~/hermes-skillopt && python -m hermes_skillopt.run_nightly --run-and-adopt"
```

## How it works

```
Your agent sessions (SQLite DB)
  → harvest: extract every task with success/fail labels
  → mine: group recurring patterns, split train/val
  → reflect: analyze failures, propose bounded edits
  → gate: validate against held-out set
  → stage: append improvements to SKILL.md LEARNED block
```

Every adoption creates a backup. Every edit has a rationale (e.g., "67% of tasks fail from missing verification — added Self-Verification directive").

## Porting to Claude Code / Codex

This plugin is built on SkillOpt's engine. SkillOpt already ships plugins for Claude Code and Codex:

- **Claude Code:** `/plugin marketplace add ./skillopt-sleep-plugin && /sleep`
- **Codex:** `bash plugins/codex/install.sh && /sleep`

See [Microsoft SkillOpt](https://github.com/microsoft/SkillOpt) for the upstream.

The value-add of this repo is the Hermes session harvester — it reads from Hermes's SQLite session DB. For Claude/Codex, the upstream SkillOpt-Sleep plugins already do the equivalent from `~/.claude/` transcripts.

## Files

| File | Purpose |
|------|---------|
| `hermes_skillopt/run_nightly.py` | Main CLI: run, stage, adopt |
| `hermes_skillopt/sleep.py` | Session harvester + task miner |
| `hermes_skillopt/backend.py` | Failure pattern analyzer |

## Backends

- **mock** (default): Pattern analysis from session history. Free, fast, no API calls.
- **claude**: Re-run tasks through Claude to validate real score improvement. Needs `claude` CLI and API key.
- **codex**: Re-run tasks through Codex. Needs Codex CLI and API key.

```bash
python -m hermes_skillopt.run_nightly --backend claude --skill my-skill
```

## License

MIT — same as upstream SkillOpt.
