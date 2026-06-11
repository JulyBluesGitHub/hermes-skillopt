"""
Hermes SkillOpt — On-demand skill optimization for Hermes Agent.

Analyzes your Hermes session history to find failure patterns and propose
targeted improvements to your skill (.md) files. Backed by Microsoft's
SkillOpt framework.

Requires:
    pip install skillopt

Usage:
    python -m hermes_skillopt.run_nightly --run-and-adopt
    python -m hermes_skillopt.run_nightly --skill codex
    python -m hermes_skillopt.run_nightly --list-staged
"""

__version__ = "0.1.0"
