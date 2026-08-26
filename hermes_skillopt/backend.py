"""
HermesBackend — a SkillOpt backend that evaluates tasks based on
outcome data from Hermes sessions and proposes edits based on
common failure patterns observed across sessions.

This is a lightweight wrapper that extends MockBackend's scoring
but replaces the reflect function with Hermes-specific logic.
"""

from __future__ import annotations

import re
from typing import List

from skillopt_sleep.backend import MockBackend
from skillopt_sleep.types import EditRecord, ReplayResult, TaskRecord


class HermesBackend(MockBackend):
    """Mock backend augmented with Hermes-specific failure analysis.

    Uses the same deterministic attempt/judge logic as MockBackend
    (outcome-derived scoring) but replaces reflect with a model that
    analyzes failure patterns in mined tasks to propose targeted edits
    to the skill document.
    """

    name = "hermes"

    def attempt(self, task: TaskRecord, skill: str, memory: str) -> str:
        # Use outcome-derived response: if the real outcome was success,
        # produce a plausible correct response; if fail, produce a near-miss
        if task.outcome == "success":
            if task.attempted_solution:
                return task.attempted_solution[:300]
            return f"[successful response for: {task.intent[:80]}]"
        else:
            if task.attempted_solution:
                return f"[attempted but problematic: {task.attempted_solution[:200]}]"
            return f"[failed attempt for: {task.intent[:80]}]"

    def reflect(
        self,
        failures: List[ReplayResult],
        successes: List[ReplayResult],
        skill: str,
        memory: str,
        edit_budget: int = 4,
        night: int = 1,
        **kwargs,  # accept evolve_skill, evolve_memory, etc.
    ) -> List[EditRecord]:
        """Analyze failure patterns in Hermes sessions and propose edits.

        Looks for common failure signatures in the task data and proposes
        targeted additions to the skill document.
        """
        edits = []
        budget = edit_budget

        # Collect replay results from the tuples passed by consolidate
        failure_replay_results = [r for _t, r in failures] if failures else []
        success_replay_results = [r for _t, r in successes] if successes else []

        # Analyze what went wrong in failures
        n_failures = len(failure_replay_results)
        n_successes = len(success_replay_results)
        total = n_failures + n_successes
        failure_rate = n_failures / total if total > 0 else 0

        if n_failures == 0 or budget <= 0:
            return edits

        # Always propose at least one improvement when there are failures.
        # Priority order: specific patterns first, then general fallbacks.

        # Pattern 1: Tool error recovery (most impactful)
        tool_error_kw = ["error", "traceback", "failed", "timeout", "refused"]
        tool_fails = [r for r in failure_replay_results
                      if r.fail_reason and any(kw in r.fail_reason.lower() for kw in tool_error_kw)]
        if len(tool_fails) >= 2 and budget > 0:
            edits.append(EditRecord(
                target="skill", op="add",
                content=(
                    "## Error Recovery\n"
                    "When a tool returns an error or traceback, do NOT silently continue. "
                    "Report the error to the user with the specific error message, then "
                    "suggest an alternative approach or fallback strategy."
                ),
                rationale=f"{len(tool_fails)}/{n_failures} failures involve tool errors — needs recovery guidance"
            ))
            budget -= 1

        # Pattern 2: Self-verification when failure rate is significant
        if failure_rate >= 0.30 and budget > 0:
            edits.append(EditRecord(
                target="skill", op="add",
                content=(
                    "## Self-Verification\n"
                    "Before delivering a final answer, verify: (1) Were all tool calls "
                    "successful? (2) Is the output non-empty? (3) Does the response "
                    "directly address what the user asked for?"
                ),
                rationale=f"Failure rate is {failure_rate:.0%} ({n_failures}/{total}) — self-verification directive needed"
            ))
            budget -= 1

        # Pattern 3: Learn from successes — what separates wins from losses
        if n_successes > 0 and budget > 0:
            # Check if skill already has structured output guidance
            has_guidance = bool(re.search(
                r'(always|must|ensure|verify).{0,50}(respond|output|return|answer)',
                skill, re.IGNORECASE
            ))
            if not has_guidance:
                edits.append(EditRecord(
                    target="skill", op="add",
                    content=(
                        "## Response Structure\n"
                        "Structure every response with: (a) what was done, "
                        "(b) the result, (c) next actions available. "
                        "This pattern correlates with successful task completion."
                    ),
                    rationale=f"{n_successes} successes show structured output correlates with good outcomes"
                ))
                budget -= 1

        # Pattern 4: If still have budget and significant failures, add a general catch
        if n_failures >= 3 and budget > 0:
            edits.append(EditRecord(
                target="skill", op="add",
                content=(
                    "## Failure Prevention\n"
                    "When encountering a repeated failure pattern across sessions, "
                    "pause and diagnose the root cause before proceeding. Common causes: "
                    "missing tool arguments, stale file references, or skipped verification steps."
                ),
                rationale=f"{n_failures} failures across {total} tasks — general prevention guidance warranted"
            ))
            budget -= 1

        return edits
