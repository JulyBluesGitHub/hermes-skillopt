"""A turn belongs to the skill it is still about, not to whatever loaded first.

Attribution is sticky: `active_skills` accumulates and is never cleared, so one
`skill_view` at turn 1 owns every later turn in that session. Measured on
`daily-ai-news-digest`, that made all five mined tasks AIMentor engineering chat,
the optimizer wrote news-skill rules from a conversation about retrieval routing,
and the gate approved them - a gate scores whether an edit helps on the tasks it
is given and cannot know the tasks are the wrong ones.

These pin the filter's properties rather than its numbers: thresholds are
calibrated per skill from real data, and a number copied into an assertion here
would be a second, stale calibration.
"""

from hermes_skillopt.sleep import harvest_hermes_sessions, mine_hermes_tasks
from hermes_skillopt.topic import (
    MIN_CALIBRATION,
    LexicalScorer,
    build_anchor,
    calibrate,
    get_scorer,
    on_topic_flags,
)
from tests.test_session_mining import _make_db, _skill_result

DIGEST = """---
name: daily-ai-news-digest
description: "Daily verified AI-news delta: fresh, deduped, actionable."
---

## Purpose
Produce a short update on which AI news stories materially changed today.

## Trigger
The user asks for today's AI news, or the weekday digest run fires.

## Pipeline
Search the feeds, dedupe against yesterday, verify each claim before writing.
"""


# ── the anchor ───────────────────────────────────────────────────────────────

def test_the_anchor_is_what_the_skill_is_about_not_how_it_works():
    """Procedure describes how work is done, not what it is about.

    A SKILL.md is mostly pipeline steps. Including them blurs the anchor toward
    the vocabulary of every technical conversation, and MiniLM truncates past 256
    word pieces, which a full body would consume on its own.
    """
    anchor = build_anchor(DIGEST, "daily-ai-news-digest")

    assert "AI-news delta" in anchor              # description
    assert "materially changed today" in anchor   # Purpose
    assert "weekday digest run" in anchor         # Trigger
    assert "dedupe against yesterday" not in anchor   # Pipeline: procedure


def test_an_unreadable_skill_still_yields_an_anchor():
    """A skill whose file is missing must not take the whole run down with it."""
    assert build_anchor("", "daily-ai-news-digest") == "daily ai news digest"


# ── calibration ──────────────────────────────────────────────────────────────

def test_a_threshold_is_never_placed_on_too_few_samples():
    """A median of two or three points is noise, and a threshold from noise
    silently drops good tasks while looking like a measurement."""
    assert calibrate([0.3] * (MIN_CALIBRATION - 1)) is None
    assert calibrate([0.3] * MIN_CALIBRATION) is not None


def test_thresholds_are_in_each_skill_s_own_units():
    """Measured in-turn medians ran from 0.150 to 0.342 under one scorer, so a
    global cutoff would keep everything for one skill and nothing for another."""
    quiet = calibrate([0.15] * 10)
    loud = calibrate([0.34] * 10)
    assert quiet is not None and loud is not None
    assert quiet < loud


# ── which turns survive ──────────────────────────────────────────────────────

def _flags(prompts, in_turn, **kw):
    return on_topic_flags(LexicalScorer(), build_anchor(DIGEST, "daily-ai-news-digest"),
                          prompts, in_turn, **kw)


def test_a_turn_that_loaded_the_skill_is_never_filtered_out():
    """It is on-topic by construction: the agent loaded the skill to answer it.

    It is also the calibration reference, so filtering it with a threshold
    derived from itself would be circular.
    """
    prompts = ["something entirely unrelated to news"] * (MIN_CALIBRATION + 2)
    assert all(_flags(prompts, [True] * len(prompts)))


def test_drifted_turns_are_dropped_and_on_topic_ones_kept():
    on_topic = ["today's AI news digest please", "fresh verified AI news delta",
                "which AI news stories materially changed", "the weekday AI news run",
                "give me today's verified AI news"]
    drifted = ["why is our embedding-based router so much better",
               "explain this repo to me in laymans terms"]

    stats = {}
    flags = _flags(on_topic + drifted, [True] * len(on_topic) + [False] * len(drifted),
                   stats=stats)

    assert stats["threshold"] > 0, "expected a calibrated threshold, not the fallback"
    assert all(flags[:len(on_topic)])
    assert not any(flags[len(on_topic):])


def test_without_enough_ground_truth_only_in_turn_turns_survive():
    """The conservative fallback. Too few labels to place a cutoff means the run
    may report too few tasks, which is visible, rather than optimizing on turns
    nothing established belong to the skill.
    """
    prompts = ["today's AI news", "why is our router better"]
    stats = {}
    assert _flags(prompts, [True, False], stats=stats) == [True, False]
    assert stats["threshold"] == -1.0


def test_no_turns_is_not_a_crash():
    assert _flags([], []) == []


# ── the scorer that actually runs ────────────────────────────────────────────

def test_a_missing_optional_dependency_degrades_the_filter_rather_than_the_run(monkeypatch):
    """torch is a large optional extra. Refusing to mine without it would be a
    worse outcome than mining with the weaker scorer."""
    import hermes_skillopt.topic as topic

    monkeypatch.delenv(topic.SCORER_ENV, raising=False)
    monkeypatch.setattr(topic, "EmbeddingScorer",
                        lambda *a, **k: (_ for _ in ()).throw(ImportError("no torch")))
    assert topic.get_scorer().name == "lexical"


def test_the_scorer_can_be_pinned_so_tests_run_what_ci_runs(monkeypatch):
    import hermes_skillopt.topic as topic

    monkeypatch.setenv(topic.SCORER_ENV, "lexical")
    assert get_scorer().name == "lexical"


def test_idf_survives_a_term_that_appears_in_every_turn():
    """Textbook idf goes negative once df equals the corpus size, and the norm is
    then the square root of a negative number. Found by the suite, not by
    reasoning about it."""
    scores = LexicalScorer().score("news digest", ["news news", "news", "news digest"])
    assert all(s >= 0.0 for s in scores)


# ── end to end through the miner ─────────────────────────────────────────────

def test_the_miner_drops_a_session_that_wandered_off_the_skill(tmp_path):
    """The measured failure, in miniature.

    A digest session whose later turns are project chat. Those turns call no
    tool, so the replayability filter keeps them; they are what the optimizer
    then writes news-skill rules from.
    """
    loaded = {"role": "tool", "tool_name": "skill_view",
              "content": _skill_result("daily-ai-news-digest")}
    messages = [{"role": "user", "content": "today's AI news digest"}, loaded,
                {"role": "assistant", "content": "here is the digest"}]
    for _ in range(MIN_CALIBRATION):
        messages += [{"role": "user", "content": "give me the verified AI news delta"},
                     loaded,
                     {"role": "assistant", "content": "digest"}]
    messages += [
        {"role": "user", "content": "why is our embedding-based router so much better"},
        {"role": "assistant", "content": "because of the cascade"},
    ]

    sessions = harvest_hermes_sessions(str(_make_db(tmp_path, messages)))
    mined = [t.intent for t in mine_hermes_tasks(sessions, skill_name="daily-ai-news-digest")]

    assert mined, "the on-topic digest turns should still be mined"
    assert not any("router" in intent for intent in mined)

    kept = [t.intent for t in mine_hermes_tasks(
        sessions, skill_name="daily-ai-news-digest", require_on_topic=False)]
    assert any("router" in intent for intent in kept), "opting out must restore it"


def test_dropped_turns_are_reported_rather_than_vanishing(tmp_path):
    """A task set that silently halved is how the sticky-attribution defect went
    unnoticed for months."""
    loaded = {"role": "tool", "tool_name": "skill_view",
              "content": _skill_result("daily-ai-news-digest")}
    messages = []
    for _ in range(MIN_CALIBRATION + 1):
        messages += [{"role": "user", "content": "today's verified AI news delta"}, loaded,
                     {"role": "assistant", "content": "digest"}]
    messages += [{"role": "user", "content": "refactor the retrieval router please"},
                 {"role": "assistant", "content": "done"}]

    sessions = harvest_hermes_sessions(str(_make_db(tmp_path, messages)))
    skipped, stats = {}, {}
    mine_hermes_tasks(sessions, skill_name="daily-ai-news-digest",
                      skipped=skipped, topic_stats=stats)

    assert skipped.get("(off-topic)", 0) >= 1
    assert stats["scorer"] == "lexical"
    assert stats["n_attributed"] > stats["n_on_topic"]
