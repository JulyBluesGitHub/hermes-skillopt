"""Is this turn still about the skill it was attributed to?

Skill attribution is sticky. ``active_skills`` in :mod:`hermes_skillopt.sleep`
accumulates and is never cleared, so one ``skill_view`` at turn 1 makes every
later turn in that session a task for that skill, however far the conversation
drifts. Measured on ``daily-ai-news-digest``: all five mined tasks were AIMentor
engineering chat from two sessions that opened as a digest and wandered. The
optimizer wrote rules for a news skill out of a conversation about retrieval
routing, and the gate approved them, because the gate scores whether an edit
helps on the mined tasks and cannot know the tasks are the wrong ones.

The replayability filter makes it worse rather than better. A real digest turn
uses ``web_extract`` and ``browser_*``, so that filter drops the skill's actual
work and keeps the toolless conversational drift. The surviving task set is close
to the inverse of what the skill does.

So a turn is kept only if it still looks like the skill's subject. Two scorers
implement that, both scoring a turn against an anchor built from the skill's own
front matter:

* :class:`EmbeddingScorer` - all-MiniLM-L6-v2 through ``sentence-transformers``.
  Optional (``pip install hermes-skillopt[topic]``), because it pulls torch.
* :class:`LexicalScorer` - idf-weighted token overlap, standard library only.
  The fallback when the model is not installed.

Measured over 200 sessions, 6,219 attributed turns, against the only labels the
transcripts carry for free: a turn whose own ``skill_view`` fired is on-topic by
construction.

    scorer     keeps of in-turn   keeps of inherited
    embedding        67.5%              39.3%
    lexical          80.6%              65.8%

The embedding separates about twice as well. Lexical scores are also
zero-inflated: a turn sharing no vocabulary with the anchor scores exactly 0.0,
so it drops honest follow-ups that use synonyms, and its threshold barely moves
the outcome. It is a fallback, not an equivalent.

Thresholds are calibrated per skill and never globally. In-turn medians ranged
from 0.150 (``hermes-agent``) to 0.342 (``daily-ai-news-digest``) on the same
scorer, so one global cutoff would gut one skill and pass another.
"""

from __future__ import annotations

import io
import math
import os
import re
import statistics
from collections import Counter
from typing import Dict, List, Optional, Sequence

#: Keep an inherited turn scoring at least this fraction of the skill's own
#: in-turn median. Chosen on the measurement above: 0.7 drops 4 of the 5 known
#: bad tasks while keeping 67.5% of turns that are on-topic by construction. 1.0
#: drops all 5 and keeps only 51%, which is too much real signal to pay.
DEFAULT_FRACTION = 0.7

#: Below this many in-turn turns a skill cannot be calibrated: the median of two
#: or three samples is noise, and a threshold from noise silently drops good
#: tasks. Uncalibratable skills keep only their in-turn turns, which is
#: conservative and correct by construction.
MIN_CALIBRATION = 5

_WORD = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "the a an and or of to in for on with is are be as at by from this that it its "
    "we you i can could would do does did not no if then than so what how why when "
    "which who whom our your my me us them they he she his her their there here".split()
)


def build_anchor(skill_text: str, skill_name: str) -> str:
    """What this skill is *about*, as text to compare a turn against.

    Front matter ``description`` plus the Purpose and Trigger sections. The body
    of a SKILL.md is procedure, and procedure describes how the work is done
    rather than what it is about, so including it blurs the anchor. MiniLM also
    truncates past 256 word pieces, which the body would consume on its own.
    """
    parts = [skill_name.replace("-", " ")]
    if skill_text:
        for line in skill_text.splitlines():
            if line.startswith("description:"):
                parts.append(line.split(":", 1)[1].strip().strip('"').strip("'"))
                break
        for heading in ("## Purpose", "## Trigger"):
            if heading in skill_text:
                body = skill_text.split(heading, 1)[1].split("\n## ", 1)[0]
                parts.append(" ".join(body.split())[:400])
    return "\n".join(p for p in parts if p)


def _tokens(text: str) -> List[str]:
    return [w for w in _WORD.findall(text.lower()) if w not in _STOPWORDS and len(w) > 2]


class LexicalScorer:
    """idf-weighted token overlap. No dependencies, and a weaker discriminator.

    idf is computed over the turns being scored, so vocabulary common to this
    corpus counts for little and a rare shared term counts for a lot. Without it
    the anchor's ordinary English matches everything.
    """

    name = "lexical"

    def score(self, anchor: str, prompts: Sequence[str]) -> List[float]:
        docs = [set(_tokens(p)) for p in prompts]
        df: Counter = Counter()
        for d in docs:
            df.update(d)
        n = max(len(docs), 1)
        # Smoothed idf, always >= 1. The textbook log(n / (1 + df)) goes negative
        # once a term appears in every document, and a negative weight makes the
        # norm below the square root of a negative number.
        idf = {w: math.log((1 + n) / (1 + c)) + 1.0 for w, c in df.items()}
        a_tokens = set(_tokens(anchor))
        a_norm = math.sqrt(sum(idf.get(w, 0.0) for w in a_tokens))

        out = []
        for d in docs:
            d_norm = math.sqrt(sum(idf.get(w, 0.0) for w in d))
            if not a_norm or not d_norm:
                out.append(0.0)
                continue
            shared = sum(idf.get(w, 0.0) for w in (a_tokens & d))
            out.append(shared / (a_norm * d_norm))
        return out


class EmbeddingScorer:
    """Cosine similarity under all-MiniLM-L6-v2.

    The model loads once per instance. Encoding 6,219 turns takes about a minute
    on CPU, which is nothing beside a replay run, and costs no API calls.
    """

    name = "embedding"

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)

    def score(self, anchor: str, prompts: Sequence[str]) -> List[float]:
        if not prompts:
            return []
        from sentence_transformers import util

        vecs = self._model.encode(
            [anchor] + [p[:2000] for p in prompts],
            normalize_embeddings=True,
            batch_size=64,
            show_progress_bar=False,
        )
        return [float(util.cos_sim(v, vecs[0])[0][0]) for v in vecs[1:]]


#: Force a scorer instead of taking the best available. "lexical" is what a
#: machine without the optional extra runs, so it is also what the test suite and
#: CI use: a suite that silently picks a different scorer than CI is testing a
#: configuration nobody ships, and loading a transformer costs 15s per run.
SCORER_ENV = "HERMES_SKILLOPT_TOPIC_SCORER"


def get_scorer(prefer_embedding: bool = True):
    """The best scorer available, never an import error.

    A missing optional dependency must degrade the filter, not fail the run: the
    caller is mining tasks, and refusing to mine because torch is absent would be
    a worse outcome than mining with the weaker scorer and saying so.
    """
    choice = os.environ.get(SCORER_ENV, "").strip().lower()
    if choice == "lexical":
        return LexicalScorer()
    if choice == "embedding":
        return EmbeddingScorer()  # explicit request: a failure here should be loud
    if prefer_embedding:
        try:
            return EmbeddingScorer()
        except Exception:  # ImportError, model download failure, torch load error
            pass
    return LexicalScorer()


def calibrate(
    in_turn_scores: Sequence[float], fraction: float = DEFAULT_FRACTION
) -> Optional[float]:
    """Threshold for one skill, or None when there is too little to calibrate on.

    The reference set is the skill's own in-turn turns, so the threshold is in
    that skill's units. Skills sit on genuinely different scales under the same
    scorer, and a global cutoff would keep everything for one and nothing for
    another.
    """
    if len(in_turn_scores) < MIN_CALIBRATION:
        return None
    return statistics.median(in_turn_scores) * fraction


def on_topic_flags(
    scorer,
    anchor: str,
    prompts: Sequence[str],
    in_turn: Sequence[bool],
    *,
    fraction: float = DEFAULT_FRACTION,
    stats: Optional[Dict[str, float]] = None,
) -> List[bool]:
    """Which turns to keep. In-turn turns are always kept.

    A turn whose own ``skill_view`` fired is on-topic by construction: the agent
    loaded the skill in order to answer it. Those are the calibration reference
    and are never filtered by a threshold derived from themselves. The threshold
    decides only the inherited turns, which are the ones in question.
    """
    if not prompts:
        return []
    scores = scorer.score(anchor, prompts)
    threshold = calibrate([s for s, i in zip(scores, in_turn) if i], fraction)

    if stats is not None:
        stats["scorer"] = getattr(scorer, "name", "?")
        stats["n_in_turn"] = sum(1 for i in in_turn if i)
        stats["threshold"] = -1.0 if threshold is None else round(threshold, 4)

    if threshold is None:
        # Not enough ground truth to place a cutoff. Keeping only the turns that
        # are on-topic by construction is conservative: the run may then report
        # too few tasks, which is a visible outcome, where a guessed threshold
        # produces a confident wrong one.
        return list(in_turn)
    return [bool(i or s >= threshold) for s, i in zip(scores, in_turn)]


def load_skill_text(path: str) -> str:
    if not path:
        return ""
    try:
        return io.open(path, encoding="utf-8").read()
    except OSError:
        return ""
