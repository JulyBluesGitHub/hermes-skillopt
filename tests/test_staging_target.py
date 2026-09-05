"""The staged proposal must target the exact file the optimizer read.

Regression guard: staging used to re-resolve the skill path from ``hermes_home``
while the optimizer had loaded the skill from the default skills directory. When
those disagreed, the proposal derived from file A was staged against file B, and
adoption's hash checks still passed because each hash was taken from a different
file — so the wrong skill got overwritten with content derived from another.
"""

import json
from pathlib import Path

import hermes_skillopt.run_nightly as runner


def _skill(root: Path, *parts, body: str) -> Path:
    path = root.joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


def test_staging_targets_the_optimized_file(tmp_path, monkeypatch):
    optimized = _skill(tmp_path, "real", "skills", "demo", "SKILL.md", body="the real skill")
    decoy = _skill(tmp_path, "other", "skills", "demo", "SKILL.md", body="a different skill")
    staging_root = tmp_path / "staging"
    staging_root.mkdir()

    def fake_sleep(**kwargs):
        # The optimizer reports the path it actually read.
        return {
            "skill_name": "demo",
            "live_skill_path": str(optimized),
            "accepted": True,
            "edits": [{"target": "skill", "op": "add", "content": "learned"}],
            "rejected_edits": [],
            "proposed_skill": "the real skill\nlearned",
            "baseline_score": 0.0,
            "candidate_score": 0.0,
        }

    monkeypatch.setattr("hermes_skillopt.sleep.run_hermes_sleep", fake_sleep)

    # hermes_home deliberately points at the decoy tree.
    runner.run_nightly(hermes_home=str(tmp_path / "other"), skills=["demo"],
                       staging_root=str(staging_root))

    staged = list(staging_root.iterdir())
    assert len(staged) == 1
    manifest = json.loads((staged[0] / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["live_skill_path"] == str(optimized)
    assert manifest["live_skill_path"] != str(decoy)
    assert decoy.read_text(encoding="utf-8") == "a different skill"


def test_unresolved_live_path_is_not_staged(tmp_path, monkeypatch):
    staging_root = tmp_path / "staging"
    staging_root.mkdir()

    def fake_sleep(**kwargs):
        return {
            "skill_name": "demo",
            "live_skill_path": "",
            "accepted": True,
            "edits": [{"target": "skill", "op": "add", "content": "learned"}],
            "rejected_edits": [],
            "proposed_skill": "proposed",
            "baseline_score": 0.0,
            "candidate_score": 0.0,
        }

    monkeypatch.setattr("hermes_skillopt.sleep.run_hermes_sleep", fake_sleep)

    results = runner.run_nightly(hermes_home=str(tmp_path), skills=["demo"],
                                 staging_root=str(staging_root))

    assert list(staging_root.iterdir()) == []
    assert results[0]["skipped"] == "unresolved live skill path"
