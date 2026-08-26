import hashlib
import json
from pathlib import Path

from hermes_skillopt.run_nightly import adopt_skill, write_staging_report


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _report(proposed: str) -> dict:
    return {
        "accepted": True,
        "edits": [{"target": "skill", "op": "add", "content": proposed}],
        "rejected_edits": [],
        "proposed_skill": proposed,
    }


def test_adopt_skill_matches_manifest_name_not_hyphen_suffix(tmp_path):
    staging = tmp_path / "staging"
    staging.mkdir()
    claude = tmp_path / "claude.md"
    code = tmp_path / "code.md"
    claude.write_text("claude old", encoding="utf-8")
    code.write_text("code old", encoding="utf-8")
    write_staging_report("claude-code", _report("claude new"), "claude new", str(claude), str(staging))
    write_staging_report("code", _report("code new"), "code new", str(code), str(staging))

    adopted = adopt_skill("claude-code", staging_root=str(staging))

    assert adopted is True
    assert claude.read_text(encoding="utf-8") == "claude new"
    assert code.read_text(encoding="utf-8") == "code old"


def test_adoption_is_idempotent(tmp_path):
    staging = tmp_path / "staging"
    staging.mkdir()
    live = tmp_path / "skill.md"
    live.write_text("old", encoding="utf-8")
    staged_dir = Path(
        write_staging_report("skill", _report("new"), "new", str(live), str(staging))
    )

    assert adopt_skill("skill", staging_root=str(staging)) is True
    assert adopt_skill("skill", staging_root=str(staging)) is False
    manifest = json.loads((staged_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "adopted"


def test_adoption_refuses_when_live_skill_changed_after_staging(tmp_path):
    staging = tmp_path / "staging"
    staging.mkdir()
    live = tmp_path / "skill.md"
    live.write_text("old", encoding="utf-8")
    write_staging_report("skill", _report("new"), "new", str(live), str(staging))
    live.write_text("changed independently", encoding="utf-8")

    adopted = adopt_skill("skill", staging_root=str(staging))

    assert adopted is False
    assert live.read_text(encoding="utf-8") == "changed independently"


def test_manifest_records_live_and_proposed_hashes(tmp_path):
    staging = tmp_path / "staging"
    staging.mkdir()
    live = tmp_path / "skill.md"
    live.write_text("old", encoding="utf-8")
    staged_dir = Path(
        write_staging_report("skill", _report("new"), "new", str(live), str(staging))
    )

    manifest = json.loads((staged_dir / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["live_skill_sha256"] == _sha256(live)
    assert manifest["proposed_skill_sha256"] == hashlib.sha256(b"new").hexdigest()
    assert manifest["status"] == "staged"
