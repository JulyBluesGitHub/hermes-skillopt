"""Skill-path resolution: one resolver, honored overrides, no silent guessing.

These cover the class of bug where the optimizer reads one file while the stager
records another as the adoption target — the safety hashes cannot catch that,
because each is computed against a different file.
"""

import pytest

from hermes_skillopt.sleep import (
    AmbiguousSkillError,
    find_skill_path,
    load_hermes_skill,
    resolve_skills_dir,
)


def _make_skill(root, *parts, body="# skill"):
    path = root.joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


def test_nested_and_flat_layouts_both_resolve(tmp_path):
    nested = _make_skill(tmp_path, "research", "arxiv", "SKILL.md", body="nested")
    flat = _make_skill(tmp_path, "solo", "SKILL.md", body="flat")

    assert find_skill_path("arxiv", skills_dir=str(tmp_path)) == str(nested)
    assert find_skill_path("solo", skills_dir=str(tmp_path)) == str(flat)


def test_bare_markdown_file_is_the_last_resort(tmp_path):
    bare = tmp_path / "quickref.md"
    bare.write_text("bare", encoding="utf-8")

    assert find_skill_path("quickref", skills_dir=str(tmp_path)) == str(bare)
    assert load_hermes_skill("quickref", skills_dir=str(tmp_path)) == "bare"


def test_missing_skill_resolves_to_empty(tmp_path):
    assert find_skill_path("nope", skills_dir=str(tmp_path)) == ""
    assert load_hermes_skill("nope", skills_dir=str(tmp_path)) == ""


def test_duplicate_skill_names_refuse_to_guess(tmp_path):
    """A real Hermes install can hold two skills with the same name in different
    categories. Picking whichever the filesystem yields first would edit an
    arbitrary one of them."""
    _make_skill(tmp_path, "github", "SKILL.md", body="top level")
    _make_skill(tmp_path, "software-development", "github", "SKILL.md", body="nested")

    with pytest.raises(AmbiguousSkillError) as excinfo:
        find_skill_path("github", skills_dir=str(tmp_path))

    assert "matches 2 files" in str(excinfo.value)


def test_hermes_home_drives_the_skills_dir(tmp_path, monkeypatch):
    monkeypatch.delenv("HERMES_HOME", raising=False)
    home = tmp_path / "hermes"
    _make_skill(home, "skills", "demo", "SKILL.md", body="from home")

    assert resolve_skills_dir(hermes_home=str(home)) == str(home / "skills")
    assert load_hermes_skill("demo", hermes_home=str(home)) == "from home"


def test_env_hermes_home_is_honored(tmp_path, monkeypatch):
    home = tmp_path / "envhome"
    _make_skill(home, "skills", "demo", "SKILL.md", body="from env")
    monkeypatch.setenv("HERMES_HOME", str(home))

    assert resolve_skills_dir() == str(home / "skills")
    assert load_hermes_skill("demo") == "from env"


def test_explicit_skills_dir_wins_over_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    _make_skill(home, "skills", "demo", "SKILL.md", body="home copy")
    override = tmp_path / "override"
    _make_skill(override, "demo", "SKILL.md", body="override copy")
    monkeypatch.setenv("HERMES_HOME", str(home))

    assert load_hermes_skill("demo", skills_dir=str(override)) == "override copy"
