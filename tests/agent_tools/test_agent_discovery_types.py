from __future__ import annotations

from agent_tools import skills


def test_manifest_and_list_preserve_raw_values(tmp_path, monkeypatch):
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text("skills: {}\n")
    raw = {"skills": {2: {"path": {"raw": True}, "task_types": ("task",)}, 1: {}}}
    monkeypatch.setattr(skills.yaml, "load", lambda text, Loader: raw)
    assert skills.load_manifest(manifest_path) is raw
    monkeypatch.setattr(skills, "load_manifest", lambda: raw)

    result = skills.list_skills()

    assert result[0] == {"name": 1, "path": None, "task_types": []}
    assert result[1]["name"] == 2
    for field in ("path", "task_types"):
        assert result[1][field] is raw["skills"][2][field]


def test_missing_manifest_omits_skills(tmp_path, monkeypatch):
    monkeypatch.setattr(skills, "REPO_ROOT", tmp_path)
    assert skills.validate_skills() == {"ok": False, "issues": ["skills/manifest.yaml is missing."]}


def test_validation_preserves_discovery_mapping_identity(tmp_path, monkeypatch):
    (tmp_path / "skills").mkdir()
    (tmp_path / "skills/manifest.yaml").write_text("skills: {}\n")
    summaries = [{"name": "raw", "path": None, "task_types": {"task": 1}}]
    monkeypatch.setattr(skills, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(skills, "list_skills", lambda: summaries)

    result = skills.validate_skills()

    assert result == {"ok": True, "issues": [], "skills": summaries}
    assert result["skills"] is summaries
    assert result["skills"][0] is summaries[0]
