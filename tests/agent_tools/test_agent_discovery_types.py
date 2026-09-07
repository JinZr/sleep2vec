from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import textwrap

from agent_tools import skills


def test_manifest_and_list_preserve_raw_values(tmp_path, monkeypatch):
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text("skills: {}\n")
    raw = {"skills": {2: {"path": {"raw": True}, "owners": None, "task_types": ("task",)}, 1: {}}}
    monkeypatch.setattr(skills.yaml, "safe_load", lambda text: raw)
    assert skills.load_manifest(manifest_path) is raw
    monkeypatch.setattr(skills, "load_manifest", lambda: raw)

    result = skills.list_skills()

    assert result[0] == {"name": 1, "path": None, "owners": [], "task_types": [], "relevant_index": []}
    assert result[1]["name"] == 2
    for field in ("path", "owners", "task_types"):
        assert result[1][field] is raw["skills"][2][field]
    assert result[1]["relevant_index"] == []


def test_missing_manifest_omits_skills(tmp_path, monkeypatch):
    monkeypatch.setattr(skills, "REPO_ROOT", tmp_path)
    assert skills.validate_skills() == {"ok": False, "issues": ["skills/manifest.yaml is missing."]}


def test_validation_preserves_discovery_mapping_identity(tmp_path, monkeypatch):
    (tmp_path / "skills").mkdir()
    (tmp_path / "skills/manifest.yaml").write_text("skills: {}\n")
    summaries = [{"name": "raw", "path": None, "owners": {"owner": 1}}]
    monkeypatch.setattr(skills, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(skills, "list_skills", lambda: summaries)

    result = skills.validate_skills()

    assert result == {"ok": True, "issues": [], "skills": summaries}
    assert result["skills"] is summaries
    assert result["skills"][0] is summaries[0]


def test_discovery_mypy_consumer_contract(tmp_path):
    root = Path(__file__).resolve().parents[2]
    probe = tmp_path / "discovery_consumer.py"
    probe.write_text(textwrap.dedent("""\
            from typing import Any
            from typing_extensions import assert_type
            from agent_tools.repo import repo_summary
            from agent_tools.skills import SkillSummary, SkillValidationResult, list_skills, validate_skills

            summary = repo_summary()
            assert_type(summary["repo_root"], str)
            assert_type(summary["git"]["available"], bool)
            assert_type(summary["git"]["branch"], str)
            assert_type(summary["git"]["commit"], str)
            assert_type(summary["git"]["dirty"], bool)
            assert_type(summary["codex_index"]["path"], str)
            assert_type(summary["codex_index"]["exists"], bool)
            assert_type(summary["important_paths"]["skills_manifest"], str)
            assert_type(summary["python"]["version"], str)
            entries = list_skills()
            assert_type(entries, list[SkillSummary])
            assert_type(entries[0]["name"], Any)
            assert_type(entries[0]["path"], Any)
            assert_type(entries[0]["owners"], Any)
            assert_type(entries[0]["task_types"], Any)
            assert_type(entries[0]["relevant_index"], Any)
            validation = validate_skills()
            assert_type(validation["ok"], bool)
            assert_type(validation["issues"], list[str])
            if "skills" in validation:
                assert_type(validation["skills"], list[SkillSummary])
            missing: SkillValidationResult = {"ok": False, "issues": ["missing"]}
            """))
    command = [
        sys.executable,
        "-m",
        "mypy",
        "--config-file",
        str(root / "pyproject.toml"),
        "--follow-imports=silent",
        "--no-incremental",
        str(probe),
    ]
    result = subprocess.run(command, cwd=root, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr

    with probe.open("a") as stream:
        stream.write(
            '\nsummary["git"]["available"] = "yes"\n'
            'summary["codex_index"]["missing"]\n'
            'summary["important_paths"]["tests"] = 1\n'
            'summary["python"]["version"] = 1\n'
            'entries[0]["unknown"]\n'
            'validation["issues"] = [1]\n'
            'invalid: SkillValidationResult = {"ok": True}\n'
        )
    result = subprocess.run(command, cwd=root, capture_output=True, text=True)
    assert result.returncode == 1, result.stdout + result.stderr
    assert result.stdout.count("error:") == 7, result.stdout + result.stderr
    assert result.stdout.count("[typeddict-item]") == 6, result.stdout
    assert result.stdout.count("[list-item]") == 1, result.stdout
