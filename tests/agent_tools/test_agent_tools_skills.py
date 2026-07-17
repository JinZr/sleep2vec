from __future__ import annotations

from agent_tools.models import REPO_ROOT
from agent_tools.skills import validate_skills

INDEX_FILES = {
    "README.md",
    "MODULE_MAP.md",
    "REUSE_GUIDE.md",
    "WORKFLOWS.md",
}
INDEX_PATHS = {f"doc/codex_index/{name}" for name in INDEX_FILES}


def test_skills_validate_repository_skill_folder():
    result = validate_skills()

    assert result["ok"], result["issues"]
    assert any(item["name"] == "finetuning" for item in result["skills"])


def test_codex_index_contains_only_shared_navigation_files():
    index_root = REPO_ROOT / "doc/codex_index"
    files = {
        path.relative_to(index_root).as_posix()
        for path in index_root.rglob("*")
        if path.is_file() and not any(part.startswith(".") for part in path.relative_to(index_root).parts)
    }

    assert files == INDEX_FILES


def test_skill_index_references_use_shared_navigation_files():
    result = validate_skills()

    assert result["ok"], result["issues"]
    for skill in result["skills"]:
        relevant_index = set(skill["relevant_index"])
        assert relevant_index
        assert relevant_index <= INDEX_PATHS
        assert all((REPO_ROOT / path).is_file() for path in relevant_index)
