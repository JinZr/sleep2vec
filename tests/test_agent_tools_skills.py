from __future__ import annotations

from agent_tools.skills import validate_skills


def test_skills_validate_repository_skill_folder():
    result = validate_skills()

    assert result["ok"], result["issues"]
    assert any(item["name"] == "finetuning" for item in result["skills"])
