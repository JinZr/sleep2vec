from __future__ import annotations

from agent_tools.models import REPO_ROOT, SUPPORTED_VARIANTS
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


def _variants_that_parse_finetune_tuning() -> list[str]:
    return [
        variant
        for variant in SUPPORTED_VARIANTS
        if "FinetuneTuningConfig" in (REPO_ROOT / variant / "config.py").read_text()
    ]


def test_finetuning_skill_names_every_variant_with_a_finetune_entrypoint():
    skill = (REPO_ROOT / "skills/finetuning/SKILL.md").read_text()

    runnable = [variant for variant in SUPPORTED_VARIANTS if (REPO_ROOT / variant / "finetune.py").is_file()]
    assert "sex_age_baseline" in runnable
    for variant in runnable:
        assert f"python -m {variant}.finetune" in skill


def test_finetuning_skill_scopes_the_tuning_block_to_the_variants_that_parse_it():
    """`sex_age_baseline` finetunes without a `finetune.tuning` block, so the skill must say so.

    An unscoped checklist item sends the agent looking for a policy that variant's loader never
    reads, and there is nothing in its configs to find.
    """
    skill = (REPO_ROOT / "skills/finetuning/SKILL.md").read_text()

    parses_tuning = _variants_that_parse_finetune_tuning()
    assert parses_tuning == ["sleep2vec", "sleep2vec2", "sleep2expert"]
    mentions = [line for line in skill.splitlines() if "finetune.tuning" in line]
    assert mentions
    for line in mentions:
        assert all(variant in line for variant in parses_tuning), line


def test_hparam_guidance_separates_search_budget_from_launch_authority():
    agents = (REPO_ROOT / "AGENTS.md").read_text()
    contract = (REPO_ROOT / "doc/agent_contracts/task_recipe.md").read_text()
    skill = (REPO_ROOT / "skills/hyperparameter_tuning/SKILL.md").read_text()

    for guidance in (agents, contract, skill):
        assert "default 12-run search budget" in guidance
        assert "default 12-run launch budget" not in guidance
    assert "Launch requires an explicit request" in agents
    assert "Launch requires an explicit request" in skill
    assert "does not authorize publication or launch" in contract


def test_user_decision_guidance_preserves_explicit_final_test_unlock():
    contract = (REPO_ROOT / "doc/agent_contracts/user_decisions.md").read_text()

    assert "A concrete user-authorized `final_eval_unlock` value retains" in contract
    assert "its task-owned final-test semantics" in contract
