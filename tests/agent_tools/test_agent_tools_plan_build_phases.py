from __future__ import annotations

import hashlib
import json
from pathlib import Path

from agent_tool_test_helpers import write_finetune_recipe
import pytest
import yaml

from agent_tools import plans
from agent_tools.adapters.base import TaskAdapter
from agent_tools.experiment_workspace import read_run_manifest


def test_single_run_deferred_plan_preserves_frozen_bytes_and_registration_boundary(tmp_path: Path):
    recipe_path = write_finetune_recipe(tmp_path / "workspace")
    recipe = yaml.safe_load(recipe_path.read_text())
    workspace = Path(recipe["experiment"]["root"])
    source_config = Path(recipe["inputs"]["config"])
    source_config_bytes = source_config.read_bytes()
    source_config_sha256 = hashlib.sha256(source_config_bytes).hexdigest()
    plan_dir = workspace / "plans" / "finetune"
    staging_dir = workspace / "plans" / ".finetune.staging"
    registered_recipe_path = plan_dir / "recipe.resolved.yaml"

    report = plans.build_plan(
        recipe_path=recipe_path,
        output_dir=plan_dir,
        source_config_sha256=source_config_sha256,
        staging_dir=staging_dir,
        defer_commit=True,
        registered_recipe_path=registered_recipe_path,
        plan_controller="pipeline",
        run_index_offset=7,
    )

    assert report.exit_code == 0
    assert not plan_dir.exists()
    assert staging_dir.is_dir()
    plan_bytes = (staging_dir / "plan.json").read_bytes()
    plan = json.loads(plan_bytes)
    run = plan["runs"][0]
    physical_run_dir = staging_dir / Path(run["script"]).parent.relative_to(plan_dir)
    assert plan_bytes == (json.dumps(plan, indent=2, sort_keys=True) + "\n").encode()
    assert plan["recipe"]["_recipe_path"] == str(registered_recipe_path.resolve())
    assert run["run_id"] == "run-007"
    assert (physical_run_dir / "config.yaml").read_bytes() == source_config_bytes
    assert (staging_dir / "run.sh").read_bytes() == (physical_run_dir / "launch.sh").read_bytes()
    assert (staging_dir / "run.sh").stat().st_mode & 0o111 == 0o111
    assert (physical_run_dir / "launch.sh").stat().st_mode & 0o111 == 0o111
    assert read_run_manifest(workspace) == []
    assert not (workspace / "steps" / recipe["step"]["id"] / "step.yaml").exists()


def test_bound_config_hash_failure_precedes_staging_and_materialization(tmp_path: Path, monkeypatch):
    recipe_path = write_finetune_recipe(tmp_path / "workspace")
    recipe = yaml.safe_load(recipe_path.read_text())
    workspace = Path(recipe["experiment"]["root"])
    plan_dir = workspace / "plans" / "finetune"
    staging_dir = workspace / "plans" / ".finetune.staging"

    monkeypatch.setattr(
        plans,
        "_materialize_single_run_plan",
        lambda **_kwargs: pytest.fail("materialization must not start before bound config validation"),
    )

    with pytest.raises(ValueError, match="externally bound SHA-256"):
        plans.build_plan(
            recipe_path=recipe_path,
            output_dir=plan_dir,
            source_config_sha256="0" * 64,
            staging_dir=staging_dir,
            defer_commit=True,
        )

    assert not plan_dir.exists()
    assert not staging_dir.exists()
    assert read_run_manifest(workspace) == []


class _PrecommitFailureAdapter(TaskAdapter):
    task = "precommit_failure"
    materializes_plan = True

    def __init__(self) -> None:
        self.calls: list[str] = []

    def write_plan(
        self,
        recipe,
        out,
        *,
        write_out=None,
        run_index_offset=None,
        unlock_final_test,
        source_config_bytes,
        source_config_sha256,
    ) -> None:
        self.calls.append("write")
        write_out.mkdir(parents=True)
        (write_out / "plan.json").write_text("{}\n")

    def precommit_plan(self, out, *, write_out):
        self.calls.append("precommit")
        raise ValueError("injected precommit failure")

    def commit_plan(self, out, *, preflight_validated=False) -> None:
        pytest.fail("failed precommit must not register the plan")


def test_adapter_precommit_failure_removes_staging_without_publication(tmp_path: Path):
    adapter = _PrecommitFailureAdapter()
    out = tmp_path / "plan"
    staging = tmp_path / ".plan.staging"
    report = plans.DecisionReport(status=plans.DecisionStatus.PASS)

    result = plans._materialize_adapter_plan(
        plan_adapter=adapter,
        recipe={},
        report=report,
        out=out,
        write_out=staging,
        output_identity=None,
        generated_staging=False,
        staging_dir=staging,
        defer_commit=False,
        plan_controller=None,
        run_index_offset=None,
        validate_only=False,
        unlock_final_test=False,
        validated_config_bytes=b"model: {}\n",
        validated_config_sha256=hashlib.sha256(b"model: {}\n").hexdigest(),
    )

    assert result.exit_code == 1
    assert adapter.calls == ["write", "precommit"]
    assert result.blocking_issues()[0].field == "execution.preflight"
    assert result.blocking_issues()[0].message == "injected precommit failure"
    assert not staging.exists()
    assert not out.exists()
