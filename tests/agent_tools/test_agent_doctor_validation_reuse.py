"""Doctor reuses plan's single validation pass: same verdict, doctor-only outputs, and one frozen config read."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from agent_tool_test_helpers import run_cli, write_finetune_recipe
import pytest
import yaml

from agent_tools import plans


def _recipe(tmp_path: Path, state: str) -> Path:
    recipe = write_finetune_recipe(tmp_path, include_label=state != "needs")
    if state == "fail":
        payload = yaml.safe_load(recipe.read_text())
        # The source config monitors val_ahi_pearson, so this explicit selection metric conflicts with it.
        payload["evaluation_policy"]["selection_metric"] = "val_loss"
        recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    return recipe


def _blocking(report: plans.DecisionReport) -> list[tuple[str, str, str]]:
    return [(issue.status.value, issue.field, issue.message) for issue in report.blocking_issues()]


@pytest.mark.parametrize(("state", "exit_code"), [("pass", 0), ("needs", 2), ("fail", 1)])
def test_doctor_and_plan_reach_the_same_verdict(tmp_path: Path, state: str, exit_code: int):
    recipe = _recipe(tmp_path, state)

    _recipe_payload, _cfg, doctor_report = plans.preflight_plan(recipe_path=recipe, output_dir=None)
    doctor = run_cli("doctor", "--recipe", str(recipe))
    plan_report = plans.build_plan(recipe_path=recipe, output_dir=tmp_path / "plan")

    assert doctor.returncode == doctor_report.exit_code == plan_report.exit_code == exit_code
    assert f"Status: {plan_report.status.value}" in doctor.stdout
    assert _blocking(doctor_report) == _blocking(plan_report)


def test_doctor_emits_decisions_template_on_needs_user_input(tmp_path: Path):
    recipe = _recipe(tmp_path, "needs")
    output_dir = tmp_path / "doctor"

    result = run_cli("doctor", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 2, result.stdout + result.stderr
    assert "Status: NEEDS_USER_INPUT" in result.stdout
    assert {path.name for path in output_dir.iterdir()} == {"questions.json", "questions.md", "decisions.yaml"}
    template = yaml.safe_load((output_dir / "decisions.yaml").read_text())["decisions"]
    assert template["label_name"]["value"] == "ASK_USER"
    assert "Fill it and rerun with --user-decisions" in result.stdout
    # Doctor publishes no plan artifacts into its own output directory.
    assert not (output_dir / "plan.json").exists()
    assert not (output_dir / "plan.blocked.md").exists()


def test_doctor_retry_reuses_its_output_directory_when_decisions_are_complete(tmp_path: Path):
    recipe = _recipe(tmp_path, "needs")
    doctor_dir = tmp_path / "doctor"
    template = doctor_dir / "decisions.yaml"
    assert run_cli("doctor", "--recipe", str(recipe), "--output-dir", str(doctor_dir)).returncode == 2
    payload = yaml.safe_load(template.read_text())
    payload["decisions"]["label_name"]["value"] = "ahi"
    template.write_text(yaml.safe_dump(payload, sort_keys=False))
    filled = template.read_bytes()

    retried = run_cli(
        "doctor", "--recipe", str(recipe), "--user-decisions", str(template), "--output-dir", str(doctor_dir)
    )

    assert retried.returncode == 0, retried.stdout + retried.stderr
    assert template.read_bytes() == filled


def test_plan_freezes_and_hashes_the_single_config_read(tmp_path: Path, monkeypatch):
    recipe = write_finetune_recipe(tmp_path)
    config = Path(yaml.safe_load(recipe.read_text())["inputs"]["config"])
    read_bytes = Path.read_bytes
    reads = []

    def read_then_edit(path: Path) -> bytes:
        data = read_bytes(path)
        if path == config:
            reads.append(data)
            # An edit after the only read must not reach the frozen plan.
            config.write_bytes(data + b"# edited after the plan read the config\n")
        return data

    monkeypatch.setattr(Path, "read_bytes", read_then_edit)
    output_dir = tmp_path / "plan"

    report = plans.build_plan(recipe_path=recipe, output_dir=output_dir)

    assert report.exit_code == 0, report.issues
    assert len(reads) == 1
    plan = json.loads((output_dir / "plan.json").read_text())
    run = plan["runs"][0]
    digest = hashlib.sha256(reads[0]).hexdigest()
    assert Path(run["config"]).read_bytes() == reads[0]
    assert run["config_sha256"] == digest
    snapshot = next(item for item in plan["recipe"]["input_snapshots"] if item["field"] == "inputs.config")
    assert snapshot == {"field": "inputs.config", "path": str(config), "sha256": digest}
