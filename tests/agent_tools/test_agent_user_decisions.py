from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

from agent_tool_test_helpers import write_finetune_recipe, write_yaml
import pytest
import yaml

from agent_tools import cli as agent_cli
from agent_tools.decision_models import DecisionIssue, DecisionReport, DecisionStatus, ResolvedDecision
from agent_tools.decisions import user_decision_template
from agent_tools.plans import evaluate_recipe, write_user_decision_template
from agent_tools.recipes import load_consultation_policy


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "agent_tools", *args], text=True, capture_output=True)


def _write_decisions(tmp_path: Path, decisions: dict) -> Path:
    return write_yaml(tmp_path / "decisions.yaml", {"decisions": decisions})


def test_user_decision_yaml_resolves_missing_label_name(tmp_path: Path):
    recipe = write_finetune_recipe(tmp_path, include_label=False)
    decisions = _write_decisions(tmp_path, {"label_name": {"value": "ahi", "source": "explicit_user"}})

    result = _run(
        "doctor",
        "--recipe",
        str(recipe),
        "--user-decisions",
        str(decisions),
        "--output-dir",
        str(tmp_path / "doctor"),
    )

    assert result.returncode == 0
    assert "Status: PASS" in result.stdout


def test_user_decision_yaml_resolves_external_test_locked(tmp_path: Path):
    base = write_finetune_recipe(tmp_path)
    recipe = write_yaml(
        tmp_path / "tune.yaml",
        {
            "name": "unit_tune",
            "task": "hparam_tune",
            "variant": "sleep2vec",
            "base_recipe": str(base),
            "search": {"method": "grid", "max_runs": 1, "parameters": {"runtime.lr": [1e-6]}},
            "evaluation_policy": {
                "selection_metric": "val_ahi_pearson",
                "selection_mode": "max",
                "selection_split": "val",
                "test_after_fit": False,
                "final_eval_split": "validation",
                "final_test_unlocked": False,
                "require_manual_unlock_for_final_test": True,
            },
            "decisions": {
                "task": {"value": "hparam_tune", "source": "explicit_recipe"},
                "label_name": {"value": "ahi", "source": "explicit_recipe"},
                "train_val_test_policy": {"value": "select on val", "source": "explicit_recipe"},
                "overwrite_policy": {"value": False, "source": "explicit_recipe"},
                "final_eval_unlock": {"value": False, "source": "explicit_recipe"},
            },
        },
    )
    decisions = _write_decisions(tmp_path, {"external_test_locked": {"value": True, "source": "explicit_user"}})

    result = _run(
        "doctor",
        "--recipe",
        str(recipe),
        "--user-decisions",
        str(decisions),
        "--output-dir",
        str(tmp_path / "doctor"),
    )

    assert result.returncode == 0
    assert "Status: PASS" in result.stdout


def test_recipe_ask_user_always_blocks(tmp_path: Path):
    recipe = write_finetune_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    payload["decisions"]["label_name"] = {
        "value": "ASK_USER",
        "source": "unresolved",
        "question": "Which label?",
    }
    write_yaml(recipe, payload)

    result = _run("doctor", "--recipe", str(recipe), "--output-dir", str(tmp_path / "doctor"))

    assert result.returncode == 2
    assert "ASK_USER" in result.stdout


def test_doctor_writes_fillable_user_decision_template(tmp_path: Path):
    recipe = write_finetune_recipe(tmp_path, include_label=False)
    output_dir = tmp_path / "doctor"

    result = _run("doctor", "--recipe", str(recipe), "--output-dir", str(output_dir))

    template = output_dir / "decisions.yaml"
    assert result.returncode == 2
    assert str(template) in result.stdout
    assert yaml.safe_load(template.read_text()) == {
        "decisions": {
            "label_name": {
                "value": "ASK_USER",
                "source": "explicit_user",
                "question": (
                    "Which label should this task use, for example ahi, age, sex, stage5, "
                    "or a custom metadata label?"
                ),
            }
        }
    }

    unresolved = _run("doctor", "--recipe", str(recipe), "--user-decisions", str(template))
    assert unresolved.returncode == 2
    assert "Status: NEEDS_USER_INPUT" in unresolved.stdout

    payload = yaml.safe_load(template.read_text())
    payload["decisions"]["label_name"]["value"] = "ahi"
    template.write_text(yaml.safe_dump(payload, sort_keys=False))
    resolved = _run("doctor", "--recipe", str(recipe), "--user-decisions", str(template))
    assert resolved.returncode == 0
    assert "Status: PASS" in resolved.stdout


def test_doctor_template_preserves_explicit_user_decisions_and_metadata(tmp_path: Path):
    recipe = write_finetune_recipe(tmp_path, include_label=False)
    decisions = _write_decisions(
        tmp_path,
        {
            "label_name": {
                "value": "ASK_USER",
                "source": "explicit_user",
                "question": "Which outcome?",
                "rationale": "Outcome is pending review.",
            },
            "overwrite_policy": {"value": False, "source": "explicit_user"},
        },
    )
    output_dir = tmp_path / "doctor"

    result = _run(
        "doctor",
        "--recipe",
        str(recipe),
        "--user-decisions",
        str(decisions),
        "--output-dir",
        str(output_dir),
    )

    assert result.returncode == 2
    assert yaml.safe_load((output_dir / "decisions.yaml").read_text()) == {
        "decisions": {
            "label_name": {
                "value": "ASK_USER",
                "source": "explicit_user",
                "question": "Which outcome?",
                "rationale": "Outcome is pending review.",
            },
            "overwrite_policy": {"value": False, "source": "explicit_user"},
        }
    }


def test_doctor_does_not_overwrite_existing_user_decisions_file(tmp_path: Path):
    recipe = write_finetune_recipe(tmp_path, include_label=False)
    output_dir = tmp_path / "doctor"
    output_dir.mkdir()
    template = output_dir / "decisions.yaml"
    original = "decisions:\n  label_name:\n    value: ahi\n"
    template.write_text(original)

    result = _run("doctor", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 2
    assert "Preserved existing user decisions file" in result.stdout
    assert template.read_text() == original


def test_plan_cli_does_not_advertise_stale_user_decisions_file(tmp_path: Path, monkeypatch, capsys):
    output_dir = tmp_path / "plan"
    output_dir.mkdir()
    stale_template = output_dir / "decisions.yaml"
    stale_template.write_text("stale template\n")
    report = DecisionReport(
        status=DecisionStatus.NEEDS_USER_INPUT,
        issues=[DecisionIssue(DecisionStatus.NEEDS_USER_INPUT, "step.purpose", "Step purpose is missing.")],
    )
    monkeypatch.setattr(agent_cli, "build_plan", lambda **_kwargs: report)

    exit_code = agent_cli.main(["plan", "--recipe", str(tmp_path / "recipe.yaml"), "--output-dir", str(output_dir)])

    stdout = capsys.readouterr().out
    assert exit_code == 2
    assert str(stale_template) not in stdout
    assert "Fill it and rerun" not in stdout


def test_doctor_does_not_overwrite_user_decisions_created_during_publication(tmp_path: Path, monkeypatch):
    recipe_path = write_finetune_recipe(tmp_path, include_label=False)
    recipe, _cfg, report = evaluate_recipe(recipe_path)
    output_dir = tmp_path / "doctor"
    template = output_dir / "decisions.yaml"
    real_link = os.link

    def competing_link(source, destination, *args, **kwargs):
        if Path(destination) == template:
            template.write_text("user competitor\n")
        return real_link(source, destination, *args, **kwargs)

    monkeypatch.setattr(os, "link", competing_link)

    assert write_user_decision_template(output_dir, recipe, report, preserve_existing=True) == (template, False)
    assert template.read_text() == "user competitor\n"
    assert not list(output_dir.glob(".decisions.yaml.*.tmp"))


def test_user_decision_template_is_complete_when_published(tmp_path: Path, monkeypatch):
    recipe_path = write_finetune_recipe(tmp_path, include_label=False)
    recipe, _cfg, report = evaluate_recipe(recipe_path)
    output_dir = tmp_path / "doctor"
    template = output_dir / "decisions.yaml"
    real_link = os.link
    published_bytes = b""

    def inspect_link(source, destination, *args, **kwargs):
        nonlocal published_bytes
        created = real_link(source, destination, *args, **kwargs)
        published_bytes = Path(destination).read_bytes()
        assert yaml.safe_load(published_bytes)["decisions"]["label_name"]["value"] == "ASK_USER"
        return created

    monkeypatch.setattr(os, "link", inspect_link)

    assert write_user_decision_template(output_dir, recipe, report, preserve_existing=True) == (template, True)
    assert template.read_bytes() == published_bytes
    assert not list(output_dir.glob(".decisions.yaml.*.tmp"))


@pytest.mark.parametrize("competitor_kind", ["symlink", "hardlink"])
def test_doctor_rejects_aliased_user_decisions_created_during_publication(
    tmp_path: Path,
    monkeypatch,
    competitor_kind: str,
):
    recipe_path = write_finetune_recipe(tmp_path, include_label=False)
    recipe, _cfg, report = evaluate_recipe(recipe_path)
    output_dir = tmp_path / "doctor"
    template = output_dir / "decisions.yaml"
    outside = tmp_path / "outside.yaml"
    outside.write_text("user competitor\n")
    real_link = os.link

    def competing_link(source, destination, *args, **kwargs):
        if Path(destination) == template:
            if competitor_kind == "symlink":
                template.symlink_to(outside)
            else:
                template.hardlink_to(outside)
        return real_link(source, destination, *args, **kwargs)

    monkeypatch.setattr(os, "link", competing_link)

    with pytest.raises(ValueError, match="Managed output paths"):
        write_user_decision_template(output_dir, recipe, report, preserve_existing=True)

    assert outside.read_text() == "user competitor\n"
    assert not list(output_dir.glob(".decisions.yaml.*.tmp"))


def test_doctor_rejects_output_directory_owned_by_pass_plan(tmp_path: Path):
    recipe = write_finetune_recipe(tmp_path)
    output_dir = tmp_path / "plan"
    planned = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))
    assert planned.returncode == 0
    before = {path.relative_to(output_dir): path.read_bytes() for path in output_dir.rglob("*") if path.is_file()}
    payload = yaml.safe_load(recipe.read_text())
    payload["inputs"].pop("label_name")
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))

    result = _run("doctor", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 1
    assert "doctor output requires a fresh --output-dir" in result.stderr
    assert {
        path.relative_to(output_dir): path.read_bytes() for path in output_dir.rglob("*") if path.is_file()
    } == before
    assert not (output_dir / "decisions.yaml").exists()


def test_doctor_rejects_registered_blocked_plan_directory_without_mutation(tmp_path: Path):
    blocked_recipe = write_finetune_recipe(tmp_path / "blocked-source", include_label=False)
    workspace = Path(yaml.safe_load(blocked_recipe.read_text())["experiment"]["root"])
    output_dir = workspace / "plans" / "blocked"
    blocked = _run("plan", "--recipe", str(blocked_recipe), "--output-dir", str(output_dir))
    assert blocked.returncode == 2, blocked.stderr
    before = {path.relative_to(output_dir): path.read_bytes() for path in output_dir.rglob("*") if path.is_file()}
    doctor_recipe = write_finetune_recipe(tmp_path / "doctor-source", include_label=False)

    result = _run("doctor", "--recipe", str(doctor_recipe), "--output-dir", str(output_dir))

    assert result.returncode == 1
    assert "doctor output requires a fresh --output-dir" in result.stderr
    assert {
        path.relative_to(output_dir): path.read_bytes() for path in output_dir.rglob("*") if path.is_file()
    } == before


def test_doctor_rejects_unregistered_blocked_plan_marker(tmp_path: Path):
    recipe = write_finetune_recipe(tmp_path / "source", include_label=False)
    output_dir = tmp_path / "doctor"
    output_dir.mkdir()
    marker = output_dir / "plan.blocked.md"
    marker.write_text("user marker\n")

    result = _run("doctor", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 1
    assert "doctor output requires a fresh --output-dir" in result.stderr
    assert marker.read_text() == "user marker\n"
    assert not (output_dir / "questions.json").exists()
    assert not (output_dir / "decisions.yaml").exists()


def test_user_decision_template_skips_non_decisions_and_deduplicates_base_issue():
    report = DecisionReport(
        status=DecisionStatus.NEEDS_USER_INPUT,
        issues=[
            DecisionIssue(
                DecisionStatus.NEEDS_USER_INPUT,
                "base_finetune.label_name",
                "label_name is not explicitly resolved.",
                "Which label?",
                {"user_decision_field": "label_name"},
            ),
            DecisionIssue(
                DecisionStatus.NEEDS_USER_INPUT,
                "base_finetune.label_name",
                "label_name is still unresolved.",
                "Which label?",
                {"user_decision_field": "label_name"},
            ),
            DecisionIssue(DecisionStatus.NEEDS_USER_INPUT, "data_input", "Data input is missing."),
        ],
        decisions={"label_name": ResolvedDecision("label_name", None, "missing", "none")},
    )

    assert user_decision_template("hparam_tune", report, load_consultation_policy(), {}) == {
        "decisions": {
            "label_name": {
                "value": "ASK_USER",
                "source": "explicit_user",
                "question": "Which label?",
            }
        }
    }


def test_user_decision_template_requires_pure_needs_status():
    report = DecisionReport(
        status=DecisionStatus.FAIL,
        issues=[
            DecisionIssue(DecisionStatus.NEEDS_USER_INPUT, "label_name", "Label is missing.", "Which label?"),
            DecisionIssue(DecisionStatus.FAIL, "config", "Config is invalid."),
        ],
        decisions={"label_name": ResolvedDecision("label_name", None, "missing", "none")},
    )

    assert user_decision_template("finetune", report, load_consultation_policy(), {}) == {}


def test_user_decision_file_requires_decisions_mapping_before_output(tmp_path: Path):
    recipe = write_finetune_recipe(tmp_path)
    decisions = tmp_path / "decisions.yaml"
    decisions.write_text("label_name: ahi\n")
    output_dir = tmp_path / "doctor"

    result = _run(
        "doctor",
        "--recipe",
        str(recipe),
        "--user-decisions",
        str(decisions),
        "--output-dir",
        str(output_dir),
    )

    assert result.returncode == 1
    assert "must contain a decisions mapping" in result.stderr
    assert not output_dir.exists()


def test_user_task_cannot_override_explicit_recipe_task(tmp_path: Path):
    recipe = write_finetune_recipe(tmp_path)
    decisions = _write_decisions(tmp_path, {"task": {"value": "evaluate", "source": "explicit_user"}})

    effective, _cfg, report = evaluate_recipe(recipe, decisions)

    assert effective["task"] == "finetune"
    assert report.exit_code == 1
    assert any(issue.field == "task" and "conflicts" in issue.message for issue in report.blocking_issues())


def test_user_task_fills_missing_recipe_task(tmp_path: Path):
    recipe = write_finetune_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    payload.pop("task")
    recipe.write_text(yaml.safe_dump(payload))
    decisions = _write_decisions(tmp_path, {"task": {"value": "finetune", "source": "explicit_user"}})

    effective, _cfg, report = evaluate_recipe(recipe, decisions)

    assert report.exit_code == 0
    assert effective["task"] == "finetune"


def test_generated_task_template_remains_unresolved_until_filled(tmp_path: Path):
    recipe = write_finetune_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    payload.pop("task")
    payload["decisions"].pop("task")
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    output_dir = tmp_path / "doctor"

    initial = _run("doctor", "--recipe", str(recipe), "--output-dir", str(output_dir))

    template = output_dir / "decisions.yaml"
    assert initial.returncode == 2
    assert yaml.safe_load(template.read_text())["decisions"]["task"]["value"] == "ASK_USER"

    unresolved = _run("doctor", "--recipe", str(recipe), "--user-decisions", str(template))

    assert unresolved.returncode == 2
    assert "Status: NEEDS_USER_INPUT" in unresolved.stdout
    assert "Unsupported task" not in unresolved.stdout

    decisions = yaml.safe_load(template.read_text())
    decisions["decisions"]["task"]["value"] = "finetune"
    template.write_text(yaml.safe_dump(decisions, sort_keys=False))
    resolved = _run("doctor", "--recipe", str(recipe), "--user-decisions", str(template))

    assert resolved.returncode == 0
    assert "Status: PASS" in resolved.stdout


def test_doctor_writes_task_template_for_explicit_ask_user_sentinel(tmp_path: Path):
    recipe = write_finetune_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    payload["task"] = "ASK_USER"
    payload["decisions"]["task"] = {"value": "ASK_USER", "source": "unresolved"}
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    output_dir = tmp_path / "doctor"

    result = _run("doctor", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 2
    assert yaml.safe_load((output_dir / "decisions.yaml").read_text())["decisions"]["task"] == {
        "value": "ASK_USER",
        "source": "explicit_user",
        "question": (
            "Which task should be performed: preset_prepare, finetune, infer, evaluate, "
            "hparam_tune, sleep2stat, or embedding_extraction?"
        ),
    }


def test_task_template_preserves_user_decisions_not_evaluated_after_task_blocker(tmp_path: Path):
    recipe = write_finetune_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    payload["task"] = "ASK_USER"
    payload["decisions"]["task"] = {"value": "ASK_USER", "source": "unresolved"}
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    decisions = _write_decisions(
        tmp_path,
        {
            "task": {"value": "ASK_USER", "source": "explicit_user"},
            "label_name": {"value": "ahi", "source": "explicit_user", "meaning": "Primary outcome."},
            "overwrite_policy": {"value": False, "source": "explicit_user"},
        },
    )
    output_dir = tmp_path / "doctor"

    result = _run(
        "doctor",
        "--recipe",
        str(recipe),
        "--user-decisions",
        str(decisions),
        "--output-dir",
        str(output_dir),
    )

    assert result.returncode == 2
    template = yaml.safe_load((output_dir / "decisions.yaml").read_text())["decisions"]
    assert template["label_name"] == {
        "value": "ahi",
        "source": "explicit_user",
        "meaning": "Primary outcome.",
    }
    assert template["overwrite_policy"] == {"value": False, "source": "explicit_user"}


def test_user_split_decision_requires_concrete_split(tmp_path: Path):
    recipe = write_finetune_recipe(tmp_path)
    decisions = _write_decisions(
        tmp_path,
        {"train_val_test_policy": {"value": "select on val", "source": "explicit_user"}},
    )

    effective, _cfg, report = evaluate_recipe(recipe, decisions)

    assert effective["evaluation_policy"]["selection_split"] == "val"
    assert report.exit_code == 1
    assert any(issue.field == "train_val_test_policy" for issue in report.blocking_issues())


def test_user_split_decision_materializes_selection_split(tmp_path: Path):
    recipe = write_finetune_recipe(tmp_path)
    decisions = _write_decisions(
        tmp_path,
        {"train_val_test_policy": {"value": "train", "source": "explicit_user"}},
    )

    effective, _cfg, report = evaluate_recipe(recipe, decisions)

    assert report.exit_code == 0
    assert effective["evaluation_policy"]["selection_split"] == "train"
