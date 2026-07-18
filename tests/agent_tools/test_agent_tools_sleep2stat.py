from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from agent_tool_test_helpers import write_yaml
import yaml

from agent_tools.configs import sleep2stat_config_summary
from agent_tools.models import REPO_ROOT
from agent_tools.plans import build_context, build_plan, evaluate_recipe
from agent_tools.skills import validate_skills

TINY_RECIPE = REPO_ROOT / "recipes/examples/tiny_fixture_sleep2stat.yaml"


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "agent_tools", *args], text=True, capture_output=True)


def _tiny_recipe_payload() -> dict:
    return yaml.safe_load(TINY_RECIPE.read_text())


def _write_tiny_recipe(tmp_path: Path, payload: dict) -> Path:
    return write_yaml(tmp_path / "tiny_sleep2stat.yaml", payload)


def _write_tiny_recipe_with_run_dir(tmp_path: Path, run_dir: Path, *, overwrite_policy: bool = False) -> Path:
    config_payload = yaml.safe_load((REPO_ROOT / "recipes/examples/fixtures/tiny_sleep2stat_config.yaml").read_text())
    config_payload["run"]["output_dir"] = str(run_dir)
    config = write_yaml(tmp_path / "tiny_sleep2stat_config.yaml", config_payload)
    payload = _tiny_recipe_payload()
    payload["inputs"]["config"] = str(config)
    payload["artifacts"]["run_dir"] = str(run_dir)
    payload["artifacts"]["overwrite"] = overwrite_policy
    payload["decisions"]["overwrite_policy"]["value"] = overwrite_policy
    return _write_tiny_recipe(tmp_path, payload)


def _write_context_decisions(tmp_path: Path) -> Path:
    return write_yaml(
        tmp_path / "decisions.yaml",
        {
            "decisions": {
                "external_test_locked": {"value": True, "source": "explicit_user"},
                "sleep2stat_split_policy": {"value": "descriptive test split only", "source": "explicit_user"},
                "sleep2stat_metric_use_policy": {
                    "value": "signal-derived proxy metrics only",
                    "source": "explicit_user",
                },
                "overwrite_policy": {"value": False, "source": "explicit_user"},
            },
        },
    )


def test_sleep2stat_omitted_variant_passes_consultation_gates():
    recipe, cfg, report = evaluate_recipe(TINY_RECIPE)

    assert "variant" not in recipe
    assert cfg is not None and cfg["is_sleep2stat"] is True
    assert report.exit_code == 0


def test_sleep2stat_variant_value_fails(tmp_path: Path):
    payload = _tiny_recipe_payload()
    payload["variant"] = "sleep2stat"
    recipe_path = _write_tiny_recipe(tmp_path, payload)

    _recipe, _cfg, report = evaluate_recipe(recipe_path)

    assert report.exit_code == 1
    assert any(issue.field == "variant" for issue in report.issues)


def test_sleep2stat_any_non_null_variant_fails(tmp_path: Path):
    payload = _tiny_recipe_payload()
    payload["variant"] = "sleep2vec"
    recipe_path = _write_tiny_recipe(tmp_path, payload)

    _recipe, _cfg, report = evaluate_recipe(recipe_path)

    assert report.exit_code == 1
    assert any(issue.field == "variant" for issue in report.issues)


def test_sleep2stat_wrong_config_writes_blocked_plan_without_private_config_bytes(tmp_path: Path):
    payload = _tiny_recipe_payload()
    experiment_root = tmp_path / "experiment"
    payload["experiment"]["root"] = str(experiment_root)
    payload["inputs"]["config"] = str(REPO_ROOT / "recipes/examples/fixtures/tiny_finetune_config.yaml")
    recipe_path = _write_tiny_recipe(tmp_path, payload)
    output_dir = experiment_root / "plans" / "wrong-config"

    report = build_plan(recipe_path=recipe_path, output_dir=output_dir)

    assert report.status.value == "FAIL"
    assert report.exit_code == 1
    issue = next(
        item for item in report.blocking_issues() if item.message == "task=sleep2stat requires a sleep2stat config."
    )
    assert "_source_config_bytes" not in issue.evidence["config_summary"]
    questions = json.loads((output_dir / "questions.json").read_text())
    question = next(item for item in questions["questions"] if item["message"] == issue.message)
    assert "_source_config_bytes" not in question["evidence"]["config_summary"]
    assert (output_dir / "plan.blocked.md").exists()


def test_sleep2stat_run_dir_mismatch_blocks_before_command_generation(tmp_path: Path):
    payload = _tiny_recipe_payload()
    payload["artifacts"]["run_dir"] = "results/sleep2stat/wrong"
    recipe_path = _write_tiny_recipe(tmp_path, payload)
    output_dir = tmp_path / "plan"

    report = build_plan(recipe_path=recipe_path, output_dir=output_dir)

    assert report.exit_code == 2
    assert any(issue.field == "artifacts.run_dir" for issue in report.issues)
    assert (output_dir / "plan.blocked.md").exists()
    assert not (output_dir / "run.sh").exists()


def test_sleep2stat_existing_run_output_dir_blocks_before_command_generation(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "old.txt").write_text("old")
    recipe_path = _write_tiny_recipe_with_run_dir(tmp_path, run_dir)
    output_dir = tmp_path / "plan"

    report = build_plan(recipe_path=recipe_path, output_dir=output_dir)

    assert report.exit_code == 2
    assert any(issue.field == "sleep2stat.run.output_dir" for issue in report.issues)
    assert (output_dir / "plan.blocked.md").exists()
    assert not (output_dir / "run.sh").exists()


def test_sleep2stat_existing_run_output_dir_blocks_even_with_overwrite_policy(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "old.txt").write_text("old")
    recipe_path = _write_tiny_recipe_with_run_dir(tmp_path, run_dir, overwrite_policy=True)
    output_dir = tmp_path / "plan"

    report = build_plan(recipe_path=recipe_path, output_dir=output_dir)

    assert report.exit_code == 2
    assert any(issue.field == "sleep2stat.run.output_dir" for issue in report.issues)
    assert not (output_dir / "run.sh").exists()


def test_sleep2stat_empty_run_output_dir_allows_command_generation(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    recipe_path = _write_tiny_recipe_with_run_dir(tmp_path, run_dir)
    output_dir = tmp_path / "plan"

    report = build_plan(recipe_path=recipe_path, output_dir=output_dir)

    assert report.exit_code == 0
    assert (output_dir / "run.sh").exists()


def test_sleep2stat_summarize_and_plot_use_config_run_dir(tmp_path: Path):
    payload = _tiny_recipe_payload()
    payload["runtime"]["dry_run"] = False
    payload["artifacts"].pop("run_dir")
    recipe_path = _write_tiny_recipe(tmp_path, payload)
    output_dir = tmp_path / "plan"

    report = build_plan(recipe_path=recipe_path, output_dir=output_dir)

    assert report.exit_code == 0
    commands = json.loads((output_dir / "plan.json").read_text())["commands"]
    config_run_dir = "results/sleep2stat/tiny_fixture"
    assert f"python -m sleep2stat summarize --run-dir {config_run_dir}" in commands
    assert any(
        command.startswith(f"python -m sleep2stat plot-cohort --run-dir {config_run_dir}") for command in commands
    )
    assert not any("--stage-source" in command for command in commands)


def test_sleep2stat_dry_run_plan_skips_post_run_commands(tmp_path: Path):
    payload = _tiny_recipe_payload()
    payload["runtime"]["dry_run"] = True
    recipe_path = _write_tiny_recipe(tmp_path, payload)
    output_dir = tmp_path / "plan"

    report = build_plan(recipe_path=recipe_path, output_dir=output_dir)

    assert report.exit_code == 0
    commands = json.loads((output_dir / "plan.json").read_text())["commands"]
    assert any(command.startswith("python -m sleep2stat run ") and "--dry-run" in command for command in commands)
    assert not any(command.startswith("python -m sleep2stat summarize ") for command in commands)
    assert not any(command.startswith("python -m sleep2stat plot-cohort ") for command in commands)


def test_sleep2stat_plot_stage_source_auto_is_rendered_as_plain_value(tmp_path: Path):
    payload = _tiny_recipe_payload()
    payload["runtime"]["dry_run"] = False
    payload["runtime"]["plot_stage_source"] = "auto"
    recipe_path = _write_tiny_recipe(tmp_path, payload)
    output_dir = tmp_path / "plan"

    report = build_plan(recipe_path=recipe_path, output_dir=output_dir)

    assert report.exit_code == 0
    commands = json.loads((output_dir / "plan.json").read_text())["commands"]
    assert any("--stage-source auto" in command for command in commands)


def test_sleep2stat_yasa_plan_adds_record_preflight_and_summary_types(tmp_path: Path):
    index = tmp_path / "index.csv"
    index.write_text(
        "path,split,duration,source,subject_id,session_id,age,sex\nmissing.npz,test,120,tiny,S001,N1,60,1.0\n"
    )
    config = write_yaml(
        tmp_path / "sleep2stat_yasa.yaml",
        {
            "run": {"name": "yasa", "output_dir": str(tmp_path / "run")},
            "data": {
                "backend": "npz",
                "index": str(index),
                "split": ["test"],
                "path_column": "path",
                "duration_column": "duration",
                "split_column": "split",
                "source_column": "source",
                "record_id_columns": ["source", "subject_id", "session_id"],
                "metadata_columns": ["age", "sex"],
                "token_sec": 30,
                "max_tokens": 4,
            },
            "signals": {
                "channels": {
                    "eeg": {
                        "source": "eeg",
                        "sfreq": 100,
                        "kind": "eeg",
                        "input_dim": 3000,
                        "mne_name": "EEG",
                    }
                }
            },
            "analyzers": [{"name": "yasa_stage", "type": "yasa_stage", "input_channels": ["eeg"]}],
            "reducers": [{"name": "yasa_stats", "type": "hypnogram_stats", "source": "yasa_stage"}],
            "outputs": {"write_global_tables": True, "write_per_record": True, "compression": "gzip"},
        },
    )
    recipe = write_yaml(
        tmp_path / "recipe.yaml",
        {
            "name": "yasa_recipe",
            "task": "sleep2stat",
            "inputs": {"config": str(config), "split": ["test"]},
            "runtime": {"device": "cpu", "num_workers": 1, "limit_records": 1, "dry_run": True},
            "artifacts": {"run_dir": str(tmp_path / "run"), "overwrite": False},
            "execution": {"target": "local", "path_context": "local", "path_validation": "local"},
            "evaluation_policy": {"external_test_locked": True},
            "decisions": {
                "task": {"value": "sleep2stat", "source": "explicit_recipe"},
                "sleep2stat_split_policy": {"value": "descriptive", "source": "explicit_recipe"},
                "sleep2stat_metric_use_policy": {"value": "proxy", "source": "explicit_recipe"},
                "overwrite_policy": {"value": False, "source": "explicit_recipe"},
            },
        },
    )

    summary = sleep2stat_config_summary(config)
    assert "yasa_stage" in summary["sleep2stat"]["supported_analyzer_types"]
    assert summary["sleep2stat"]["reducers"][0]["type"] == "hypnogram_stats"

    output_dir = tmp_path / "plan"
    report = build_plan(recipe_path=recipe, output_dir=output_dir)

    assert report.exit_code == 0
    commands = json.loads((output_dir / "plan.json").read_text())["commands"]
    frozen_config = output_dir / "runs" / "run-000--yasa-recipe" / "config.yaml"
    assert commands[1] == (
        f"python -m sleep2stat validate-config --config {frozen_config} "
        "--check-records --split test --limit-records 1"
    )


def test_sleep2stat_plan_materializes_user_decision_config_override(tmp_path: Path):
    override_payload = yaml.safe_load((REPO_ROOT / "recipes/examples/fixtures/tiny_sleep2stat_config.yaml").read_text())
    override_payload["run"]["seed"] = 9001
    override_config = write_yaml(tmp_path / "other_sleep2stat_config.yaml", override_payload)
    decisions = write_yaml(
        tmp_path / "decisions.yaml",
        {"decisions": {"config": {"value": str(override_config), "source": "explicit_user"}}},
    )
    output_dir = tmp_path / "plan"
    recipe_path = _write_tiny_recipe(tmp_path, _tiny_recipe_payload())

    report = build_plan(recipe_path=recipe_path, output_dir=output_dir, user_decisions_path=decisions)

    assert report.exit_code == 0
    commands = json.loads((output_dir / "plan.json").read_text())["commands"]
    frozen_config = output_dir / "runs" / "run-000--tiny-fixture-sleep2stat" / "config.yaml"
    assert f"python -m sleep2stat validate-config --config {frozen_config}" in commands
    assert any(command.startswith(f"python -m sleep2stat run --config {frozen_config}") for command in commands)
    assert yaml.safe_load(frozen_config.read_text())["run"]["seed"] == 9001
    assert json.loads((output_dir / "plan.json").read_text())["recipe"]["inputs"]["config"] == str(override_config)
    assert str(override_config) not in "\n".join(commands)


def test_sleep2stat_missing_test_split_policy_blocks(tmp_path: Path):
    payload = _tiny_recipe_payload()
    payload["decisions"].pop("sleep2stat_split_policy")
    recipe_path = _write_tiny_recipe(tmp_path, payload)

    _recipe, _cfg, report = evaluate_recipe(recipe_path)

    assert report.exit_code == 2
    assert any(issue.field == "sleep2stat_split_policy" for issue in report.issues)


def test_sleep2stat_missing_metric_use_policy_blocks(tmp_path: Path):
    payload = _tiny_recipe_payload()
    payload["decisions"].pop("sleep2stat_metric_use_policy")
    recipe_path = _write_tiny_recipe(tmp_path, payload)

    _recipe, _cfg, report = evaluate_recipe(recipe_path)

    assert report.exit_code == 2
    assert any(issue.field == "sleep2stat_metric_use_policy" for issue in report.issues)


def test_sleep2stat_external_test_locked_false_blocks_test_split(tmp_path: Path):
    payload = _tiny_recipe_payload()
    payload["evaluation_policy"]["external_test_locked"] = False
    recipe_path = _write_tiny_recipe(tmp_path, payload)

    _recipe, _cfg, report = evaluate_recipe(recipe_path)

    assert report.exit_code == 2
    assert any(issue.field == "external_test_locked" for issue in report.issues)


def test_sleep2stat_config_summary_omits_removed_run_controls():
    summary = sleep2stat_config_summary(REPO_ROOT / "recipes/examples/fixtures/tiny_sleep2stat_config.yaml")

    assert "overwrite" not in summary["sleep2stat"]["run"]
    assert "skip_existing" not in summary["sleep2stat"]["run"]


def test_sleep2stat_kaldi_relative_manifest_is_not_resolved_under_data_root(tmp_path: Path):
    kaldi_root = tmp_path / "kaldi"
    kaldi_root.mkdir()
    (kaldi_root / "manifest.json").write_text('{"splits": {"test": {"manifest": "test.csv"}}}')
    config = write_yaml(
        tmp_path / "sleep2stat_kaldi.yaml",
        {
            "run": {
                "name": "kaldi_relative_manifest",
                "output_dir": str(tmp_path / "run"),
            },
            "data": {
                "backend": "kaldi",
                "kaldi_data_root": str(kaldi_root),
                "kaldi_manifest": "manifest.json",
                "split": ["test"],
                "path_column": "path",
                "duration_column": "duration",
                "split_column": "split",
                "token_sec": 30,
                "max_tokens": 4,
            },
            "signals": {
                "channels": {
                    "ppg": {
                        "source": "ppg",
                        "sfreq": 1,
                        "kind": "ppg",
                        "input_dim": 30,
                    }
                }
            },
            "analyzers": [
                {
                    "name": "stage_model",
                    "type": "sleep2vec_downstream",
                    "enabled": False,
                    "namespace": "sleep2vec2",
                    "label_name": "stage5",
                    "config": "configs/sleep2vec2/ppg_stage5_finetune_large.yaml",
                    "ckpt_path": str(tmp_path / "stage.ckpt"),
                    "input_channels": ["ppg"],
                }
            ],
            "reducers": [],
            "outputs": {
                "write_global_tables": True,
                "write_per_record": True,
                "compression": "gzip",
                "global_tables": {"event_alignment": True, "night_stats": True},
            },
        },
    )
    recipe = write_yaml(
        tmp_path / "recipe.yaml",
        {
            "name": "kaldi_relative_manifest",
            "task": "sleep2stat",
            "inputs": {"config": str(config), "split": ["test"]},
            "artifacts": {"run_dir": str(tmp_path / "run"), "overwrite": False},
            "execution": {"target": "local", "path_context": "local", "path_validation": "local"},
            "evaluation_policy": {"external_test_locked": True},
            "decisions": {
                "task": {"value": "sleep2stat", "source": "explicit_recipe"},
                "sleep2stat_split_policy": {"value": "descriptive test split only", "source": "explicit_recipe"},
                "sleep2stat_metric_use_policy": {
                    "value": "model outputs are proxy metrics",
                    "source": "explicit_recipe",
                },
                "overwrite_policy": {"value": False, "source": "explicit_recipe"},
            },
        },
    )

    _recipe, _cfg, report = evaluate_recipe(recipe)

    assert report.exit_code == 1
    assert any(issue.field == "sleep2stat.data.kaldi_manifest" for issue in report.issues)


def test_sleep2stat_placeholder_model_ckpt_blocks_as_agent_risk_issue(tmp_path: Path):
    index = tmp_path / "index.csv"
    index.write_text("path,split,duration,source,subject_id,session_id\n" "missing.npz,test,120,tiny,S001,N1\n")
    config = write_yaml(
        tmp_path / "sleep2stat_model.yaml",
        {
            "run": {
                "name": "placeholder_model",
                "output_dir": str(tmp_path / "run"),
            },
            "data": {
                "backend": "npz",
                "index": str(index),
                "split": ["test"],
                "path_column": "path",
                "duration_column": "duration",
                "split_column": "split",
                "token_sec": 30,
                "max_tokens": 4,
            },
            "signals": {
                "channels": {
                    "ppg": {
                        "source": "ppg",
                        "sfreq": 1,
                        "kind": "ppg",
                        "input_dim": 30,
                    }
                }
            },
            "analyzers": [
                {
                    "name": "stage_model",
                    "type": "sleep2vec_downstream",
                    "namespace": "sleep2vec2",
                    "label_name": "stage5",
                    "config": "configs/sleep2vec2/ppg_stage5_finetune_large.yaml",
                    "ckpt_path": "/path/to/stage5.ckpt",
                    "input_channels": ["ppg"],
                }
            ],
            "reducers": [],
            "outputs": {
                "write_global_tables": True,
                "write_per_record": True,
                "compression": "gzip",
                "global_tables": {"event_alignment": True, "night_stats": True},
            },
        },
    )
    recipe = write_yaml(
        tmp_path / "recipe.yaml",
        {
            "name": "placeholder_model_ckpt",
            "task": "sleep2stat",
            "inputs": {"config": str(config), "split": ["test"]},
            "artifacts": {"run_dir": str(tmp_path / "run"), "overwrite": False},
            "execution": {"target": "local", "path_context": "local", "path_validation": "local"},
            "evaluation_policy": {"external_test_locked": True},
            "decisions": {
                "task": {"value": "sleep2stat", "source": "explicit_recipe"},
                "sleep2stat_split_policy": {"value": "descriptive test split only", "source": "explicit_recipe"},
                "sleep2stat_metric_use_policy": {
                    "value": "model outputs are proxy metrics",
                    "source": "explicit_recipe",
                },
                "overwrite_policy": {"value": False, "source": "explicit_recipe"},
            },
        },
    )

    _recipe, _cfg, report = evaluate_recipe(recipe)

    assert report.exit_code == 2
    assert any(issue.field == "sleep2stat_config" and "ckpt_path" in issue.message for issue in report.issues)


def test_sleep2stat_direct_context_honors_user_decisions_and_summarizes_index(tmp_path: Path):
    decisions = _write_context_decisions(tmp_path)
    output_dir = tmp_path / "context"

    report = build_context(
        task="sleep2stat",
        config=REPO_ROOT / "recipes/examples/fixtures/tiny_sleep2stat_config.yaml",
        output_dir=output_dir,
        user_decisions_path=decisions,
    )

    assert report.exit_code == 2
    context = json.loads((output_dir / "context.json").read_text())
    assert context["index_summary"]["rows"] == 1
    assert context["index_summary"]["blocking_issues"] == []
    assert context["can_generate_commands"] is False
    assert context["recommended_commands"] == []
    assert any("experiment" in issue for issue in context["blocking_issues"])
    assert (output_dir / "commands.blocked.sh").exists()
    assert not (output_dir / "commands.sh").exists()
    assert not (output_dir / "validation.sh").exists()


def test_sleep2stat_skill_examples_validate_without_variant():
    result = validate_skills()

    assert result["ok"], result["issues"]


def test_sleep2stat_skill_documents_stable_sidecars():
    text = (REPO_ROOT / "skills/sleep2stat/SKILL.md").read_text()

    for expected in ["events.csv.gz", "events.csv", "night_stats.json", "result_manifest.csv"]:
        assert expected in text
    assert "_SUCCESS.json" not in text
    assert "arrays.npz" in text
    assert "run directories are single-use" in text


def test_sleep2stat_cli_skills_validate_accepts_examples():
    result = _run("skills", "--validate")

    assert result.returncode == 0, result.stdout + result.stderr


def test_sleep2stat_index_summary_runs_without_config():
    result = _run("index-summary", "--index", "recipes/examples/fixtures/tiny_sleep2stat_index.csv", "--json")

    assert result.returncode == 0, result.stdout + result.stderr
    assert (
        "index-summary --index <index> --config <config>" not in (REPO_ROOT / "skills/sleep2stat/SKILL.md").read_text()
    )
