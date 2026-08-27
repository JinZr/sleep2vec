from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

from agent_tool_test_helpers import (
    run_execution_preflight_fixture,
    survival_config_payload,
    write_finetune_recipe,
    write_survival_sidecars,
    write_yaml,
)
import pytest
import yaml

from agent_tools import managed_scheduler
from agent_tools.models import REPO_ROOT

_RUNTIME_COMMIT = subprocess.run(
    ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, text=True, capture_output=True
).stdout.strip()


@pytest.fixture(autouse=True)
def _stub_execution_target(monkeypatch):
    monkeypatch.setattr(managed_scheduler, "run_execution_command", run_execution_preflight_fixture)


def _run(*args: str) -> subprocess.CompletedProcess:
    runner = Path(__file__).with_name("agent_tools_cli_stub.py")
    return subprocess.run([sys.executable, str(runner), *args], text=True, capture_output=True)


def _first_run(plan_dir: Path) -> dict:
    return json.loads((plan_dir / "plan.json").read_text())["runs"][0]


def _bound_config_summary(recipe: dict) -> dict:
    config_bytes = Path(recipe["inputs"]["config"]).read_bytes()
    return {
        "_source_config_bytes": config_bytes,
        "_source_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
    }


def _valid_final_config_bytes(tmp_path: Path) -> bytes:
    payload = yaml.safe_load((tmp_path / "config.yaml").read_text())
    payload["model"]["head"].update(
        {
            "channel_agg": {"name": "mean"},
            "temporal_agg": {"name": "mean"},
        }
    )
    return yaml.safe_dump(payload, sort_keys=False).encode()


def _hparam_recipe(
    tmp_path: Path,
    *,
    variant: str = "sleep2vec",
    parameters: dict | None = None,
    max_runs: int | str = 1,
    ckpt_path: Path | None = None,
    final_config_path: Path | None = None,
    selection_metric: str = "val_ahi_pearson",
    selection_mode: str = "max",
) -> Path:
    base = write_finetune_recipe(tmp_path, variant=variant)
    inputs = {"ckpt_path": str(ckpt_path)} if ckpt_path else {}
    if final_config_path is not None:
        inputs["final_eval_config_path"] = str(final_config_path)
    return write_yaml(
        tmp_path / f"tune_{variant}.yaml",
        {
            "name": f"unit_tune_{variant}",
            "task": "hparam_tune",
            "variant": variant,
            "base_recipe": str(base),
            "inputs": inputs,
            "search": {"method": "grid", "max_runs": max_runs, "parameters": parameters or {"runtime.lr": [1e-6]}},
            "evaluation_policy": {
                "selection_metric": selection_metric,
                "selection_mode": selection_mode,
                "selection_split": "val",
                "final_eval_split": "test",
                "external_test_locked": True,
                "test_after_fit": False,
                "final_test_unlocked": False,
                "require_manual_unlock_for_final_test": True,
            },
            "decisions": {
                "task": {"value": "hparam_tune", "source": "explicit_recipe"},
                "label_name": {"value": "ahi", "source": "explicit_recipe"},
                "external_test_locked": {"value": True, "source": "explicit_recipe"},
                "train_val_test_policy": {"value": "select on val", "source": "explicit_recipe"},
                "overwrite_policy": {"value": False, "source": "explicit_recipe"},
                "final_eval_unlock": {"value": False, "source": "explicit_recipe"},
            },
        },
    )


def _local_runtime_execution(workdir: Path) -> dict:
    return {"workdir": str(workdir), "python": sys.executable, "runtime_commit": _RUNTIME_COMMIT}


def _survival_recipe_with_missing_sidecar_key(tmp_path: Path) -> tuple[Path, Path]:
    index = tmp_path / "survival_index.csv"
    index.write_text("path,split,duration,eid,ppg_mask\na.npz,train,60,001,1\nb.npz,val,60,003,1\n")
    config = write_yaml(
        tmp_path / "survival_config.yaml",
        survival_config_payload(index, write_survival_sidecars(tmp_path)),
    )
    recipe = {
        "name": "unit_survival_missing_sidecar_key",
        "task": "finetune",
        "variant": "sleep2vec",
        "inputs": {"config": str(config), "label_name": "incident_cox", "pretrained_backbone_path": None},
        "runtime": {"devices": [0]},
        "artifacts": {"results_csv_path": str(tmp_path / "results.csv"), "version_name": "unit"},
        "evaluation_policy": {
            "selection_metric": "val_loss",
            "selection_mode": "min",
            "selection_split": "val",
            "external_test_locked": True,
            "test_after_fit": False,
        },
        "decisions": {
            "task": {"value": "finetune", "source": "explicit_recipe"},
            "label_name": {"value": "incident_cox", "source": "explicit_recipe"},
            "pretrained_backbone_path": {
                "value": None,
                "source": "explicit_recipe",
                "meaning": "train from scratch",
            },
            "train_val_test_policy": {"value": "select on val", "source": "explicit_recipe"},
            "overwrite_policy": {"value": False, "source": "explicit_recipe"},
        },
    }
    return write_yaml(tmp_path / "survival_recipe.yaml", recipe), config


def _bad_survival_sidecars() -> dict[str, str]:
    return {
        "disease_columns_index": "/path/to/disease_columns.txt",
        "event_time_index": "/path/to/event_time.csv",
        "is_event_index": "/path/to/is_event.csv",
        "has_label_index": "/path/to/has_label.csv",
    }


def _write_survival_config_with_bad_sidecars(tmp_path: Path, *, preset_path: Path | None = None) -> Path:
    index = tmp_path / "survival_runtime_index.csv"
    index.write_text("path,split,duration,eid,ppg_mask\na.npz,test,60,001,1\n")
    payload = survival_config_payload(index, _bad_survival_sidecars())
    if preset_path is not None:
        payload["data"]["finetune_preset_path"] = str(preset_path)
    return write_yaml(tmp_path / "bad_survival_config.yaml", payload)


def _write_infer_recipe(
    tmp_path: Path,
    config: Path,
    *,
    inference_preset_path: Path | None = None,
    eval_split: str = "test",
) -> Path:
    ckpt = tmp_path / "model.ckpt"
    ckpt.write_text("checkpoint")
    inputs = {
        "config": str(config),
        "label_name": "incident_cox",
        "ckpt_path": str(ckpt),
        "eval_split": eval_split,
    }
    if inference_preset_path is not None:
        inputs["inference_preset_path"] = str(inference_preset_path)
    return write_yaml(
        tmp_path / "infer_survival.yaml",
        {
            "name": "unit_infer_survival",
            "task": "infer",
            "variant": "sleep2vec",
            "inputs": inputs,
            "evaluation_policy": {"external_test_locked": False, "final_test_unlocked": True},
            "artifacts": {"overwrite": True},
            "decisions": {
                "task": {"value": "infer", "source": "explicit_recipe"},
                "label_name": {"value": "incident_cox", "source": "explicit_recipe"},
                "ckpt_path": {"value": str(ckpt), "source": "explicit_recipe"},
                "external_test_locked": {"value": False, "source": "explicit_recipe"},
                "final_eval_unlock": {"value": True, "source": "explicit_recipe"},
                "overwrite_policy": {"value": True, "source": "explicit_recipe"},
            },
        },
    )


def _write_preset_recipe(
    tmp_path: Path,
    *,
    config: str | Path,
    index: str | Path,
    variant: str = "sleep2vec",
    preset: dict | None = None,
    execution: dict | None = None,
    name: str = "unit_preset",
) -> Path:
    preset = preset or {"n_tokens": 128, "split": ["train"], "allow_missing_channels": False}
    config_payload = yaml.safe_load(Path(config).read_text())
    preset_build = config_payload.get("preset_build") or {}
    required_channels = preset.get("channels")
    required_channels_source = "explicit_recipe"
    if required_channels is None and preset_build.get("required_channels") is not None:
        required_channels = preset_build["required_channels"]
        required_channels_source = "explicit_config"
    min_channels = preset.get("min_channels")
    min_channels_source = "explicit_recipe"
    if min_channels is None and preset_build.get("min_channels") is not None:
        min_channels = preset_build["min_channels"]
        min_channels_source = "explicit_config"
    payload = {
        "name": name,
        "task": "preset_prepare",
        "variant": variant,
        "inputs": {"config": str(config), "index": [str(index)], "dataset_name": "unit"},
        "preset": preset,
        "decisions": {
            "task": {"value": "preset_prepare", "source": "explicit_recipe"},
            "preset_regeneration": {"value": True, "source": "explicit_recipe"},
            "overwrite_policy": {"value": bool(preset.get("overwrite", False)), "source": "explicit_recipe"},
            "required_channels": {
                "value": required_channels if required_channels is not None else ["ppg"],
                "source": required_channels_source,
            },
            "min_channels": {"value": min_channels if min_channels is not None else 1, "source": min_channels_source},
        },
    }
    if execution is not None:
        payload["execution"] = execution
    return write_yaml(tmp_path / f"{name}.yaml", payload)


def test_plan_does_not_create_run_all_when_consultation_required(tmp_path: Path):
    recipe = write_finetune_recipe(tmp_path, include_label=False)
    output_dir = tmp_path / "plan"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 2
    assert "Questions for user" in result.stdout
    assert "label_name" in result.stdout
    assert str(output_dir / "decisions.yaml") in result.stdout
    assert (output_dir / "plan.blocked.md").exists()
    assert (output_dir / "decisions.yaml").exists()
    assert "fresh `--output-dir`" in (output_dir / "plan.blocked.md").read_text()
    assert not (output_dir / "run_all.sh").exists()


def test_plan_with_unresolved_experiment_metadata_does_not_write_output(tmp_path: Path):
    recipe = write_finetune_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    payload["step"]["purpose"] = "ASK_USER"
    recipe.write_text(yaml.safe_dump(payload))
    output_dir = tmp_path / "unresolved-plan"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 2
    assert not output_dir.exists()


@pytest.mark.parametrize("section", ["experiment", "step"])
def test_plan_rejects_non_string_workspace_ids_before_creating_workspace(tmp_path: Path, section: str):
    source = tmp_path / "source"
    recipe = write_finetune_recipe(source)
    payload = yaml.safe_load(recipe.read_text())
    workspace = tmp_path / "workspace"
    payload["experiment"]["root"] = str(workspace)
    payload[section]["id"] = 123
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(workspace / "plans" / "first"))

    assert result.returncode == 1
    assert f"{section}.id must be a string" in result.stdout
    assert not workspace.exists()


def test_blocked_plan_initializes_workspace_and_retry_uses_new_plan_dir(tmp_path: Path):
    source = tmp_path / "source"
    recipe = write_finetune_recipe(source, include_label=False)
    workspace = tmp_path / "workspace"
    payload = yaml.safe_load(recipe.read_text())
    payload["experiment"]["root"] = str(workspace)
    recipe.write_text(yaml.safe_dump(payload))
    blocked_dir = workspace / "plans" / "blocked"

    blocked = _run("plan", "--recipe", str(recipe), "--output-dir", str(blocked_dir))

    assert blocked.returncode == 2
    assert (workspace / "experiment.yaml").exists()
    assert (blocked_dir / "plan.blocked.md").exists()
    decisions = blocked_dir / "decisions.yaml"
    decision_payload = yaml.safe_load(decisions.read_text())
    decision_payload["decisions"]["label_name"]["value"] = "ahi"
    decisions.write_text(yaml.safe_dump(decision_payload, sort_keys=False))
    retry_dir = workspace / "plans" / "retry"

    retry = _run(
        "plan",
        "--recipe",
        str(recipe),
        "--user-decisions",
        str(decisions),
        "--output-dir",
        str(retry_dir),
    )

    assert retry.returncode == 0, retry.stderr
    assert (retry_dir / "run.sh").exists()


def test_generic_blocked_plan_retry_rejects_same_output_dir(tmp_path: Path):
    source = tmp_path / "source"
    recipe = write_finetune_recipe(source, include_label=False)
    workspace = tmp_path / "workspace"
    payload = yaml.safe_load(recipe.read_text())
    payload["experiment"]["root"] = str(workspace)
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    blocked_dir = workspace / "plans" / "blocked"

    blocked = _run("plan", "--recipe", str(recipe), "--output-dir", str(blocked_dir))

    assert blocked.returncode == 2
    decisions = blocked_dir / "decisions.yaml"
    decision_payload = yaml.safe_load(decisions.read_text())
    decision_payload["decisions"]["label_name"]["value"] = "ahi"
    decisions.write_text(yaml.safe_dump(decision_payload, sort_keys=False))
    blocked_files = {path.name: path.read_bytes() for path in blocked_dir.iterdir() if path.is_file()}
    manifest = workspace / "run_manifest.tsv"
    manifest_bytes = manifest.read_bytes()

    retry = _run(
        "plan",
        "--recipe",
        str(recipe),
        "--user-decisions",
        str(decisions),
        "--output-dir",
        str(blocked_dir),
    )

    assert retry.returncode == 1
    assert "fresh --output-dir" in retry.stdout
    assert {path.name: path.read_bytes() for path in blocked_dir.iterdir() if path.is_file()} == blocked_files
    assert manifest.read_bytes() == manifest_bytes
    assert not (blocked_dir / "plan.json").exists()
    assert not (blocked_dir / "runs").exists()


def test_hparam_blocked_plan_writes_user_decision_template(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    payload["decisions"]["overwrite_policy"]["value"] = "ASK_USER"
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    output_dir = tmp_path / "hparam-blocked"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 2
    assert yaml.safe_load((output_dir / "decisions.yaml").read_text())["decisions"]["overwrite_policy"] == {
        "value": "ASK_USER",
        "source": "explicit_user",
        "question": "Is overwriting existing output files allowed for this task?",
    }


def test_hparam_blocked_plan_retry_rejects_same_output_dir_even_with_overwrite(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    payload["decisions"]["overwrite_policy"]["value"] = "ASK_USER"
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    blocked_dir = tmp_path / "hparam-blocked"

    blocked = _run("plan", "--recipe", str(recipe), "--output-dir", str(blocked_dir))

    assert blocked.returncode == 2
    decisions = blocked_dir / "decisions.yaml"
    decision_payload = yaml.safe_load(decisions.read_text())
    decision_payload["decisions"]["overwrite_policy"]["value"] = True
    decisions.write_text(yaml.safe_dump(decision_payload, sort_keys=False))
    blocked_files = {path.name: path.read_bytes() for path in blocked_dir.iterdir() if path.is_file()}
    manifest = tmp_path / "run_manifest.tsv"
    manifest_bytes = manifest.read_bytes()

    retry = _run(
        "plan",
        "--recipe",
        str(recipe),
        "--user-decisions",
        str(decisions),
        "--output-dir",
        str(blocked_dir),
    )

    assert retry.returncode == 1
    assert "fresh --output-dir" in retry.stdout
    assert {path.name: path.read_bytes() for path in blocked_dir.iterdir() if path.is_file()} == blocked_files
    assert manifest.read_bytes() == manifest_bytes
    assert not (blocked_dir / "plan.json").exists()
    assert not (blocked_dir / "runs").exists()


def test_context_blocks_survival_index_keys_missing_from_sidecars(tmp_path: Path):
    _recipe, config = _survival_recipe_with_missing_sidecar_key(tmp_path)
    output_dir = tmp_path / "context"

    result = _run(
        "context",
        "--task",
        "finetune",
        "--variant",
        "sleep2vec",
        "--label-name",
        "incident_cox",
        "--config",
        str(config),
        "--output-dir",
        str(output_dir),
    )

    assert result.returncode in {1, 2}
    assert not (output_dir / "decisions.yaml").exists()
    assert (output_dir / "commands.blocked.sh").exists()
    assert not (output_dir / "commands.sh").exists()
    context = json.loads((output_dir / "context.json").read_text())
    assert any("survival key values missing from sidecars" in issue for issue in context["blocking_issues"])


def test_context_writes_questions_and_blocked_script(tmp_path: Path):
    config = yaml.safe_load(write_finetune_recipe(tmp_path).read_text())["inputs"]["config"]
    output_dir = tmp_path / "context"

    result = _run("context", "--task", "finetune", "--config", config, "--output-dir", str(output_dir))

    assert result.returncode == 2
    assert (output_dir / "questions.md").exists()
    assert (output_dir / "commands.blocked.sh").exists()
    assert not (output_dir / "commands.sh").exists()
    context = json.loads((output_dir / "context.json").read_text())
    assert context["skill"]["name"] == "finetuning"
    assert "runtime-orchestrator" in context["owners"]
    assert context["relevant_docs"] == ["doc/codex_index/WORKFLOWS.md"]
    assert context["index_summary"]["rows"] == 1
    assert context["preset_summary"] is None
    assert context["expected_artifacts"]


def test_context_writes_questions_for_mixed_fail_and_user_input(tmp_path: Path):
    recipe = write_finetune_recipe(tmp_path, include_label=False)
    config = Path(yaml.safe_load(recipe.read_text())["inputs"]["config"])
    payload = yaml.safe_load(config.read_text())
    payload["data"]["finetune_data_index"] = None
    payload["data"]["finetune_preset_path"] = str(tmp_path / "missing preset.pkl")
    write_yaml(config, payload)
    output_dir = tmp_path / "context"

    result = _run(
        "context",
        "--task",
        "finetune",
        "--variant",
        "sleep2vec",
        "--config",
        str(config),
        "--output-dir",
        str(output_dir),
    )

    assert result.returncode == 1
    context = json.loads((output_dir / "context.json").read_text())
    assert context["consultation_required"] is True
    assert (output_dir / "questions.md").exists()
    assert (output_dir / "commands.blocked.sh").exists()
    assert not (output_dir / "commands.sh").exists()


def test_plan_blocks_user_decision_test_after_fit_when_finetune_lock_stays_resolved(tmp_path: Path):
    recipe = write_finetune_recipe(tmp_path)
    decisions = write_yaml(
        tmp_path / "decisions.yaml",
        {"decisions": {"test_after_fit": {"value": True, "source": "explicit_user"}}},
    )
    output_dir = tmp_path / "plan"

    result = _run("plan", "--recipe", str(recipe), "--user-decisions", str(decisions), "--output-dir", str(output_dir))

    assert result.returncode == 2
    assert "test_after_fit=true would evaluate test" in result.stdout
    assert not (output_dir / "run.sh").exists()


def test_context_without_workspace_writes_blocked_script(tmp_path: Path):
    output_dir = tmp_path / "context"

    result = _run("context", "--task", "pretrain", "--variant", "sleep2vec", "--output-dir", str(output_dir))

    assert result.returncode == 1
    assert "Unsupported task: pretrain" in json.loads((output_dir / "context.json").read_text())["blocking_issues"]
    assert (output_dir / "commands.blocked.sh").exists()
    assert not (output_dir / "commands.sh").exists()
    assert not (output_dir / "validation.sh").exists()


def test_importing_decisions_does_not_import_torch_or_lightning(monkeypatch):
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.delitem(sys.modules, "pytorch_lightning", raising=False)

    import agent_tools.decisions  # noqa: F401

    assert "torch" not in sys.modules
    assert "pytorch_lightning" not in sys.modules
