from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from shlex import quote as shlex_quote
import subprocess
import sys

from agent_tool_test_helpers import write_finetune_recipe, write_yaml
from test_agent_plan_blocks_on_ambiguity import (
    _RUNTIME_COMMIT,
    _first_run,
    _hparam_recipe,
    _local_runtime_execution,
    _run,
    _survival_recipe_with_missing_sidecar_key,
    _valid_final_config_bytes,
)
from test_agent_plan_blocks_on_ambiguity import _stub_execution_target  # noqa: F401
import yaml

from agent_tools import configs, plan_context, plan_hparam, plans
from agent_tools.adapters.hparam_tune import HparamTuneAdapter
from agent_tools.models import REPO_ROOT
from agent_tools.plan_hparam import final_test_checkpoint_issues


def test_infer_eval_split_ask_user_blocks_command_generation(tmp_path: Path):
    ckpt = tmp_path / "model.ckpt"
    ckpt.write_text("checkpoint")
    config = yaml.safe_load(write_finetune_recipe(tmp_path).read_text())["inputs"]["config"]
    recipe = write_yaml(
        tmp_path / "infer.yaml",
        {
            "name": "unit_infer",
            "task": "infer",
            "variant": "sleep2vec",
            "inputs": {
                "config": config,
                "label_name": "ahi",
                "ckpt_path": str(ckpt),
                "eval_split": "ASK_USER",
            },
            "evaluation_policy": {"external_test_locked": True, "final_test_unlocked": False},
            "decisions": {
                "task": {"value": "infer", "source": "explicit_recipe"},
                "label_name": {"value": "ahi", "source": "explicit_recipe"},
                "ckpt_path": {"value": str(ckpt), "source": "explicit_recipe"},
                "overwrite_policy": {"value": False, "source": "explicit_recipe"},
            },
        },
    )
    output_dir = tmp_path / "plan"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 2
    assert "eval_split" in result.stdout
    assert not (output_dir / "run.sh").exists()


def test_unlock_final_test_with_yaml_search_requires_explicit_final_config(tmp_path: Path):
    ckpt = tmp_path / "best.ckpt"
    ckpt.write_text("checkpoint")
    recipe = _hparam_recipe(tmp_path, parameters={"yaml:/finetune/task/output_dim": [31]}, ckpt_path=ckpt)
    output_dir = tmp_path / "unlocked"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir), "--unlock-final-test")

    assert result.returncode == 2
    assert not (output_dir / "final_external_test.sh").exists()
    assert "final_eval_config_path" in (output_dir / "questions.md").read_text()


def test_unlock_final_test_with_yaml_search_uses_explicit_final_config(tmp_path: Path):
    ckpt = tmp_path / "best.ckpt"
    ckpt.write_text("checkpoint")
    selected_config = tmp_path / "selected_run.yaml"
    recipe = _hparam_recipe(
        tmp_path,
        parameters={"yaml:/finetune/task/output_dim": [31]},
        ckpt_path=ckpt,
        final_config_path=selected_config,
    )
    selected_bytes = _valid_final_config_bytes(tmp_path)
    selected_config.write_bytes(selected_bytes)
    output_dir = tmp_path / "unlocked"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir), "--unlock-final-test")

    assert result.returncode == 0
    frozen_config = output_dir / "config.final_eval.yaml"
    script = (output_dir / "final_external_test.sh").read_text()
    assert frozen_config.read_bytes() == selected_bytes
    assert shlex_quote(str(frozen_config)) in script
    assert shlex_quote(str(selected_config)) not in script
    assert "runs/run-000" not in script
    plan = json.loads((output_dir / "plan.json").read_text())
    assert plan["final_eval_config"] == {
        "path": str(frozen_config),
        "sha256": hashlib.sha256(selected_bytes).hexdigest(),
        "source_path": str(selected_config),
    }
    assert "_final_eval_config_snapshot" not in plan["recipe"]
    assert "_final_eval_config_snapshot" not in (output_dir / "recipe.resolved.yaml").read_text()


def test_relative_final_eval_config_remains_repo_relative_with_execution_workdir(tmp_path: Path):
    ckpt = tmp_path / "best.ckpt"
    ckpt.write_text("checkpoint")
    selected_config = tmp_path / "selected_run.yaml"
    recipe = _hparam_recipe(
        tmp_path,
        parameters={"yaml:/finetune/task/output_dim": [31]},
        ckpt_path=ckpt,
        final_config_path=selected_config,
    )
    selected_bytes = _valid_final_config_bytes(tmp_path)
    selected_config.write_bytes(selected_bytes)
    payload = yaml.safe_load(recipe.read_text())
    payload["inputs"]["final_eval_config_path"] = os.path.relpath(selected_config, REPO_ROOT)
    payload["execution"] = _local_runtime_execution(tmp_path / "runtime")
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    output_dir = tmp_path / "unlocked"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir), "--unlock-final-test")

    assert result.returncode == 0, result.stderr or result.stdout
    assert (output_dir / "config.final_eval.yaml").read_bytes() == selected_bytes


def test_hparam_final_inference_resolves_preset_from_execution_workdir(tmp_path: Path):
    ckpt = tmp_path / "best.ckpt"
    ckpt.write_text("checkpoint")
    recipe = _hparam_recipe(tmp_path, ckpt_path=ckpt)
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    payload = yaml.safe_load(recipe.read_text())
    payload["inputs"]["inference_preset_path"] = "AGENTS.md"
    payload["execution"] = _local_runtime_execution(runtime)
    write_yaml(recipe, payload)
    output_dir = tmp_path / "unlocked_relative_preset"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir), "--unlock-final-test")

    assert result.returncode == 1
    assert "inference_preset_path" in result.stdout
    assert "AGENTS.md" in result.stdout
    assert not (output_dir / "final_external_test.sh").exists()


def test_hparam_final_inference_rejects_npz_preset_for_kaldi_backend(tmp_path: Path):
    ckpt = tmp_path / "best.ckpt"
    ckpt.write_text("checkpoint")
    recipe = _hparam_recipe(tmp_path, ckpt_path=ckpt)
    payload = yaml.safe_load(recipe.read_text())
    config_path = Path(payload["base_recipe"]).parent / "config.yaml"
    config = yaml.safe_load(config_path.read_text())
    kaldi_root = tmp_path / "kaldi"
    kaldi_root.mkdir()
    manifest = kaldi_root / "manifest.json"
    manifest.write_text("{}\n")
    config["data"].update(
        {
            "backend": "kaldi",
            "finetune_data_index": None,
            "finetune_preset_path": None,
            "kaldi_data_root": str(kaldi_root),
            "kaldi_manifest": str(manifest),
        }
    )
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "external.pickle").write_bytes(b"preset")
    payload["inputs"]["inference_preset_path"] = "external.pickle"
    payload["execution"] = _local_runtime_execution(runtime)
    write_yaml(recipe, payload)
    output_dir = tmp_path / "unlocked_kaldi_preset"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir), "--unlock-final-test")

    assert result.returncode == 1
    assert "inference_preset_path" in result.stdout
    assert "Kaldi backend does not support" in result.stdout
    assert not (output_dir / "final_external_test.sh").exists()


def test_hparam_final_inference_rejects_ahi_checkpoint_averaging(tmp_path: Path):
    ckpt = tmp_path / "best.ckpt"
    ckpt.write_text("checkpoint")
    averages = tmp_path / "averages"
    averages.mkdir()
    recipe = _hparam_recipe(tmp_path, ckpt_path=ckpt)
    payload = yaml.safe_load(recipe.read_text())
    payload["runtime"] = {"avg_ckpts": 2, "avg_ckpt_dir": str(averages)}
    write_yaml(recipe, payload)
    output_dir = tmp_path / "unlocked_ahi_averages"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir), "--unlock-final-test")

    assert result.returncode == 1
    assert "runtime.avg_ckpts" in result.stdout
    assert "AHI inference does not support checkpoint averaging" in result.stdout
    assert not (output_dir / "plan.json").exists()
    assert not (output_dir / "final_external_test.sh").exists()


def test_hparam_final_checkpoint_must_be_a_file(tmp_path: Path):
    checkpoint_dir = tmp_path / "checkpoint-directory"
    checkpoint_dir.mkdir()

    issues = final_test_checkpoint_issues(
        {"variant": "sleep2vec", "inputs": {"ckpt_path": str(checkpoint_dir)}},
        None,
        unlock_final_test=True,
    )

    assert [(issue.field, issue.status.value) for issue in issues] == [("ckpt_path", "FAIL")]


def test_hparam_final_config_resolves_runtime_inputs_from_execution_workdir(tmp_path: Path):
    ckpt = tmp_path / "best.ckpt"
    ckpt.write_text("checkpoint")
    selected_config = tmp_path / "selected_run.yaml"
    recipe = _hparam_recipe(tmp_path, ckpt_path=ckpt, final_config_path=selected_config)
    selected_payload = yaml.safe_load(_valid_final_config_bytes(tmp_path))
    selected_payload["data"]["finetune_data_index"] = "AGENTS.md"
    selected_config.write_text(yaml.safe_dump(selected_payload, sort_keys=False))
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    payload = yaml.safe_load(recipe.read_text())
    payload["execution"] = _local_runtime_execution(runtime)
    write_yaml(recipe, payload)
    output_dir = tmp_path / "unlocked_relative_final_input"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir), "--unlock-final-test")

    assert result.returncode == 1
    assert "finetune_data_index" in result.stdout
    assert "AGENTS.md" in result.stdout
    assert not (output_dir / "final_external_test.sh").exists()


def test_hparam_final_config_resolves_generic_kaldi_inputs_from_execution_workdir(tmp_path: Path):
    ckpt = tmp_path / "best.ckpt"
    ckpt.write_text("checkpoint")
    selected_config = tmp_path / "selected_kaldi.yaml"
    recipe = _hparam_recipe(tmp_path, ckpt_path=ckpt, final_config_path=selected_config)
    selected_payload = yaml.safe_load(_valid_final_config_bytes(tmp_path))
    selected_payload["data"].update(
        {
            "backend": "kaldi",
            "finetune_data_index": None,
            "kaldi_data_root": "AGENTS.md",
            "kaldi_manifest": "AGENTS.md",
        }
    )
    selected_config.write_text(yaml.safe_dump(selected_payload, sort_keys=False))
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    payload = yaml.safe_load(recipe.read_text())
    payload["execution"] = _local_runtime_execution(runtime)
    write_yaml(recipe, payload)
    output_dir = tmp_path / "unlocked_relative_kaldi"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir), "--unlock-final-test")

    assert result.returncode == 1
    assert "kaldi_data_root" in result.stdout
    assert "kaldi_manifest" in result.stdout
    assert "AGENTS.md" in result.stdout
    assert not (output_dir / "final_external_test.sh").exists()


def test_hparam_final_config_resolves_multilabel_sidecars_from_execution_workdir(tmp_path: Path):
    ckpt = tmp_path / "best.ckpt"
    ckpt.write_text("checkpoint")
    selected_config = tmp_path / "selected_multilabel.yaml"
    recipe = _hparam_recipe(tmp_path, ckpt_path=ckpt, final_config_path=selected_config)
    selected_payload = yaml.safe_load(_valid_final_config_bytes(tmp_path))
    selected_payload["finetune"]["task"].update(
        {
            "type": "multilabel_classification",
            "output_dim": 2,
            "is_seq": False,
            "monitor": "val_loss",
            "monitor_mod": "min",
        }
    )
    selected_payload["finetune"]["multilabel"] = {
        "key_column": "eid",
        "disease_columns_index": "AGENTS.md",
        "label_index": "AGENTS.md",
        "has_label_index": "AGENTS.md",
    }
    selected_config.write_text(yaml.safe_dump(selected_payload, sort_keys=False))
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    payload = yaml.safe_load(recipe.read_text())
    payload["execution"] = _local_runtime_execution(runtime)
    write_yaml(recipe, payload)
    output_dir = tmp_path / "unlocked_relative_multilabel"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir), "--unlock-final-test")

    assert result.returncode == 2
    assert "multilabel_sidecars" in result.stdout
    assert "AGENTS.md" in result.stdout
    assert not (output_dir / "final_external_test.sh").exists()


def test_unlock_final_test_rejects_invalid_explicit_config_before_workspace_mutation(tmp_path: Path):
    ckpt = tmp_path / "best.ckpt"
    ckpt.write_text("checkpoint")
    selected_config = tmp_path / "invalid_selected_run.yaml"
    selected_config.write_text("{}\n")
    recipe = _hparam_recipe(
        tmp_path,
        parameters={"yaml:/finetune/task/output_dim": [31]},
        ckpt_path=ckpt,
        final_config_path=selected_config,
    )
    output_dir = tmp_path / "unlocked"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir), "--unlock-final-test")

    assert result.returncode == 1
    assert "Final evaluation config is invalid for variant=sleep2vec" in result.stdout
    assert "Finetune YAML must include a top-level 'finetune' block" in result.stdout
    assert not output_dir.exists()
    assert not (tmp_path / "steps" / "unit-hparam-tune").exists()


def test_unlock_final_test_rejects_variant_incompatible_explicit_config_before_workspace_mutation(tmp_path: Path):
    ckpt = tmp_path / "best.ckpt"
    ckpt.write_text("checkpoint")
    selected_config = tmp_path / "sex_age_selected_run.yaml"
    selected_config.write_bytes((REPO_ROOT / "configs" / "sex_age_baseline" / "cox.yaml").read_bytes())
    recipe = _hparam_recipe(tmp_path, ckpt_path=ckpt, final_config_path=selected_config)
    output_dir = tmp_path / "unlocked"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir), "--unlock-final-test")

    assert result.returncode == 1
    assert "Final evaluation config is invalid for variant=sleep2vec" in result.stdout
    assert not output_dir.exists()
    assert not (tmp_path / "steps" / "unit-hparam-tune").exists()


def test_unlock_final_test_detects_explicit_config_drift_before_workspace_mutation(tmp_path: Path, monkeypatch):
    from sleep2vec import config as runtime_config

    ckpt = tmp_path / "best.ckpt"
    ckpt.write_text("checkpoint")
    selected_config = tmp_path / "selected_run.yaml"
    recipe = _hparam_recipe(
        tmp_path,
        parameters={"yaml:/finetune/task/output_dim": [31]},
        ckpt_path=ckpt,
        final_config_path=selected_config,
    )
    selected_bytes = _valid_final_config_bytes(tmp_path) + b"# Selected final-evaluation snapshot.\n"
    selected_config.write_bytes(selected_bytes)
    real_load_finetune_config = runtime_config.load_finetune_config
    mutated = []

    def mutate_source_after_validation(path: Path):
        bundle = real_load_finetune_config(path)
        if Path(path).read_bytes() == selected_bytes:
            selected_config.write_text("{}\n")
            mutated.append(selected_bytes)
        return bundle

    monkeypatch.setattr(runtime_config, "load_finetune_config", mutate_source_after_validation)
    output_dir = tmp_path / "unlocked"

    report = plans.build_plan(recipe_path=recipe, output_dir=output_dir, unlock_final_test=True)

    assert report.exit_code == 1
    assert mutated == [selected_bytes]
    assert any("changed while plan preflight" in issue.message for issue in report.issues)
    assert not output_dir.exists()
    assert not (tmp_path / "steps" / "unit-hparam-tune").exists()


def test_unlock_final_test_freezes_explicit_config_without_yaml_search(tmp_path: Path):
    ckpt = tmp_path / "best.ckpt"
    ckpt.write_text("checkpoint")
    selected_config = tmp_path / "selected_run.yaml"
    recipe = _hparam_recipe(tmp_path, ckpt_path=ckpt, final_config_path=selected_config)
    selected_bytes = _valid_final_config_bytes(tmp_path)
    selected_config.write_bytes(selected_bytes)
    output_dir = tmp_path / "unlocked"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir), "--unlock-final-test")

    assert result.returncode == 0, result.stderr or result.stdout
    frozen_config = output_dir / "config.final_eval.yaml"
    assert frozen_config.read_bytes() == selected_bytes
    script = (output_dir / "final_external_test.sh").read_text()
    assert shlex_quote(str(frozen_config)) in script
    assert shlex_quote(str(selected_config)) not in script


def test_unlock_final_test_captures_remote_explicit_config_over_ssh(tmp_path: Path, monkeypatch):
    ckpt = tmp_path / "best.ckpt"
    ckpt.write_text("checkpoint")
    remote_config = "/remote/selected_run.yaml"
    recipe = _hparam_recipe(tmp_path, ckpt_path=ckpt, final_config_path=Path(remote_config))
    selected_bytes = _valid_final_config_bytes(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    payload["execution"] = {
        "target": "ssh",
        "host": "unit-host",
        "path_context": "remote",
        "path_validation": "ssh",
        "python": sys.executable,
        "runtime_commit": _RUNTIME_COMMIT,
    }
    write_yaml(recipe, payload)
    calls = []

    def fake_run_ssh(host: str, command: str, **kwargs):
        calls.append((host, command, kwargs))
        if command.startswith("cat -- "):
            return subprocess.CompletedProcess([], 0, stdout=selected_bytes, stderr=b"")
        return subprocess.CompletedProcess([], 0, stdout="path-present\n", stderr="")

    monkeypatch.setattr("agent_tools.transport.run_ssh", fake_run_ssh)
    output_dir = tmp_path / "unlocked"

    report = plans.build_plan(recipe_path=recipe, output_dir=output_dir, unlock_final_test=True)

    assert report.exit_code == 0
    frozen_config = output_dir / "config.final_eval.yaml"
    assert frozen_config.read_bytes() == selected_bytes
    assert [command for _host, command, _kwargs in calls].count(f"cat -- {remote_config}") == 2
    assert shlex_quote(str(frozen_config)) in (output_dir / "final_external_test.sh").read_text()


def test_unlock_final_test_rejects_deferred_remote_explicit_config_before_workspace_mutation(tmp_path: Path):
    ckpt = tmp_path / "best.ckpt"
    ckpt.write_text("checkpoint")
    recipe = _hparam_recipe(
        tmp_path,
        ckpt_path=ckpt,
        final_config_path=Path("/remote/selected_run.yaml"),
    )
    payload = yaml.safe_load(recipe.read_text())
    payload["execution"] = {
        "target": "ssh",
        "host": "unit-host",
        "path_context": "remote",
        "path_validation": "defer",
        "python": sys.executable,
        "runtime_commit": _RUNTIME_COMMIT,
    }
    write_yaml(recipe, payload)
    output_dir = tmp_path / "unlocked"

    report = plans.build_plan(recipe_path=recipe, output_dir=output_dir, unlock_final_test=True)

    assert report.exit_code == 1
    assert any(
        "remote final_eval_config_path requires execution.path_validation=ssh" in issue.message
        for issue in report.issues
    )
    assert not output_dir.exists()
    assert not (tmp_path / "steps" / "unit-hparam-tune").exists()


def test_infer_user_decision_ckpt_path_must_exist(tmp_path: Path):
    config = yaml.safe_load(write_finetune_recipe(tmp_path).read_text())["inputs"]["config"]
    recipe = write_yaml(
        tmp_path / "infer.yaml",
        {
            "name": "unit_infer",
            "task": "infer",
            "variant": "sleep2vec",
            "inputs": {
                "config": config,
                "label_name": "ahi",
                "ckpt_path": "ASK_USER",
                "eval_split": "validation",
            },
            "evaluation_policy": {"final_test_unlocked": False},
            "artifacts": {"overwrite": True},
            "decisions": {
                "task": {"value": "infer", "source": "explicit_recipe"},
                "label_name": {"value": "ahi", "source": "explicit_recipe"},
                "overwrite_policy": {"value": True, "source": "explicit_recipe"},
            },
        },
    )
    decisions = write_yaml(
        tmp_path / "decisions.yaml",
        {
            "decisions": {"ckpt_path": {"value": str(tmp_path / "missing.ckpt"), "source": "explicit_user"}},
        },
    )
    output_dir = tmp_path / "plan"

    result = _run(
        "plan",
        "--recipe",
        str(recipe),
        "--user-decisions",
        str(decisions),
        "--output-dir",
        str(output_dir),
    )

    assert result.returncode == 1
    assert "ckpt_path" in (output_dir / "questions.md").read_text()
    assert not (output_dir / "run.sh").exists()


def test_finetune_plan_freezes_config_bytes_validated_before_workspace_setup(tmp_path: Path, monkeypatch):
    recipe = write_finetune_recipe(tmp_path)
    config = Path(yaml.safe_load(recipe.read_text())["inputs"]["config"])
    validated_bytes = config.read_bytes()
    real_ensure_workspace = plans.ensure_experiment_workspace

    def mutate_source_after_preflight(recipe_payload: dict, output_dir: Path, **workspace_options):
        payload = yaml.safe_load(config.read_text())
        payload["finetune"]["task"].update({"monitor": "val_loss", "monitor_mod": "min"})
        config.write_text(yaml.safe_dump(payload, sort_keys=False))
        return real_ensure_workspace(recipe_payload, output_dir, **workspace_options)

    monkeypatch.setattr(plans, "ensure_experiment_workspace", mutate_source_after_preflight)
    output_dir = tmp_path / "plan"

    report = plans.build_plan(recipe_path=recipe, output_dir=output_dir)

    assert report.exit_code == 0
    frozen_config = Path(_first_run(output_dir)["config"])
    assert frozen_config.read_bytes() == validated_bytes
    assert yaml.safe_load(frozen_config.read_text())["finetune"]["task"]["monitor"] == "val_ahi_pearson"


def test_finetune_plan_validates_captured_bytes_during_aba_source_swap(tmp_path: Path, monkeypatch):
    recipe = write_finetune_recipe(tmp_path)
    config = Path(yaml.safe_load(recipe.read_text())["inputs"]["config"])
    validated_bytes = config.read_bytes()
    swapped = yaml.safe_load(config.read_text())
    swapped["finetune"]["task"]["output_dim"] = 31
    swapped_bytes = yaml.safe_dump(swapped, sort_keys=False).encode()
    real_summary = configs.finetune_summary_body

    def summarize_while_source_is_swapped(config_path: Path, **kwargs):
        config.write_bytes(swapped_bytes)
        try:
            return real_summary(config_path, **kwargs)
        finally:
            config.write_bytes(validated_bytes)

    monkeypatch.setattr(configs, "finetune_summary_body", summarize_while_source_is_swapped)
    output_dir = tmp_path / "plan"

    report = plans.build_plan(recipe_path=recipe, output_dir=output_dir)

    assert report.exit_code == 0
    frozen_config = Path(_first_run(output_dir)["config"])
    assert frozen_config.read_bytes() == validated_bytes
    assert yaml.safe_load(frozen_config.read_text())["finetune"]["task"]["output_dim"] == 30


def test_hparam_plan_materializes_config_validated_before_workspace_setup(tmp_path: Path, monkeypatch):
    recipe = _hparam_recipe(tmp_path)
    base_recipe = Path(yaml.safe_load(recipe.read_text())["base_recipe"])
    config = Path(yaml.safe_load(base_recipe.read_text())["inputs"]["config"])
    validated_bytes = config.read_bytes()
    real_ensure_workspace = plan_hparam.ensure_experiment_workspace

    def mutate_source_after_preflight(recipe_payload: dict, output_dir: Path, **workspace_options):
        payload = yaml.safe_load(config.read_text())
        payload["finetune"]["task"].update({"monitor": "val_loss", "monitor_mod": "min"})
        config.write_text(yaml.safe_dump(payload, sort_keys=False))
        return real_ensure_workspace(recipe_payload, output_dir, **workspace_options)

    monkeypatch.setattr(plan_hparam, "ensure_experiment_workspace", mutate_source_after_preflight)
    output_dir = tmp_path / "plan"

    report = plans.build_plan(recipe_path=recipe, output_dir=output_dir)

    assert report.exit_code == 0
    assert (output_dir / "config.source.yaml").read_bytes() == validated_bytes
    run_config = Path(_first_run(output_dir)["config"])
    assert yaml.safe_load(run_config.read_text())["finetune"]["task"]["monitor"] == "val_ahi_pearson"


def test_hparam_override_checks_captured_config_during_aba_source_swap(tmp_path: Path, monkeypatch):
    recipe = _hparam_recipe(tmp_path)
    base_recipe = Path(yaml.safe_load(recipe.read_text())["base_recipe"])
    config = Path(yaml.safe_load(base_recipe.read_text())["inputs"]["config"])
    captured_bytes = config.read_bytes()
    swapped = yaml.safe_load(captured_bytes)
    swapped["finetune"]["task"].update({"monitor": "val_loss", "monitor_mod": "min"})
    swapped_bytes = yaml.safe_dump(swapped, sort_keys=False).encode()
    real_override_issues = HparamTuneAdapter.config_override_issues

    def check_overrides_while_source_is_swapped(self, recipe_payload: dict, config_payload: dict | None):
        config.write_bytes(swapped_bytes)
        try:
            return real_override_issues(self, recipe_payload, config_payload)
        finally:
            config.write_bytes(captured_bytes)

    monkeypatch.setattr(HparamTuneAdapter, "config_override_issues", check_overrides_while_source_is_swapped)
    output_dir = tmp_path / "plan"

    report = plans.build_plan(recipe_path=recipe, output_dir=output_dir)

    assert report.exit_code == 0
    assert (output_dir / "config.source.yaml").read_bytes() == captured_bytes


def test_hparam_override_does_not_approve_invalid_captured_config_during_aba_source_swap(tmp_path: Path, monkeypatch):
    recipe = _hparam_recipe(tmp_path)
    base_recipe = Path(yaml.safe_load(recipe.read_text())["base_recipe"])
    config = Path(yaml.safe_load(base_recipe.read_text())["inputs"]["config"])
    valid_bytes = config.read_bytes()
    captured = yaml.safe_load(valid_bytes)
    captured["finetune"]["task"].update({"monitor": "val_loss", "monitor_mod": "min"})
    captured_bytes = yaml.safe_dump(captured, sort_keys=False).encode()
    config.write_bytes(captured_bytes)
    real_override_issues = HparamTuneAdapter.config_override_issues

    def check_overrides_while_source_is_swapped(self, recipe_payload: dict, config_payload: dict | None):
        config.write_bytes(valid_bytes)
        try:
            return real_override_issues(self, recipe_payload, config_payload)
        finally:
            config.write_bytes(captured_bytes)

    monkeypatch.setattr(HparamTuneAdapter, "config_override_issues", check_overrides_while_source_is_swapped)
    output_dir = tmp_path / "plan"

    report = plans.build_plan(recipe_path=recipe, output_dir=output_dir)

    assert report.exit_code == 1
    assert any(issue.field == "selection_metric" for issue in report.issues)
    assert not (output_dir / "plan.json").exists()
    assert config.read_bytes() == captured_bytes


def test_hparam_override_validation_fails_without_bound_config_bytes(tmp_path: Path):
    recipe_path = _hparam_recipe(tmp_path)
    recipe, cfg, report = plans.evaluate_recipe(recipe_path)
    assert report.exit_code == 0
    assert cfg is not None
    cfg.pop("_source_config_bytes")

    issues = HparamTuneAdapter().config_override_issues(recipe, cfg)

    assert issues is not None
    assert len(issues) == 1
    assert issues[0].status.value == "FAIL"
    assert issues[0].field == "config"


def test_survival_index_summary_checks_captured_config_during_aba_source_swap(tmp_path: Path, monkeypatch):
    recipe, config = _survival_recipe_with_missing_sidecar_key(tmp_path)
    config_payload = yaml.safe_load(config.read_text())
    index = Path(config_payload["data"]["finetune_data_index"])
    index.write_text("path,split,duration,eid,ppg_mask\na.npz,train,60,001,1\nb.npz,val,60,002,1\n")
    captured_bytes = config.read_bytes()
    swapped = yaml.safe_load(captured_bytes)
    swapped["finetune"]["survival"]["key_column"] = "subject_id"
    swapped_bytes = yaml.safe_dump(swapped, sort_keys=False).encode()
    real_index_summary = plan_context.index_summary

    def summarize_while_source_is_swapped(*args, **kwargs):
        config.write_bytes(swapped_bytes)
        try:
            return real_index_summary(*args, **kwargs)
        finally:
            config.write_bytes(captured_bytes)

    monkeypatch.setattr(plan_context, "index_summary", summarize_while_source_is_swapped)
    output_dir = tmp_path / "plan"

    report = plans.build_plan(recipe_path=recipe, output_dir=output_dir)

    assert report.exit_code == 0
    assert Path(_first_run(output_dir)["config"]).read_bytes() == captured_bytes


def test_survival_index_summary_does_not_approve_invalid_captured_config_during_aba_source_swap(
    tmp_path: Path, monkeypatch
):
    recipe, config = _survival_recipe_with_missing_sidecar_key(tmp_path)
    captured_bytes = config.read_bytes()
    swapped = yaml.safe_load(captured_bytes)
    swapped["finetune"]["task"]["type"] = "classification"
    swapped["finetune"].pop("survival")
    valid_bytes = yaml.safe_dump(swapped, sort_keys=False).encode()
    real_index_summary = plan_context.index_summary

    def summarize_while_source_is_swapped(*args, **kwargs):
        config.write_bytes(valid_bytes)
        try:
            return real_index_summary(*args, **kwargs)
        finally:
            config.write_bytes(captured_bytes)

    monkeypatch.setattr(plan_context, "index_summary", summarize_while_source_is_swapped)
    output_dir = tmp_path / "plan"

    report = plans.build_plan(recipe_path=recipe, output_dir=output_dir)

    assert report.exit_code == 1
    assert any("survival key values missing from sidecars" in issue.message for issue in report.issues)
    assert not (output_dir / "run.sh").exists()
    assert config.read_bytes() == captured_bytes
