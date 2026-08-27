from __future__ import annotations

from pathlib import Path
from shlex import quote as shlex_quote
import sys

from agent_tool_test_helpers import survival_config_payload, write_finetune_recipe, write_survival_sidecars, write_yaml
from test_agent_plan_blocks_on_ambiguity import (
    _RUNTIME_COMMIT,
    _run,
    _survival_recipe_with_missing_sidecar_key,
    _write_infer_recipe,
    _write_preset_recipe,
    _write_survival_config_with_bad_sidecars,
)
from test_agent_plan_blocks_on_ambiguity import _stub_execution_target  # noqa: F401
import yaml


def test_doctor_blocks_survival_index_keys_missing_from_sidecars(tmp_path: Path):
    recipe, _config = _survival_recipe_with_missing_sidecar_key(tmp_path)

    result = _run("doctor", "--recipe", str(recipe), "--output-dir", str(tmp_path / "doctor"))

    assert result.returncode == 1
    assert "Status: FAIL" in result.stdout
    assert "survival key values missing from sidecars" in result.stdout


def test_plan_blocks_survival_index_keys_missing_from_sidecars(tmp_path: Path):
    recipe, _config = _survival_recipe_with_missing_sidecar_key(tmp_path)
    output_dir = tmp_path / "plan"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 1
    assert "survival key values missing from sidecars" in result.stdout
    assert (output_dir / "plan.blocked.md").exists()
    assert not (output_dir / "run.sh").exists()


def test_plan_skips_survival_index_gate_when_finetune_preset_is_configured(tmp_path: Path):
    recipe, config = _survival_recipe_with_missing_sidecar_key(tmp_path)
    preset = tmp_path / "preset.pkl"
    preset.write_bytes(b"preset")
    payload = yaml.safe_load(config.read_text())
    payload["data"]["finetune_preset_path"] = str(preset)
    write_yaml(config, payload)
    output_dir = tmp_path / "plan"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 0
    assert "survival key values missing from sidecars" not in result.stdout
    assert (output_dir / "run.sh").exists()


def test_plan_skips_missing_index_path_when_finetune_preset_is_configured(tmp_path: Path):
    recipe, config = _survival_recipe_with_missing_sidecar_key(tmp_path)
    preset = tmp_path / "preset.pkl"
    preset.write_bytes(b"preset")
    payload = yaml.safe_load(config.read_text())
    payload["data"]["finetune_data_index"] = str(tmp_path / "missing_index.csv")
    payload["data"]["finetune_preset_path"] = str(preset)
    write_yaml(config, payload)
    output_dir = tmp_path / "plan"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 0
    assert "finetune_data_index" not in result.stdout
    assert (output_dir / "run.sh").exists()


def test_plan_blocks_missing_finetune_preset_path(tmp_path: Path):
    recipe, config = _survival_recipe_with_missing_sidecar_key(tmp_path)
    payload = yaml.safe_load(config.read_text())
    payload["data"]["finetune_preset_path"] = str(tmp_path / "missing_preset.pkl")
    write_yaml(config, payload)
    output_dir = tmp_path / "plan"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 1
    assert "finetune_preset_path" in result.stdout
    assert "missing_preset.pkl" in result.stdout
    assert not (output_dir / "run.sh").exists()


def test_hparam_plan_skips_survival_index_gate_when_base_preset_is_configured(tmp_path: Path):
    base, config = _survival_recipe_with_missing_sidecar_key(tmp_path)
    preset = tmp_path / "preset.pkl"
    preset.write_bytes(b"preset")
    payload = yaml.safe_load(config.read_text())
    payload["data"]["finetune_preset_path"] = str(preset)
    write_yaml(config, payload)
    recipe = write_yaml(
        tmp_path / "tune_survival.yaml",
        {
            "name": "unit_tune_survival",
            "task": "hparam_tune",
            "variant": "sleep2vec",
            "base_recipe": str(base),
            "search": {"method": "grid", "max_runs": 1, "parameters": {"runtime.lr": [1e-6]}},
            "evaluation_policy": {
                "selection_metric": "val_loss",
                "selection_mode": "min",
                "selection_split": "val",
                "final_eval_split": "test",
                "external_test_locked": True,
                "test_after_fit": False,
                "final_test_unlocked": False,
                "require_manual_unlock_for_final_test": True,
            },
            "decisions": {
                "task": {"value": "hparam_tune", "source": "explicit_recipe"},
                "label_name": {"value": "incident_cox", "source": "explicit_recipe"},
                "external_test_locked": {"value": True, "source": "explicit_recipe"},
                "train_val_test_policy": {"value": "select on val", "source": "explicit_recipe"},
                "overwrite_policy": {"value": False, "source": "explicit_recipe"},
                "final_eval_unlock": {"value": False, "source": "explicit_recipe"},
            },
        },
    )
    output_dir = tmp_path / "plan"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 0
    assert "survival key values missing from sidecars" not in result.stdout
    assert (output_dir / "run_all.sh").exists()


def test_hparam_plan_skips_remote_deferred_survival_index_summary(tmp_path: Path):
    index = tmp_path / "local_index.csv"
    index.write_text("path,split,duration,eid,ppg_mask\nx.npz,train,60,001,1\n")
    payload = survival_config_payload(
        index,
        {
            "disease_columns_index": "/wujidata/survival/disease_columns.txt",
            "event_time_index": "/wujidata/survival/event_time.csv",
            "is_event_index": "/wujidata/survival/is_event.csv",
            "has_label_index": "/wujidata/survival/has_label.csv",
        },
    )
    payload["data"]["finetune_data_index"] = "/wujidata/survival/index.csv"
    config = write_yaml(tmp_path / "remote_survival_config.yaml", payload)
    base = _survival_recipe_with_missing_sidecar_key(tmp_path)[0]
    base_payload = yaml.safe_load(base.read_text())
    base_payload["inputs"]["config"] = str(config)
    write_yaml(base, base_payload)
    recipe = write_yaml(
        tmp_path / "tune_remote_survival.yaml",
        {
            "name": "unit_tune_remote_survival",
            "task": "hparam_tune",
            "variant": "sleep2vec",
            "base_recipe": str(base),
            "search": {"method": "grid", "max_runs": 1, "parameters": {"runtime.lr": [1e-6]}},
            "evaluation_policy": {
                "selection_metric": "val_loss",
                "selection_mode": "min",
                "selection_split": "val",
                "final_eval_split": "test",
                "external_test_locked": True,
                "test_after_fit": False,
                "final_test_unlocked": False,
                "require_manual_unlock_for_final_test": True,
            },
            "execution": {
                "target": "ssh",
                "host": "baichuan3",
                "path_context": "remote",
                "path_validation": "defer",
                "python": sys.executable,
                "runtime_commit": _RUNTIME_COMMIT,
            },
            "decisions": {
                "task": {"value": "hparam_tune", "source": "explicit_recipe"},
                "label_name": {"value": "incident_cox", "source": "explicit_recipe"},
                "external_test_locked": {"value": True, "source": "explicit_recipe"},
                "train_val_test_policy": {"value": "select on val", "source": "explicit_recipe"},
                "overwrite_policy": {"value": False, "source": "explicit_recipe"},
                "final_eval_unlock": {"value": False, "source": "explicit_recipe"},
            },
        },
    )
    output_dir = tmp_path / "plan"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 0
    assert "Index CSV not found" not in result.stdout
    assert (output_dir / "run_all.sh").exists()


def test_infer_invalid_eval_split_blocks_command_generation(tmp_path: Path):
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
                "eval_split": "validation",
            },
            "evaluation_policy": {"external_test_locked": False, "final_test_unlocked": False},
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

    assert result.returncode == 1
    assert "eval_split must be one of" in result.stdout
    assert not (output_dir / "run.sh").exists()


def test_infer_blocks_missing_inference_preset_path(tmp_path: Path):
    config = _write_survival_config_with_bad_sidecars(tmp_path)
    recipe = _write_infer_recipe(tmp_path, config, inference_preset_path=tmp_path / "missing_override.pkl")
    output_dir = tmp_path / "plan"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 1
    assert "inference_preset_path" in result.stdout
    assert "missing_override.pkl" in result.stdout
    assert not (output_dir / "run.sh").exists()


def test_infer_blocks_missing_config_finetune_preset_path(tmp_path: Path):
    config = _write_survival_config_with_bad_sidecars(tmp_path, preset_path=tmp_path / "missing_config_preset.pkl")
    recipe = _write_infer_recipe(tmp_path, config)
    output_dir = tmp_path / "plan"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 1
    assert "finetune_preset_path" in result.stdout
    assert "missing_config_preset.pkl" in result.stdout
    assert not (output_dir / "run.sh").exists()


def test_infer_survival_blocks_invalid_sidecars_without_preset(tmp_path: Path):
    config = _write_survival_config_with_bad_sidecars(tmp_path)
    recipe = _write_infer_recipe(tmp_path, config)
    output_dir = tmp_path / "plan"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 2
    assert "survival_sidecars" in result.stdout
    assert not (output_dir / "run.sh").exists()


def test_infer_survival_allows_invalid_sidecars_with_preset(tmp_path: Path):
    preset = tmp_path / "preset.pkl"
    preset.write_bytes(b"preset")
    config = _write_survival_config_with_bad_sidecars(tmp_path, preset_path=preset)
    recipe = _write_infer_recipe(tmp_path, config)
    output_dir = tmp_path / "plan"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 0
    assert "survival_sidecars" not in result.stdout
    assert (output_dir / "run.sh").exists()


def test_infer_preset_path_does_not_skip_survival_sidecar_checks(tmp_path: Path):
    preset = tmp_path / "preset.pkl"
    preset.write_bytes(b"preset")
    config = _write_survival_config_with_bad_sidecars(tmp_path)
    recipe = _write_infer_recipe(tmp_path, config)
    payload = yaml.safe_load(recipe.read_text())
    payload["inputs"]["preset_path"] = str(preset)
    write_yaml(recipe, payload)
    output_dir = tmp_path / "plan"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 1
    assert "inputs.preset_path" in result.stdout
    assert not (output_dir / "run.sh").exists()


def test_infer_checks_survival_sidecar_keys_only_for_eval_split(tmp_path: Path):
    index = tmp_path / "survival_infer_index.csv"
    index.write_text("path,split,duration,eid,ppg_mask\n" "val.npz,val,60,001,1\n" "test.npz,test,60,003,1\n")
    config = write_yaml(
        tmp_path / "survival_infer_config.yaml",
        survival_config_payload(index, write_survival_sidecars(tmp_path)),
    )
    recipe = _write_infer_recipe(tmp_path, config, eval_split="val")
    output_dir = tmp_path / "plan_infer_val"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 0
    assert "survival key values missing from sidecars" not in result.stdout
    assert (output_dir / "run.sh").exists()


def test_preset_plan_checks_inputs_index_even_when_config_has_finetune_preset(tmp_path: Path):
    base = write_finetune_recipe(tmp_path)
    config = Path(yaml.safe_load(base.read_text())["inputs"]["config"])
    payload = yaml.safe_load(config.read_text())
    preset = tmp_path / "existing_preset.pkl"
    preset.write_bytes(b"preset")
    payload["data"]["finetune_preset_path"] = str(preset)
    write_yaml(config, payload)
    bad_index = tmp_path / "bad_preset_index.csv"
    bad_index.write_text("eid\n001\n")
    recipe = write_yaml(
        tmp_path / "preset_bad_index.yaml",
        {
            "name": "unit_preset_bad_index",
            "task": "preset_prepare",
            "variant": "sleep2vec",
            "inputs": {"config": str(config), "index": [str(bad_index)], "dataset_name": "unit"},
            "preset": {"n_tokens": 128, "split": ["train"]},
            "decisions": {
                "task": {"value": "preset_prepare", "source": "explicit_recipe"},
                "preset_regeneration": {"value": True, "source": "explicit_recipe"},
                "overwrite_policy": {"value": False, "source": "explicit_recipe"},
                "required_channels": {
                    "value": ["ppg", "ahi", "stage5"],
                    "source": "explicit_config",
                },
                "min_channels": {"value": 3, "source": "explicit_config"},
            },
        },
    )
    output_dir = tmp_path / "plan_bad_index"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 1
    assert "Index CSV missing required column: path" in result.stdout
    assert not (output_dir / "run.sh").exists()


def test_preset_plan_blocks_survival_config_with_invalid_sidecars(tmp_path: Path):
    config = _write_survival_config_with_bad_sidecars(tmp_path)
    index = tmp_path / "preset_index.csv"
    index.write_text("path,split,duration,eid,ppg_mask\nx.npz,train,60,001,1\n")
    recipe = write_yaml(
        tmp_path / "preset_survival_bad_sidecars.yaml",
        {
            "name": "unit_preset_survival_bad_sidecars",
            "task": "preset_prepare",
            "variant": "sleep2vec",
            "inputs": {"config": str(config), "index": [str(index)], "dataset_name": "unit"},
            "preset": {"n_tokens": 128, "split": ["train"], "allow_missing_channels": False},
            "decisions": {
                "task": {"value": "preset_prepare", "source": "explicit_recipe"},
                "preset_regeneration": {"value": True, "source": "explicit_recipe"},
                "overwrite_policy": {"value": False, "source": "explicit_recipe"},
                "required_channels": {"value": ["ppg"], "source": "explicit_recipe"},
                "min_channels": {"value": 1, "source": "explicit_recipe"},
            },
        },
    )
    output_dir = tmp_path / "plan_bad_sidecars"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 2
    assert "survival_sidecars" in result.stdout
    assert not (output_dir / "run.sh").exists()


def test_preset_plan_blocks_multilabel_sidecars_missing_from_execution_workdir(tmp_path: Path):
    base = write_finetune_recipe(tmp_path / "source")
    config = Path(yaml.safe_load(base.read_text())["inputs"]["config"])
    config_payload = yaml.safe_load(config.read_text())
    config_payload["finetune"]["task"].update({"type": "multilabel_classification", "output_dim": 2, "is_seq": False})
    config_payload["finetune"]["multilabel"] = {
        "key_column": "eid",
        "disease_columns_index": "AGENTS.md",
        "label_index": "AGENTS.md",
        "has_label_index": "AGENTS.md",
    }
    config_payload["preset_build"] = {"required_channels": ["ppg"], "min_channels": 1}
    Path(config_payload["data"]["finetune_data_index"]).write_text(
        "path,split,duration,eid,ppg_mask\nx.npz,train,60,001,1\n"
    )
    config.write_text(yaml.safe_dump(config_payload, sort_keys=False))
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    recipe = _write_preset_recipe(
        tmp_path,
        config=config,
        index=config_payload["data"]["finetune_data_index"],
        execution={"workdir": str(runtime)},
        name="preset_multilabel_relative_sidecars",
    )
    output_dir = tmp_path / "plan_multilabel_sidecars"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 2
    assert "multilabel_sidecars" in result.stdout
    assert "AGENTS.md" in result.stdout
    assert not (output_dir / "run.sh").exists()


def test_preset_plan_checks_survival_sidecar_keys_only_for_requested_split(tmp_path: Path):
    index = tmp_path / "preset_survival_index.csv"
    index.write_text("path,split,duration,eid,ppg_mask\n" "train.npz,train,60,001,1\n" "test.npz,test,60,003,1\n")
    config = write_yaml(
        tmp_path / "survival_config.yaml",
        survival_config_payload(index, write_survival_sidecars(tmp_path)),
    )
    recipe = write_yaml(
        tmp_path / "preset_survival_train.yaml",
        {
            "name": "unit_preset_survival_train",
            "task": "preset_prepare",
            "variant": "sleep2vec",
            "inputs": {"config": str(config), "index": [str(index)], "dataset_name": "unit"},
            "preset": {"n_tokens": 128, "split": ["train"], "allow_missing_channels": False},
            "decisions": {
                "task": {"value": "preset_prepare", "source": "explicit_recipe"},
                "preset_regeneration": {"value": True, "source": "explicit_recipe"},
                "overwrite_policy": {"value": False, "source": "explicit_recipe"},
                "required_channels": {"value": ["ppg"], "source": "explicit_recipe"},
                "min_channels": {"value": 1, "source": "explicit_recipe"},
            },
        },
    )
    output_dir = tmp_path / "plan_survival_train"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 0
    assert "survival key values missing from sidecars" not in result.stdout
    assert (output_dir / "run.sh").exists()


def test_preset_plan_skips_local_index_summary_for_remote_deferred_index(tmp_path: Path):
    config = yaml.safe_load(write_finetune_recipe(tmp_path).read_text())["inputs"]["config"]
    recipe = write_yaml(
        tmp_path / "preset_remote_index.yaml",
        {
            "name": "unit_preset_remote_index",
            "task": "preset_prepare",
            "variant": "sleep2vec",
            "inputs": {"config": config, "index": ["/wujidata/index.csv"], "dataset_name": "unit"},
            "preset": {"n_tokens": 128, "split": ["train"], "allow_missing_channels": False},
            "execution": {
                "target": "ssh",
                "host": "baichuan3",
                "path_context": "remote",
                "path_validation": "defer",
            },
            "decisions": {
                "task": {"value": "preset_prepare", "source": "explicit_recipe"},
                "preset_regeneration": {"value": True, "source": "explicit_recipe"},
                "overwrite_policy": {"value": False, "source": "explicit_recipe"},
                "required_channels": {
                    "value": ["ppg", "ahi", "stage5"],
                    "source": "explicit_config",
                },
                "min_channels": {"value": 3, "source": "explicit_config"},
            },
        },
    )
    output_dir = tmp_path / "plan_remote_index"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 0
    assert "Index CSV not found" not in result.stdout
    assert (output_dir / "run.sh").exists()


def test_finetune_plan_includes_explicit_input_and_runtime_args(tmp_path: Path):
    recipe = write_finetune_recipe(tmp_path)
    pretrained = tmp_path / "pretrained model.ckpt"
    resume = tmp_path / "resume checkpoint.ckpt"
    pretrained.write_text("checkpoint")
    resume.write_text("checkpoint")
    payload = yaml.safe_load(recipe.read_text())
    payload["inputs"]["pretrained_backbone_path"] = str(pretrained)
    payload["inputs"]["ckpt_path"] = str(resume)
    payload["runtime"].update(
        {
            "device": "cuda",
            "warmup_steps": 11,
            "gradient_clip_val": 0.5,
            "accumulate_grad_batches": 2,
            "patience": 4,
            "check_val_every_n_epoch": 2,
            "ckpt_every_n_epochs": 3,
        }
    )
    payload["decisions"]["pretrained_backbone_path"] = {
        "value": str(pretrained),
        "source": "explicit_recipe",
    }
    write_yaml(recipe, payload)
    output_dir = tmp_path / "plan"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 0
    script = (output_dir / "run.sh").read_text()
    assert f"--pretrained-backbone-path {shlex_quote(str(pretrained))}" in script
    assert f"--ckpt-path {shlex_quote(str(resume))}" in script
    assert "--device cuda" in script
    assert "--warmup-steps 11" in script
    assert "--gradient-clip-val 0.5" in script
    assert "--accumulate-grad-batches 2" in script
    assert "--patience 4" in script
    assert "--check-val-every-n-epoch 2" in script
    assert "--ckpt-every-n-epochs 3" in script


def test_finetune_blocks_missing_pretrained_backbone_path(tmp_path: Path):
    recipe = write_finetune_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    missing = tmp_path / "missing_pretrained.ckpt"
    payload["inputs"]["pretrained_backbone_path"] = str(missing)
    payload["decisions"]["pretrained_backbone_path"] = {
        "value": str(missing),
        "source": "explicit_recipe",
    }
    write_yaml(recipe, payload)
    output_dir = tmp_path / "plan"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 1
    assert "pretrained_backbone_path" in result.stdout
    assert "missing_pretrained.ckpt" in result.stdout
    assert not (output_dir / "run.sh").exists()


def test_finetune_blocks_missing_resume_ckpt_path(tmp_path: Path):
    recipe = write_finetune_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    payload["inputs"]["ckpt_path"] = str(tmp_path / "missing_resume.ckpt")
    write_yaml(recipe, payload)
    output_dir = tmp_path / "plan"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 1
    assert "ckpt_path" in result.stdout
    assert "missing_resume.ckpt" in result.stdout
    assert not (output_dir / "run.sh").exists()
