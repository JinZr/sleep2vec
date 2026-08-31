from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import yaml


def run_execution_preflight_fixture(execution: dict, command: list[str]) -> subprocess.CompletedProcess:
    from agent_tools import python_programs

    python_command, flag, script, *arguments = command
    if flag != "-c":
        raise AssertionError(f"Unexpected execution preflight command: {command}")
    if script == python_programs.source("managed_scheduler.runtime_identity"):
        (module,) = arguments
        repo_root = str(execution.get("workdir") or Path(__file__).resolve().parents[1])
        payload = {
            "python": python_command,
            "python_version": sys.version.split()[0],
            "runtime_commit": str(execution["runtime_commit"]),
            "runtime_repo_root": repo_root,
            "runtime_hostname": "test-runtime",
            "module": module,
            "module_origin": str(Path(repo_root) / f"{module.replace('.', '/')}.py"),
        }
        return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")

    _module, planned_argv_json, _module_origin = arguments
    planned_argv = json.loads(planned_argv_json)
    supported_options = sorted(
        {token for planned in planned_argv for token in planned["args"] if token.startswith("--")}
    )
    normalized = json.dumps(supported_options, separators=(",", ":"))
    evidence = {
        "supported_options": supported_options,
        "cli_options_sha256": hashlib.sha256(normalized.encode()).hexdigest(),
    }
    return subprocess.CompletedProcess(
        command,
        0,
        f"AGENT_CLI_PREFLIGHT={json.dumps(evidence, sort_keys=True)}\n",
        "",
    )


def config_payload(index_path: Path) -> dict:
    return {
        "model": {
            "backbone": {"name": "roformer", "hidden_size": 8},
            "projection": {"name": "simclr", "enabled": True},
            "cls": {"embedding_type": "bert", "downstream": "tokens"},
            "channels": [{"name": "ppg", "input_dim": 8, "tokenizer": {"name": "linear", "out_dim": 8}}],
            "head": {
                "name": "classification",
                "temporal_agg": {"name": "mean"},
                "channel_agg": {"name": "mean"},
            },
        },
        "data": {
            "backend": "npz",
            "max_tokens": 4,
            "data_channel_names": ["ppg"],
            "finetune_data_index": str(index_path),
            "finetune_preset_path": None,
        },
        "finetune": {
            "task": {
                "type": "classification",
                "output_dim": 30,
                "is_seq": True,
                "monitor": "val_ahi_pearson",
                "monitor_mod": "max",
            }
        },
        "preset_build": {"required_channels": ["ppg", "ahi", "stage5"], "min_channels": 3},
    }


def write_survival_sidecars(tmp_path: Path, *, disease_count: int = 2) -> dict[str, str]:
    diseases = [f"d{i + 1}" for i in range(disease_count)]
    disease_columns = tmp_path / "disease_columns.txt"
    event_time = tmp_path / "event_time.csv"
    is_event = tmp_path / "is_event.csv"
    has_label = tmp_path / "has_label.csv"
    disease_columns.write_text("\n".join(diseases) + "\n")
    header = ",".join(["eid", *diseases])
    event_time.write_text(f"{header}\n001,10,20\n002,30,40\n")
    is_event.write_text(f"{header}\n001,1,0\n002,0,1\n")
    has_label.write_text(f"{header}\n001,1,1\n002,1,1\n")
    return {
        "disease_columns_index": str(disease_columns),
        "event_time_index": str(event_time),
        "is_event_index": str(is_event),
        "has_label_index": str(has_label),
    }


def survival_config_payload(index_path: Path, sidecars: dict[str, str], *, output_dim: int = 2) -> dict:
    payload = config_payload(index_path)
    payload["model"]["head"]["name"] = "regression"
    payload["finetune"]["task"] = {
        "type": "survival",
        "output_dim": output_dim,
        "is_seq": False,
        "monitor": "val_loss",
        "monitor_mod": "min",
    }
    payload["finetune"]["survival"] = {"key_column": "eid", **sidecars}
    payload["preset_build"] = {"required_channels": ["ppg"], "min_channels": 1}
    return payload


def write_yaml(path: Path, payload: dict) -> Path:
    if "task" in payload:
        phase = {
            "preset_prepare": "prepare",
            "pretrain": "train",
            "finetune": "train",
            "hparam_tune": "train",
            "infer": "evaluate",
            "sleep2stat": "analyze",
        }.get(str(payload["task"]), "analyze")
        experiment = payload.get("experiment") or {
            "id": "unit-experiment",
            "title": "Unit experiment",
            "objective": "Exercise agent tooling contracts.",
            "baseline": {"type": "none", "rationale": "unit fixture"},
        }
        experiment = {**experiment, "root": str(path.parent)}
        step = payload.get("step") or {
            "id": f"unit-{str(payload['task']).replace('_', '-')}",
            "phase": phase,
            "purpose": "Exercise the requested agent tooling step.",
        }
        payload = {
            **payload,
            "experiment": experiment,
            "step": step,
        }
        root = Path(experiment["root"])
        root.mkdir(parents=True, exist_ok=True)
        manifest = root / "experiment.yaml"
        if not manifest.exists():
            manifest.write_text(yaml.safe_dump({"experiment": experiment}, sort_keys=False))
        run_manifest = root / "run_manifest.tsv"
        if not run_manifest.exists():
            run_manifest.write_text("step_id\trun_id\n")
    path.write_text(yaml.safe_dump(payload))
    return path


def write_finetune_recipe(tmp_path: Path, *, include_label: bool = True, variant: str = "sleep2vec") -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    index = tmp_path / "index.csv"
    index.write_text("path,split,duration,ppg_mask,ah_event_mask,stage_mask\nx.npz,train,60,1,1,1\n")
    config = write_yaml(tmp_path / "config.yaml", config_payload(index))
    inputs = {"config": str(config), "pretrained_backbone_path": None}
    if include_label:
        inputs["label_name"] = "ahi"
    recipe = {
        "name": "unit_finetune",
        "task": "finetune",
        "variant": variant,
        "experiment": {
            "id": "unit-experiment",
            "title": "Unit experiment",
            "objective": "Exercise agent tooling contracts.",
            "root": str(tmp_path),
            "baseline": {"type": "none", "rationale": "unit fixture"},
        },
        "step": {
            "id": "unit-finetune",
            "phase": "train",
            "purpose": "Run the unit finetune fixture.",
        },
        "inputs": inputs,
        "runtime": {"devices": [0]},
        "artifacts": {
            "results_csv_path": str(tmp_path / "results.csv"),
            "version_name": "unit",
            "overwrite": False,
        },
        "evaluation_policy": {
            "selection_metric": "val_ahi_pearson",
            "selection_mode": "max",
            "selection_split": "val",
            "external_test_locked": True,
            "test_after_fit": False,
        },
        "decisions": {
            "task": {"value": "finetune", "source": "explicit_recipe"},
            "pretrained_backbone_path": {
                "value": None,
                "source": "explicit_recipe",
                "meaning": "train from scratch",
            },
            "train_val_test_policy": {"value": "select on val", "source": "explicit_recipe"},
            "overwrite_policy": {"value": False, "source": "explicit_recipe"},
        },
    }
    return write_yaml(tmp_path / "recipe.yaml", recipe)
