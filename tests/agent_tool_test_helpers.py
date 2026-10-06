from __future__ import annotations

from collections.abc import Callable
import contextlib
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from typing import TypeVar

import yaml

_T = TypeVar("_T")


def run_execution_preflight_fixture(execution: dict, command: list[str]) -> subprocess.CompletedProcess:
    from agent_tools import python_programs

    python_command, flag, script, *arguments = command
    if flag != "-c":
        raise AssertionError(f"Unexpected execution preflight command: {command}")
    if script == python_programs.source("managed_scheduler.runtime_identity"):
        module = arguments[0]
        if len(arguments) > 1:
            if len(arguments) != 4:
                raise AssertionError(f"Unexpected runtime identity arguments: {arguments}")
            expected, artifacts = map(json.loads, arguments[1:3])
            planned_commit = arguments[3]
            assert expected == {}
            assert artifacts == []
            assert len(planned_commit) == 40 and all(
                character.lower() in "0123456789abcdef" for character in planned_commit
            )
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


def call_while_run_lock_holder_commits(monkeypatch, workspace: Path, call: Callable[[], _T]) -> _T:
    """Run call in a thread while this thread holds the workspace run lock and republishes run_manifest.tsv.

    The caller signals when it stats run_manifest.tsv or contends for the run lock. A stat then parks it between
    stat and open until the republish has replaced the manifest inode, so any managed read made outside the lock
    fails deterministically. Returns the call's result and re-raises its exception."""
    from agent_tools import managed_scheduler
    from agent_tools.experiment_workspace import merge_run_manifest, read_run_manifest

    waiting = threading.Event()
    committed = threading.Event()
    outcome: dict[str, _T] = {}
    failures: list[BaseException] = []
    real_lstat = os.lstat
    real_run_lock = managed_scheduler.managed_run_lock

    def lstat(path, *args, **kwargs):
        info = real_lstat(path, *args, **kwargs)
        if threading.current_thread() is caller and os.fspath(path).endswith("run_manifest.tsv"):
            waiting.set()
            assert committed.wait(timeout=5)
        return info

    @contextlib.contextmanager
    def run_lock(root):
        if threading.current_thread() is caller:
            waiting.set()
        with real_run_lock(root):
            yield

    def target():
        try:
            outcome["result"] = call()
        except BaseException as exc:
            failures.append(exc)

    caller = threading.Thread(target=target)
    monkeypatch.setattr(os, "lstat", lstat)
    monkeypatch.setattr(managed_scheduler, "managed_run_lock", run_lock)
    with real_run_lock(workspace):
        caller.start()
        assert waiting.wait(timeout=5)
        # Republish canonical rows unchanged, as a concurrent monitor poll does under the lock.
        merge_run_manifest(workspace, read_run_manifest(workspace), lock_held=True)
        committed.set()
    caller.join(timeout=30)
    assert not caller.is_alive()
    if failures:
        raise failures[0]
    return outcome["result"]


def prepare_hparam_plan_fixture(recipe: Path, plan_dir: Path) -> None:
    from contextlib import redirect_stderr, redirect_stdout
    from io import StringIO

    import pytest

    from agent_tools import cli, experiment_io, managed_scheduler

    original_validate_paths = experiment_io.validate_managed_output_paths

    def validate_managed_output_paths(root, paths, *, remote=None):
        if remote is None:
            return original_validate_paths(root, paths)

    stdout, stderr = StringIO(), StringIO()
    # Match the CLI stub's two substitutions only while preparing this fixture.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(managed_scheduler, "run_execution_command", run_execution_preflight_fixture)
        patch.setattr(experiment_io, "validate_managed_output_paths", validate_managed_output_paths)
        with redirect_stdout(stdout), redirect_stderr(stderr):
            returncode = cli.main(["plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)])
    assert returncode == 0, (
        f"Plan fixture failed (exit={returncode})\n" f"stdout:\n{stdout.getvalue()}\nstderr:\n{stderr.getvalue()}"
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
            "tuning": {
                "preset": "full",
                "groups": {"tokenizers": {"train": False}},
            },
            "task": {
                "type": "classification",
                "output_dim": 30,
                "is_seq": True,
                "monitor": "val_ahi_pearson",
                "monitor_mod": "max",
            },
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
            "train_val_test_policy": {"value": "val", "source": "explicit_recipe"},
            "overwrite_policy": {"value": False, "source": "explicit_recipe"},
        },
    }
    return write_yaml(tmp_path / "recipe.yaml", recipe)


def hparam_search_defaults() -> dict[str, int]:
    """The consultation policy's default hparam search size, keyed by its adaptive recipe fields."""
    from agent_tools.recipes import load_consultation_policy

    return {key: entry["value"] for key, entry in load_consultation_policy()["hparam_search_defaults"].items()}
