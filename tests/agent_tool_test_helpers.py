from __future__ import annotations

from collections.abc import Callable
import contextlib
import csv
import functools
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import threading
from typing import TypeVar
from unittest import mock

import yaml

from agent_tools import (
    cli,
    execution_snapshot,
    experiment_io,
    experiment_tracking,
    experiments,
    managed_scheduler,
    manifests,
    plan_contract,
    plan_hparam,
    plans,
    python_programs,
    run_artifacts,
)
from agent_tools.adapters.registry import get_adapter
from agent_tools.experiment_workspace import file_sha256, managed_run_parameters
from agent_tools.manifests import write_rows
from agent_tools.models import REPO_ROOT

_T = TypeVar("_T")

# Wall-clock guard for waits on real subprocess chains: login shells, several interpreter starts and git
# probes per launch. The longest chain, a preset ``run.sh --execute``, takes about 2 s idle but 9 s while the
# rest of the suite forks around it under ``-n auto``, and has exceeded 10 s. The guard only bounds a hang,
# so it stays an order of magnitude above that loaded cost rather than near the idle one.
SUBPROCESS_WAIT_SECONDS = 120


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


# Bound at import, before any test patches it, so the wrapper always reaches the real validator as in the stub.
_validate_managed_output_paths = experiment_io.validate_managed_output_paths


def validate_local_managed_output_paths(root, paths):
    """Validate managed output paths through the validator bound before any test patched it."""
    return _validate_managed_output_paths(root, paths)


def run_cli(*args: str) -> subprocess.CompletedProcess:
    """Run the agent_tools CLI in this process with the stub's two substitutions, saving an interpreter per call.

    Use run_cli_subprocess when a test needs a real process: CLI calls from threads, process identity, or a timeout."""
    stdout, stderr = io.StringIO(), io.StringIO()
    with (
        mock.patch.object(execution_snapshot, "run_execution_command", run_execution_preflight_fixture),
        mock.patch.object(experiment_io, "validate_managed_output_paths", validate_local_managed_output_paths),
        contextlib.redirect_stdout(stdout),
        contextlib.redirect_stderr(stderr),
    ):
        try:
            returncode = cli.main(list(args))
        except SystemExit as exc:
            # Map exit requests as the interpreter does; argparse usage errors arrive here with code 2.
            if exc.code is None:
                returncode = 0
            elif isinstance(exc.code, int):
                returncode = exc.code
            else:
                print(exc.code, file=sys.stderr)
                returncode = 1
    return subprocess.CompletedProcess(list(args), returncode, stdout.getvalue(), stderr.getvalue())


def run_cli_subprocess(*args: str) -> subprocess.CompletedProcess:
    """Run the agent_tools CLI through the stub in a fresh interpreter, killed after SUBPROCESS_WAIT_SECONDS."""
    runner = Path(__file__).with_name("agent_tools") / "agent_tools_cli_stub.py"
    return subprocess.run(
        [sys.executable, str(runner), *args], text=True, capture_output=True, timeout=SUBPROCESS_WAIT_SECONDS
    )


def call_while_run_lock_holder_commits(monkeypatch, workspace: Path, call: Callable[[], _T]) -> _T:
    """Run call in a thread while this thread holds the workspace run lock and republishes run_manifest.tsv.

    The caller signals when it stats run_manifest.tsv or contends for the run lock. A stat then parks it between
    stat and open until the republish has replaced the manifest inode, so any managed read made outside the lock
    fails deterministically. Returns the call's result and re-raises its exception."""
    from agent_tools import experiment_workspace
    from agent_tools.experiment_workspace import merge_run_manifest, read_run_manifest

    waiting = threading.Event()
    committed = threading.Event()
    outcome: dict[str, _T] = {}
    failures: list[BaseException] = []
    real_lstat = os.lstat
    real_run_lock = experiment_workspace.managed_run_lock

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
    monkeypatch.setattr(experiment_workspace, "managed_run_lock", run_lock)
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

    original_validate_paths = experiment_io.validate_managed_output_paths

    def validate_managed_output_paths(root, paths):
        return original_validate_paths(root, paths)

    stdout, stderr = StringIO(), StringIO()
    # Match the CLI stub's two substitutions only while preparing this fixture.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(execution_snapshot, "run_execution_command", run_execution_preflight_fixture)
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
        },
    }
    return write_yaml(tmp_path / "recipe.yaml", recipe)


def hparam_search_defaults() -> dict[str, int]:
    """The consultation policy's default hparam search size, keyed by its policy field names."""
    from agent_tools.recipes import load_consultation_policy

    return {key: entry["value"] for key, entry in load_consultation_policy()["hparam_search_defaults"].items()}


class FakeLauncher:
    """Scripted process start for ``hparam_runtime.launch_hparam_runs(..., hooks=launcher.hooks())`` in tests.

    Each start records ``(execution, command)`` in ``starts`` and returns the next outcome, repeating the last one:
    a status such as ``"launched"`` or ``"launch_failed"``, a ``(status, scheduler_reason)`` pair, an exception to
    raise, or a callable that takes ``(execution, command)`` and returns either. A bare status starts with an empty
    reason, as ``start_process`` returns it. ``verify_target=False`` also skips the frozen
    execution-target check and launches without a snapshot, for hand-written plans that froze none and for tests that
    do not exercise the check."""

    def __init__(self, *outcomes, verify_target: bool = True):
        self.outcomes = outcomes or ("launched",)
        self.verify_target = verify_target
        self.starts: list[tuple[dict, str]] = []

    @property
    def commands(self) -> list[str]:
        return [command for _execution, command in self.starts]

    def hooks(self):
        from agent_tools import managed_scheduler

        return managed_scheduler.SchedulerHooks(
            start_process=self.start_process,
            validated_snapshot=None if self.verify_target else lambda *_args: (None, False),
        )

    def start_process(self, execution: dict, command: str) -> tuple[str, str]:
        self.starts.append((execution, command))
        outcome = self.outcomes[min(len(self.starts), len(self.outcomes)) - 1]
        if isinstance(outcome, BaseException):
            raise outcome
        result = outcome(execution, command) if callable(outcome) else outcome
        return result if isinstance(result, tuple) else (result, "")

    def assert_not_started(self) -> None:
        assert self.starts == [], f"Unexpected process starts: {self.commands}"


def frozen_artifact_launch(_execution: dict, command: str) -> tuple[str, str]:
    """FakeLauncher outcome that applies the launch command's own frozen-artifact hash check.

    It reads the artifact list the command passes to its runtime_identity program and returns the definite
    ``launch_failed`` outcome ``start_process`` reports when that program rejects a changed file."""
    identity = json.loads(shlex.split(command)[-1])[0]
    for artifact in json.loads(identity[5]):
        if hashlib.sha256(Path(artifact["path"]).read_bytes()).hexdigest() != artifact["sha256"]:
            return (
                "launch_failed",
                f"launch exit 2: Frozen run artifact changed before process start: {artifact['path']}",
            )
    return "launched", ""


@functools.cache
def checkout_commit() -> str:
    """HEAD of this checkout, read on first use so collection works outside a git checkout."""
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, text=True, capture_output=True
    ).stdout.strip()


def git_commit(repo: Path, message: str, *, allow_empty: bool = False) -> str:
    """Commit what is staged in repo as a fixed test identity, without hooks, and return the new HEAD."""
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "-c",
            "core.hooksPath=/dev/null",
            "commit",
            "-q",
            *(["--allow-empty"] if allow_empty else []),
            "-m",
            message,
        ],
        cwd=repo,
        check=True,
    )
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, text=True, capture_output=True
    ).stdout.strip()


def read_table(path: Path) -> list[dict[str, str]]:
    delimiter = "\t" if path.suffix == ".tsv" else ","
    with path.open(newline="") as file_obj:
        return list(csv.DictReader(file_obj, delimiter=delimiter))


def read_manifest_rows(root: Path) -> list[dict[str, str]]:
    text = (root / "run_manifest.tsv").read_text().splitlines()
    if len(text) <= 1:
        return []
    return list(csv.DictReader(text, delimiter="\t"))


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_files(root: Path) -> dict[Path, bytes]:
    """Every regular file under root by relative path, for before/after no-write assertions."""
    return {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def workspace_files(root: Path) -> dict[Path, bytes]:
    """tree_files without run_manifest.tsv.lock: workspace reads take the run lock, which creates that empty file
    and holds no state."""
    return {path: data for path, data in tree_files(root).items() if path != Path("run_manifest.tsv.lock")}


def plan_first_run(plan_dir: Path) -> dict:
    return json.loads((plan_dir / "plan.json").read_text())["runs"][0]


def hparam_recipe_payload(
    base: Path,
    *,
    name: str,
    search: dict,
    variant: str = "sleep2vec",
    inputs: dict | None = None,
    execution: dict | None = None,
    selection_metric: str = "val_ahi_pearson",
    selection_mode: str = "max",
) -> dict:
    """An hparam_tune recipe over base with a locked test split; inputs and execution are omitted when None."""
    payload = {
        "name": name,
        "task": "hparam_tune",
        "variant": variant,
        "base_recipe": str(base),
        "search": search,
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
            "train_val_test_policy": {"value": "val", "source": "explicit_recipe"},
            "final_eval_unlock": {"value": False, "source": "explicit_recipe"},
        },
    }
    if inputs is not None:
        payload["inputs"] = inputs
    if execution is not None:
        payload["execution"] = execution
    return payload


def write_hparam_recipe(path: Path, base: Path, **fields) -> Path:
    """Write hparam_recipe_payload(base, **fields) to path through write_yaml."""
    return write_yaml(path, hparam_recipe_payload(base, **fields))


def write_variant_hparam_recipe(
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
    """An hparam recipe without execution identity, so the plan uses the manager runtime."""
    base = write_finetune_recipe(tmp_path, variant=variant)
    inputs = {"ckpt_path": str(ckpt_path)} if ckpt_path else {}
    if final_config_path is not None:
        inputs["final_eval_config_path"] = str(final_config_path)
    return write_hparam_recipe(
        tmp_path / f"tune_{variant}.yaml",
        base,
        name=f"unit_tune_{variant}",
        variant=variant,
        inputs=inputs,
        search={"method": "grid", "max_runs": max_runs, "parameters": parameters or {"runtime.lr": [1e-6]}},
        selection_metric=selection_metric,
        selection_mode=selection_mode,
    )


def write_local_hparam_recipe(
    tmp_path: Path,
    *,
    execution: dict | None = None,
    variant: str = "sleep2vec",
    selection_metric: str = "val_ahi_pearson",
    selection_mode: str = "max",
    config_monitor: str | None = None,
    max_runs: int = 1,
) -> Path:
    """An hparam recipe whose base config monitors config_monitor (default: the selection metric).

    Execution defaults to a workdir of tmp_path; outside the manager runtime it is pinned to this interpreter and
    checkout."""
    base = write_finetune_recipe(tmp_path, variant=variant)
    base_payload = yaml.safe_load(base.read_text())
    config_path = Path(base_payload["inputs"]["config"])
    config = yaml.safe_load(config_path.read_text())
    monitor_metric = config_monitor or selection_metric
    config["finetune"]["task"]["monitor"] = monitor_metric
    config["finetune"]["task"]["monitor_mod"] = selection_mode
    write_yaml(config_path, config)
    base_payload["evaluation_policy"]["selection_metric"] = monitor_metric
    base_payload["evaluation_policy"]["selection_mode"] = selection_mode
    write_yaml(base, base_payload)
    execution_payload = dict(execution) if execution is not None else {"workdir": str(tmp_path)}
    manager_runtime = execution_payload.get("workdir") in (None, "", str(REPO_ROOT)) and execution_payload.get(
        "conda_env"
    ) in (None, "")
    if not manager_runtime:
        execution_payload.setdefault("python", sys.executable)
        execution_payload.setdefault("runtime_commit", checkout_commit())
    return write_hparam_recipe(
        tmp_path / "tune.yaml",
        base,
        name="unit_hparam",
        variant=variant,
        search={
            "method": "grid",
            "max_runs": max_runs,
            "parameters": {"runtime.lr": [float(index + 1) * 1e-6 for index in range(max_runs)]},
        },
        execution=execution_payload,
        selection_metric=selection_metric,
        selection_mode=selection_mode,
    )


def local_runtime_execution(workdir: Path) -> dict:
    return {"workdir": str(workdir), "python": sys.executable, "runtime_commit": checkout_commit()}


def bound_config_summary(recipe: dict) -> dict:
    config_bytes = Path(recipe["inputs"]["config"]).read_bytes()
    return {
        "_source_config_bytes": config_bytes,
        "_source_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
    }


def valid_final_config_bytes(tmp_path: Path) -> bytes:
    payload = yaml.safe_load((tmp_path / "config.yaml").read_text())
    payload["model"]["head"].update(
        {
            "channel_agg": {"name": "mean"},
            "temporal_agg": {"name": "mean"},
        }
    )
    return yaml.safe_dump(payload, sort_keys=False).encode()


PRESET_SCRIPTS = {
    "sleep2vec": "preprocess/save_dataset_presets.py",
    "sleep2vec2": "sleep2vec2/preprocess/save_dataset_presets.py",
    "sleep2expert": "sleep2expert/preprocess/save_dataset_presets.py",
}


def write_survival_recipe_with_missing_sidecar_key(tmp_path: Path) -> tuple[Path, Path]:
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
            "train_val_test_policy": {"value": "val", "source": "explicit_recipe"},
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


def write_survival_config_with_bad_sidecars(tmp_path: Path, *, preset_path: Path | None = None) -> Path:
    index = tmp_path / "survival_runtime_index.csv"
    index.write_text("path,split,duration,eid,ppg_mask\na.npz,test,60,001,1\n")
    payload = survival_config_payload(index, _bad_survival_sidecars())
    if preset_path is not None:
        payload["data"]["finetune_preset_path"] = str(preset_path)
    return write_yaml(tmp_path / "bad_survival_config.yaml", payload)


def write_survival_infer_recipe(
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
            "decisions": {
                "task": {"value": "infer", "source": "explicit_recipe"},
                "label_name": {"value": "incident_cox", "source": "explicit_recipe"},
                "ckpt_path": {"value": str(ckpt), "source": "explicit_recipe"},
                "external_test_locked": {"value": False, "source": "explicit_recipe"},
                "final_eval_unlock": {"value": True, "source": "explicit_recipe"},
            },
        },
    )


def write_preset_recipe(
    tmp_path: Path,
    *,
    config: str | Path,
    index: str | Path,
    variant: str = "sleep2vec",
    preset: dict | None = None,
    execution: dict | None = None,
    name: str = "unit_preset",
) -> Path:
    preset = {
        "output_template": str(tmp_path / "presets" / "{dataset}_{split}_{tokens}{meta_suffix}.pickle"),
        **(preset or {"n_tokens": 128, "split": ["train"], "allow_missing_channels": False}),
    }
    config_data = yaml.safe_load(Path(config).read_text())
    preset_build = config_data.get("preset_build") or {}
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


def write_slurm_plan(
    tmp_path: Path,
    *,
    run_count: int = 1,
    execution: dict | None = None,
) -> tuple[Path, dict]:
    """Freeze a Slurm hparam plan in tmp_path/slurm-plan from a direct plan's recipe; return its dir and plan.json."""
    recipe_path = write_local_hparam_recipe(tmp_path)
    if run_count > 1:
        payload = yaml.safe_load(recipe_path.read_text())
        payload["search"]["max_runs"] = run_count
        payload["search"]["parameters"]["runtime.lr"] = [1e-6 * (index + 1) for index in range(run_count)]
        recipe_path.write_text(yaml.safe_dump(payload, sort_keys=False))
    source_plan_dir = tmp_path / "source-plan"
    result = run_cli("plan", "--recipe", str(recipe_path), "--output-dir", str(source_plan_dir))
    assert result.returncode == 0, result.stderr
    source_plan = json.loads((source_plan_dir / "plan.json").read_text())
    recipe = source_plan["recipe"]
    recipe["execution"].update(
        {
            "gpus_per_run": 1,
            "scheduler": {
                "type": "slurm",
                "partition": "gpu",
                "cpus_per_task": 8,
                "memory": "64G",
                "walltime": "01:00:00",
            },
        }
    )
    recipe["execution"].update(execution or {})
    source_config = (source_plan_dir / "config.source.yaml").read_bytes()
    plan_dir = tmp_path / "slurm-plan"
    plan_hparam.write_hparam_plan(
        recipe,
        plan_dir,
        unlock_final_test=False,
        source_config_bytes=source_config,
        source_config_sha256=hashlib.sha256(source_config).hexdigest(),
        final_eval_config=None,
    )
    run_artifacts.preflight_hparam_plan(plan_dir, semantic_out=plan_dir)
    run_artifacts.commit_hparam_plan(plan_dir)
    return plan_dir, json.loads((plan_dir / "plan.json").read_text())


def fake_process_identity(pid: int = 123) -> dict[str, int | str]:
    return {"pid": pid, "process_group_id": pid, "process_start_token": "proc:unit-start"}


def write_fake_process_identity(path: str | Path, pid: int = 123) -> dict[str, int | str]:
    identity = fake_process_identity(pid)
    Path(path).write_text(json.dumps(identity) + "\n")
    return identity


def write_proc_stat(proc_root: Path, pid: int, pgid: int, state: str) -> None:
    process_dir = proc_root / str(pid)
    process_dir.mkdir(parents=True)
    (process_dir / "stat").write_text(f"{pid} (unit) {state} 1 {pgid}\n")


def set_execution_probe(
    monkeypatch,
    plan_dir: Path,
    *,
    commit: str | None = None,
    missing_options: set[str] | None = None,
    extra_options: set[str] | None = None,
    parse_error: str | None = None,
) -> list[str]:
    plan = json.loads((plan_dir / "plan.json").read_text())
    frozen = json.loads((plan_dir / execution_snapshot.EXECUTION_SNAPSHOT_NAME).read_text())
    command = shlex.split(plan["runs"][0]["command"])
    module = command[command.index("-m") + 1]
    options = set(frozen["supported_options"]) - set(missing_options or set())
    options.update(extra_options or set())
    runtime_commit = commit or frozen["runtime_commit"]
    calls = []

    def run_probe(_execution, probe_command):
        if len(probe_command) > 2 and probe_command[2] == python_programs.source("managed_scheduler.runtime_identity"):
            calls.append("identity")
            return subprocess.CompletedProcess(
                probe_command,
                0,
                json.dumps(
                    {
                        "python": frozen["python"],
                        "python_version": frozen["python_version"],
                        "runtime_commit": runtime_commit,
                        "runtime_repo_root": frozen["runtime_repo_root"],
                        "runtime_hostname": frozen["runtime_hostname"],
                        "module": module,
                        "module_origin": frozen["module_origin"],
                    }
                ),
                "",
            )
        calls.append("parse")
        evidence = json.dumps(
            {
                "supported_options": sorted(options),
                "cli_options_sha256": "changed-cli-options" if extra_options else frozen["cli_options_sha256"],
            }
        )
        return subprocess.CompletedProcess(
            probe_command,
            2 if parse_error else 0,
            "" if parse_error else f"AGENT_CLI_PREFLIGHT={evidence}\n",
            parse_error or "",
        )

    monkeypatch.setattr(execution_snapshot, "run_execution_command", run_probe)
    return calls


def write_runtime_rows(root: Path, specs: list[dict]) -> list[dict]:
    """Hand-write a registered direct hparam plan at root with one canonical row per spec; return the rows."""
    experiment = {
        "id": "unit-experiment",
        "title": "Unit experiment",
        "objective": "Exercise hparam runtime state transitions.",
        "root": str(root),
        "baseline": {"type": "none", "rationale": "Unit fixture."},
    }
    step = {"id": "train-model", "phase": "train", "purpose": "Exercise managed runs."}
    (root / "experiment.yaml").write_text(yaml.safe_dump({"experiment": experiment}, sort_keys=False))
    step_dir = root / "steps" / step["id"]
    step_dir.mkdir(parents=True, exist_ok=True)
    (step_dir / "step.yaml").write_text(
        yaml.safe_dump(
            {
                "step": step,
                "experiment_id": experiment["id"],
                "recipe_path": "",
                "plans": [str(root.resolve())],
            },
            sort_keys=False,
        )
    )
    runs = []
    rows = []
    for index, spec in enumerate(specs):
        run_id = str(spec["run_id"])
        managed_dir = root / "runs" / run_id
        managed_dir.mkdir(parents=True, exist_ok=True)
        config = managed_dir / "config.yaml"
        script = managed_dir / "launch.sh"
        artifacts_path = managed_dir / "artifacts.json"
        config.write_text(yaml.safe_dump(config_payload(root / "index.csv")))
        script.write_text("#!/usr/bin/env bash\ntrue\n")
        artifacts_path.write_text("{}\n")
        version = str(spec.get("version") or f"version-{index}")
        runtime_dir = root / "log-finetune" / version
        run = {
            "experiment_id": "unit-experiment",
            "step_id": "train-model",
            "run_id": run_id,
            "run_name": run_id,
            "version": version,
            "run_dir": str(managed_dir),
            "runtime_dir": str(runtime_dir),
            "checkpoint_dir": str(runtime_dir / "checkpoints"),
            "config": str(config),
            "config_sha256": file_sha256(config),
            "script": str(script),
            "script_sha256": file_sha256(script),
            "artifacts": str(artifacts_path),
        }
        runs.append(run)
        row = {
            **run,
            "target": "local",
            "workdir": str(root),
            "gpus": "",
            "pid_path": str(managed_dir / "pid"),
            "log_path": str(managed_dir / "stdout.log"),
            "command": managed_scheduler.build_launch_command(
                {"workdir": str(root)},
                script,
                managed_dir / "stdout.log",
                managed_dir / "pid",
                [],
            ),
            "status": "planned",
            "launched_at": "",
            **spec,
        }
        rows.append(row)
    resolved_recipe = {
        "variant": "sleep2vec",
        "experiment": experiment,
        "step": step,
        "execution": {"workdir": str(root)},
    }
    resolved_path = root / "recipe.resolved.yaml"
    resolved_path.write_text(yaml.safe_dump(resolved_recipe, sort_keys=False))
    snapshot_path = root / execution_snapshot.EXECUTION_SNAPSHOT_NAME
    snapshot_path.write_text("{}\n")
    (root / "plan.json").write_text(
        json.dumps(
            {
                "runs": runs,
                "recipe": resolved_recipe,
                "resolved_recipe_sha256": file_sha256(resolved_path),
                "execution_snapshot": {"path": str(snapshot_path), "sha256": file_sha256(snapshot_path)},
            }
        )
    )
    manifests.write_rows(
        root / "run_manifest.tsv",
        rows,
    )
    (root / "run_manifest.tsv.lock").touch()
    return rows


def init_status_workspace(root: Path) -> None:
    root.mkdir()
    experiment = {
        "id": "status-unit",
        "title": "Status unit experiment",
        "objective": "Exercise the read-only status contract.",
        "root": str(root),
        "baseline": "unit baseline",
    }
    (root / "experiment.yaml").write_text(yaml.safe_dump({"experiment": experiment}, sort_keys=False))
    (root / "run_manifest.tsv").write_text("step_id\trun_id\n")


def add_plan(
    root: Path,
    *,
    step_id: str,
    status: str = "planned",
    task: str = "finetune",
) -> tuple[Path, dict[str, str]]:
    """Register a compiled plan for step_id under plans/ with one canonical row in status; return both."""
    experiment = yaml.safe_load((root / "experiment.yaml").read_text())["experiment"]
    step = {"id": step_id, "phase": "train", "purpose": f"Run {step_id}."}
    plan_dir = root / "plans" / step_id
    recipe_path = root / f"{step_id}.yaml"
    recipe = {
        "name": "default",
        "task": task,
        "experiment": experiment,
        "step": step,
        "inputs": {"config": str(recipe_path), "label_name": "unit"},
        "runtime": {},
        "artifacts": {},
        "execution": {},
        "evaluation_policy": {"test_after_fit": False},
        "_recipe_path": str(recipe_path),
    }
    if task == "sleep2stat":
        recipe["inputs"] = {"config": str(recipe_path)}
        recipe["evaluation_policy"] = {"external_test_locked": True}
    if task != "sleep2stat":
        recipe["variant"] = "sleep2vec"
    if task == "hparam_tune":
        base_recipe = {
            "task": "finetune",
            "variant": "sleep2vec",
            "experiment": experiment,
            "step": step,
        }
        local_recipe = {
            "task": "hparam_tune",
            "variant": "sleep2vec",
            "base_recipe": "base.yaml",
            "experiment": experiment,
            "step": step,
        }
        recipe.update(
            {
                "base_recipe": "base.yaml",
                "search": {"method": "grid", "max_runs": 1, "parameters": {}},
                "execution": {
                    "python": "python",
                    "runtime_commit": "0" * 40,
                    "workdir": str(root),
                    "scheduler": {"type": "direct"},
                },
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
                "_base_recipe": base_recipe,
                "_local_recipe": local_recipe,
            }
        )
    if task == "sleep2stat":
        config_bytes = yaml.safe_dump(
            {"run": {"output_dir": str(root / "analysis")}, "analyzers": [], "reducers": []},
            sort_keys=False,
        ).encode()
    else:
        config_bytes = b"model: unit\n"
    recipe["input_snapshots"] = [
        {
            "field": "inputs.config",
            "path": str(recipe_path),
            "sha256": hashlib.sha256(config_bytes).hexdigest(),
        }
    ]
    plan_contract.bind_plan_context(recipe)
    if task == "hparam_tune":
        run_files = plan_hparam.compile_hparam_run_contracts(recipe, plan_dir, 0, source_config_bytes=config_bytes)[0]
        run = dict(run_files["row"])
    else:
        adapter = get_adapter(task)
        assert adapter is not None
        contract = adapter.compile_plan_contract(recipe, plan_dir, run_index_offset=0, config_bytes=config_bytes)
        run = dict(contract["runs"][0])
    run_dir = Path(run["run_dir"])
    run_dir.mkdir(parents=True)
    config_path = Path(run["config"])
    script_path = Path(run["script"])
    artifacts_path = Path(run["artifacts"])
    if task == "hparam_tune":
        config_path.write_bytes(run_files["config_bytes"])
        script_path.write_text(run_files["script_text"])
        (plan_dir / "config.source.yaml").write_bytes(config_bytes)
        commands = [run["command"]]
    else:
        config_path.write_bytes(config_bytes)
        script_path.write_text(contract["script_text"])
        commands = contract["commands"]
    artifacts_path.write_text("{}\n")
    run.update({"config_sha256": sha256_of(config_path), "script_sha256": sha256_of(script_path)})
    if task != "hparam_tune":
        run["status"] = "planned"
    resolved_recipe = {key: value for key, value in recipe.items() if key != "_recipe_path"}
    resolved_text = yaml.safe_dump(resolved_recipe, sort_keys=False)
    resolved_path = plan_dir / "recipe.resolved.yaml"
    resolved_path.write_text(resolved_text)
    plan_run = dict(run)
    if task != "hparam_tune" and len(commands) == 1:
        plan_run["command"] = commands[0]
    plan = {"status": "PASS", "recipe": recipe, "runs": [plan_run]}
    if task == "hparam_tune":
        plan["resolved_recipe_sha256"] = sha256_of(resolved_path)
        (plan_dir / "run_all.sh").write_text(plan_hparam.compile_hparam_run_all_script(recipe, plan_dir))
    else:
        plan["commands"] = commands
        (plan_dir / "run.sh").write_text(script_path.read_text())
    (plan_dir / "plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")

    step_dir = root / "steps" / step_id
    step_dir.mkdir(parents=True)
    (step_dir / "step.yaml").write_text(
        yaml.safe_dump(
            {
                "step": step,
                "experiment_id": experiment["id"],
                "recipe_path": str(recipe_path),
                "plans": [str(plan_dir)],
            },
            sort_keys=False,
        )
    )
    canonical = {
        **run,
        "status": status,
        "parameter_summary": run.get("parameter_summary", "single resolved recipe"),
    }
    existing = read_manifest_rows(root)
    write_rows(root / "run_manifest.tsv", [*existing, canonical])
    return plan_dir, canonical


def record_hparam_selection(
    root: Path, *, step_id: str = "tune", write_report: bool = False, score: str = "0.25"
) -> Path:
    rows = read_manifest_rows(root)
    winner = next(row for row in rows if row["step_id"] == step_id)
    winner["target"] = winner.get("target") or "local"
    checkpoint = Path(winner["checkpoint_dir"]) / "epoch=1.ckpt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"unit checkpoint\n")
    winner.update(
        {
            "selection_task": "hparam_tune",
            "metric": "val_loss",
            "selection_mode": "min",
            "selection_split": "val",
            "score": score,
            "rank": "1",
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": sha256_of(checkpoint),
            "run_manifest": str(Path(winner["runtime_dir"]) / "run_manifest.json"),
        }
    )
    write_rows(root / "run_manifest.tsv", rows)
    experiment = yaml.safe_load((root / "experiment.yaml").read_text())["experiment"]
    registered = experiments._registered_plan_steps(
        root,
        experiment,
        read_manifest_rows(root),
        require_registered_rows=True,
    )
    lifecycle = experiment_tracking.hparam_selection_lifecycle(
        registered,
        read_manifest_rows(root),
        root=root,
    )
    report_path = root / "reports" / "hparam_selection.md"
    if write_report:
        report_path.parent.mkdir(exist_ok=True)
        report_text = lifecycle["expected_report"]
        report_path.write_text(report_text)
        write_rows(
            root / "reports" / "ranking.csv",
            [
                {
                    "step_id": row["step_id"],
                    "run_id": row["run_id"],
                    "run_name": row["run_name"],
                    "parameter_summary": row["parameter_summary"],
                    "version": row["version"],
                    "config": row["config"],
                    "metric": row["metric"],
                    "score": row["score"],
                    "rank": row["rank"],
                    "checkpoint_path": row["checkpoint_path"],
                    "checkpoint_sha256": row["checkpoint_sha256"],
                    "run_manifest": row.get("run_manifest", ""),
                    "status": row["status"],
                    **managed_run_parameters(row),
                }
                for row in rows
                if row["step_id"] == step_id and row.get("rank") not in (None, "")
            ],
        )
        rows = read_manifest_rows(root)
        digest = hashlib.sha256(report_text.encode()).hexdigest()
        for row in rows:
            if row["step_id"] == step_id:
                row.update({"selection_report": str(report_path), "selection_report_sha256": digest})
        write_rows(root / "run_manifest.tsv", rows)
    return report_path


def write_public_hparam_recipe(
    root: Path,
    parameters: dict,
    *,
    selection_metric: str = "val_ahi_pearson",
    selection_mode: str = "max",
    max_runs: int = 1,
) -> Path:
    base_recipe = write_finetune_recipe(root)
    base_payload = yaml.safe_load(base_recipe.read_text())
    base_payload["evaluation_policy"].update({"selection_metric": selection_metric, "selection_mode": selection_mode})
    base_recipe.write_text(yaml.safe_dump(base_payload, sort_keys=False))
    config_path = Path(base_payload["inputs"]["config"])
    config = yaml.safe_load(config_path.read_text())
    config["finetune"]["task"].update({"monitor": selection_metric, "monitor_mod": selection_mode})
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    return write_hparam_recipe(
        root / "tune.yaml",
        base_recipe,
        name="status_public_hparam",
        inputs={},
        search={"method": "grid", "max_runs": max_runs, "parameters": parameters},
        execution={"workdir": str(root), "python": sys.executable, "runtime_commit": checkout_commit()},
        selection_metric=selection_metric,
        selection_mode=selection_mode,
    )


def write_preflight_hparam_recipe(tmp_path: Path, *, task: str = "hparam_tune") -> tuple[Path, Path]:
    """Return the recipe and its workspace root; hparam_tune pins a separate runtime workdir."""
    base = write_finetune_recipe(tmp_path / "source")
    if task != "hparam_tune":
        recipe_path = base
    else:
        recipe_path = write_hparam_recipe(
            tmp_path / "tune.yaml",
            base,
            name="preflight_hparam",
            search={"method": "grid", "max_runs": 1, "parameters": {"runtime.lr": [1e-6]}},
            execution={
                "workdir": str(tmp_path / "runtime"),
                "python": sys.executable,
                "runtime_commit": checkout_commit(),
            },
        )
    effective, _summary, _report = plans.evaluate_recipe(recipe_path)
    return recipe_path, Path(effective["experiment"]["root"])


def fake_execution_snapshot(execution: dict, runs: list[dict]) -> dict:
    planned_argv = []
    modules = set()
    for run in runs:
        tokens = shlex.split(run["command"])
        module_index = tokens.index("-m") + 1
        modules.add(tokens[module_index])
        planned_argv.append({"run_id": run["run_id"], "args": tokens[module_index + 1 :]})
    module = modules.pop()
    return {
        "workdir": str(execution.get("workdir") or REPO_ROOT),
        "conda_env": str(execution.get("conda_env") or ""),
        "python_command": execution["python"],
        "expected_runtime_commit": execution["runtime_commit"],
        "execution_env_sha256": hashlib.sha256(b"{}").hexdigest(),
        "python": execution["python"],
        "python_version": "3.10.0",
        "runtime_commit": execution["runtime_commit"],
        "runtime_repo_root": str(REPO_ROOT),
        "runtime_hostname": "unit-host",
        "module": module,
        "module_origin": str(REPO_ROOT / Path(*module.split(".")).with_suffix(".py")),
        "required_options": [],
        "supported_options": [],
        "cli_options_sha256": hashlib.sha256(b"[]").hexdigest(),
        "validated_argv_sha256": hashlib.sha256(
            json.dumps(planned_argv, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


def make_preset_runtime(tmp_path: Path) -> dict:
    """A committed runtime checkout whose preset entrypoints record how they were launched."""
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "agent_tools").symlink_to(REPO_ROOT / "agent_tools", target_is_directory=True)
    for script in PRESET_SCRIPTS.values():
        entrypoint = runtime / script
        entrypoint.parent.mkdir(parents=True, exist_ok=True)
        entrypoint.write_text(
            "import json, os, sys\n"
            "from pathlib import Path\n"
            "from agent_tools.experiment_workspace import read_run_manifest\n"
            "payload = {'python': sys.executable, 'cwd': str(Path.cwd()), 'argv': sys.argv,\n"
            "           'status': read_run_manifest(os.environ['PRESET_TEST_WORKSPACE'])[0]['status']}\n"
            "Path(os.environ['PRESET_TEST_PAYLOAD']).write_text(json.dumps(payload))\n"
        )
    subprocess.run(["git", "init", "--quiet", str(runtime)], check=True)
    commit = git_commit(runtime, "Initialize test runtime", allow_empty=True)
    calls = tmp_path / "python-calls.txt"
    runtime_python = runtime / "python-runtime"
    runtime_python.write_text(
        "#!/bin/sh\n"
        f"printf '%s\\n' \"$1\" >> {shlex.quote(str(calls))}\n"
        f'exec {shlex.quote(sys.executable)} "$@"\n'
    )
    runtime_python.chmod(0o755)
    poison = tmp_path / "poison"
    poison.mkdir()
    poison_marker = tmp_path / "poison-ran.txt"
    poison_python = poison / "python"
    poison_python.write_text(f"#!/bin/sh\nprintf poison > {shlex.quote(str(poison_marker))}\nexit 97\n")
    poison_python.chmod(0o755)
    workspace = tmp_path / "workspace"
    payload = tmp_path / "payload.json"
    return {
        "execution": {
            "workdir": str(runtime),
            "python": str(runtime_python),
            "runtime_commit": commit,
        },
        "workspace": workspace,
        "payload": payload,
        "calls": calls,
        "poison_marker": poison_marker,
        "env": {
            **os.environ,
            "PATH": str(poison) + os.pathsep + os.environ["PATH"],
            "PRESET_TEST_WORKSPACE": str(workspace),
            "PRESET_TEST_PAYLOAD": str(payload),
        },
    }


def write_preset_runtime_recipe(tmp_path: Path, preset_runtime: dict, variant: str = "sleep2vec") -> Path:
    source = tmp_path / "source"
    base = write_finetune_recipe(source, variant=variant)
    config = yaml.safe_load(base.read_text())["inputs"]["config"]
    recipe = write_preset_recipe(
        source,
        config=config,
        index=source / "index.csv",
        variant=variant,
        execution=preset_runtime["execution"],
    )
    payload = yaml.safe_load(recipe.read_text())
    payload["experiment"]["root"] = str(preset_runtime["workspace"])
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    return recipe
