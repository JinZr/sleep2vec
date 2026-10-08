from __future__ import annotations

from collections.abc import Callable
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from typing import TypeVar, cast
from unittest import mock

import yaml

from agent_tools import cli, execution_snapshot, experiment_io

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


def validate_local_managed_output_paths(root, paths, *, remote=None):
    """Validate local managed output paths; skip remote validation, which would need a reachable SSH host."""
    if remote is None:
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

    def validate_managed_output_paths(root, paths, *, remote=None):
        if remote is None:
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


def prepare_pipeline_sources(tmp_path: Path, monkeypatch, spec: dict, *, scores: tuple[float, ...] = (4.5,)) -> Path:
    """Publish a managed workspace that a managed pipeline spec can run against; return the spec path.

    The spec's experiment.yaml is active and its run_manifest.tsv holds one completed source run per score, all
    owned by the spec's single checkpoint source plan, with real configs, checkpoints, runtime manifests and the
    spec's job presets. The data readers that need a real hparam plan or torch are stubbed at their module-level
    names: hparam plan reads, ranking and candidate resolution (rank order follows ``scores``), and checkpoint
    payload inspection. Drive the pipeline with ``FakePipelineRuntime(spec).hooks()``."""
    from agent_tools import experiment_pipeline, plan_contract
    from agent_tools.experiment_workspace import file_sha256, managed_run_key
    from agent_tools.manifests import write_rows

    root = Path(spec["checkpoint_sources"][next(iter(spec["checkpoint_sources"]))]["plan"]).parents[1]
    root.mkdir(parents=True, exist_ok=True)
    experiment = {
        "id": spec["pipeline"]["experiment_id"],
        "title": "Unit",
        "objective": "Exercise the managed pipeline end to end.",
        "root": str(root),
        "baseline": {"type": "none"},
        "status": "active",
    }
    (root / "experiment.yaml").write_text(yaml.safe_dump({"experiment": experiment}, sort_keys=False))
    ((source_id, source),) = spec["checkpoint_sources"].items()
    plan_dir = Path(source["plan"])
    plan_dir.mkdir(parents=True)
    (plan_dir / "plan.json").write_text("{}\n")
    (plan_dir / "recipe.resolved.yaml").write_text("task: hparam_tune\n")
    source_recipe = yaml.safe_load(write_finetune_recipe(tmp_path / "source", variant=source["variant"]).read_text())
    config = Path(source_recipe["inputs"]["config"])
    recipe = {
        "task": "hparam_tune",
        "variant": source["variant"],
        "experiment": {"id": experiment["id"], "root": str(root)},
        "step": {"id": "train-age"},
        "inputs": {"label_name": source["label_name"]},
        "evaluation_policy": {
            "selection_metric": source["selection_metric"],
            "selection_mode": source["selection_mode"],
            "selection_split": "val",
        },
        "execution": {"target": "local"},
    }
    runs, ranked = [], []
    for rank, score in enumerate(scores, start=1):
        run_dir = plan_dir / "runs" / f"run-{rank:03d}"
        checkpoint_dir = run_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True)
        checkpoint = checkpoint_dir / "epoch=1.ckpt"
        checkpoint.write_bytes(f"checkpoint {rank}".encode())
        runtime_dir = run_dir / "runtime"
        runtime_dir.mkdir()
        (runtime_dir / "run_manifest.json").write_text(json.dumps({"status": "completed"}) + "\n")
        run = {
            "experiment_id": experiment["id"],
            "step_id": "train-age",
            "run_id": f"run-{rank:03d}",
            "status": "completed",
            "config": str(config),
            "config_sha256": file_sha256(config),
            "checkpoint_dir": str(checkpoint_dir),
            "runtime_dir": str(runtime_dir),
        }
        runs.append(run)
        ranked.append(
            {
                **run,
                "run_name": f"rank-{rank}",
                "rank": str(rank),
                "score": str(score),
                "checkpoint_path": str(checkpoint),
                "checkpoint_sha256": file_sha256(checkpoint),
            }
        )
    write_rows(root / "run_manifest.tsv", runs)
    plan = cast(plan_contract.HparamPlan, {"recipe": recipe, "runs": runs})
    for job in spec["jobs"]:
        preset = Path(job["inference_preset_path"])
        preset.parent.mkdir(parents=True, exist_ok=True)
        preset.write_bytes(f"preset {job['id']}".encode())

    def resolve(_plan_dir, _runs, *, top_k: int = 1, all_candidates: bool = False):
        selected = ranked if all_candidates else ranked[:top_k]
        return [dict(row) for row in selected], {managed_run_key(row): plan for row in selected}

    monkeypatch.setattr(experiment_pipeline.artifacts, "read_hparam_plan", lambda *_args, **_kwargs: plan)
    monkeypatch.setattr(
        experiment_pipeline.artifacts,
        "iter_registered_hparam_plans",
        lambda *_args, **_kwargs: iter([(plan_dir, plan)]),
    )
    monkeypatch.setattr(experiment_pipeline, "select_hparam_candidates", lambda *_args: None)
    monkeypatch.setattr(experiment_pipeline, "resolve_hparam_candidates", resolve)
    monkeypatch.setattr(
        experiment_pipeline,
        "_validate_checkpoint_payload",
        lambda *_args: {"state_dict_key_count": 1, "has_ahi_eval_threshold": False},
    )
    spec_path = tmp_path / f"{spec['pipeline']['id']}.yaml"
    spec_path.write_text(yaml.safe_dump(spec, sort_keys=False))
    return spec_path


def complete_experiment(root: Path) -> None:
    """Mark the managed experiment completed, as experiment finalization does."""
    manifest = yaml.safe_load((root / "experiment.yaml").read_text())
    manifest["experiment"]["status"] = "completed"
    (root / "experiment.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))


class PipelineInterrupted(BaseException):
    """Raised from a pipeline hook to stop the runner the way a killed process would: no failure state is recorded."""


def run_pipeline(spec_path: Path, hooks, *, resume: bool = False, finalized: list | None = None):
    """Execute the pipeline spec through ``run_experiment_pipeline`` with zero-second polls.

    The run directory is the spec's experiment root; finalization calls are appended to ``finalized``."""
    from agent_tools import experiment_pipeline

    spec = yaml.safe_load(spec_path.read_text())
    root = Path(next(iter(spec["checkpoint_sources"].values()))["plan"]).parents[1]
    calls = [] if finalized is None else finalized
    return experiment_pipeline.run_experiment_pipeline(
        root,
        spec_path,
        unlock_final_test=True,
        execute=True,
        resume=resume,
        poll_seconds=0,
        finalize_callback=lambda run_dir, report: calls.append((Path(run_dir), Path(report))),
        hooks=hooks,
    )


def dry_run_pipeline(root: Path, spec: dict):
    """Write spec next to the run directory and return the pipeline's dry-run result."""
    from agent_tools import experiment_pipeline

    spec_path = root.parent / f"{spec['pipeline']['id']}.yaml"
    spec_path.write_text(yaml.safe_dump(spec, sort_keys=False))
    return experiment_pipeline.run_experiment_pipeline(root, spec_path)


class FakePipelineRuntime:
    """Process and runtime-probe effects for ``experiment_pipeline.PipelineHooks`` in tests.

    ``monitor_runs`` only records the source plan it was asked to refresh. ``inspect_target`` returns one constant
    execution snapshot. ``launch_runs`` commits each launchable run's terminal status as its launch script would,
    under the run lock, with ``runtime_commit`` (the spec's by default), and for a successful run writes the
    inference result tree that result verification expects. ``outcome(run)`` picks the terminal status, or ``None``
    to leave the run launchable as a full GPU pool would, and ``metrics(run)`` the reported metrics. Every effect
    is recorded in ``calls``."""

    def __init__(self, spec: dict, *, outcome=None, metrics=None, runtime_commit: str | None = None):
        self.spec = spec
        self.outcome = outcome or (lambda _run: "completed")
        self.metrics = metrics or (lambda _run: {"mae": 4.0})
        self.runtime_commit = spec["runtime"]["runtime_commit"] if runtime_commit is None else runtime_commit
        self.calls: list[tuple[str, list[str]]] = []

    def hooks(self, **overrides):
        from agent_tools import experiment_pipeline

        effects = {
            "monitor_runs": self.monitor_runs,
            "inspect_target": self.inspect_target,
            "launch_runs": self.launch_runs,
            **overrides,
        }
        return experiment_pipeline.PipelineHooks(**effects)

    def monitor_runs(self, plan_dir: Path, **_kwargs) -> Path:
        self.calls.append(("monitor", [str(plan_dir)]))
        return plan_dir

    def inspect_target(self, execution: dict, runs: list[dict], **_kwargs) -> dict:
        self.calls.append(("inspect", [str(run["run_id"]) for run in runs]))
        return {"target": execution.get("target", "local"), "runtime_commit": execution["runtime_commit"]}

    def launch_runs(self, root, _owner_dir, runs, _execution, _runtime, **_kwargs):
        from types import SimpleNamespace

        from agent_tools import experiment_workspace, managed_scheduler
        from agent_tools.experiment_workspace import managed_run_key, merge_run_manifest, read_run_manifest

        self.calls.append(("launch", [str(run["run_id"]) for run in runs]))
        with experiment_workspace.managed_run_lock(root):
            canonical = {managed_run_key(row): row for row in read_run_manifest(root)}
            updates = []
            for run in runs:
                row = canonical[managed_run_key(run)]
                if (row.get("status") or "planned") not in managed_scheduler.LAUNCHABLE_STATUSES:
                    continue
                status = self.outcome(run)
                if status is None:
                    continue
                if status == "completed":
                    self._write_result(run)
                updates.append({**row, "status": status, "runtime_commit": self.runtime_commit})
            committed = merge_run_manifest(root, updates, lock_held=True) if updates else list(canonical.values())
        return SimpleNamespace(committed_rows=committed)

    def _write_result(self, run: dict) -> None:
        # A cohort-phase job id ends with its template id.
        job = next(job for job in self.spec["jobs"] if run["job_id"].split("--")[-1] == job["id"])
        run_dir = Path(str(run["result_root"])) / "infer"
        run_dir.mkdir(parents=True)
        metrics_path = run_dir / "metrics.csv"
        prediction_path = run_dir / "predictions.csv"
        metrics = self.metrics(run)
        metrics_path.write_text("metric,value\n" + "".join(f"{name},{value}\n" for name, value in metrics.items()))
        prediction_path.write_text("prediction\n0.5\n")
        manifest_path = run_dir / "run_manifest.json"
        runtime = self.spec["runtime"]
        manifest = {
            "namespace": job["variant"],
            "config_path": str(run["config"]),
            "label_name": job["label_name"],
            "eval_split": "test",
            "checkpoint": {"input": run["checkpoint"], "resolved_path": run["checkpoint"], "avg_ckpts": 1},
            "runtime": {
                "inference_preset_path": job["inference_preset_path"],
                "batch_size": runtime["batch_size"],
                "accelerator": runtime["accelerator"],
                "precision": runtime["precision"],
                "devices": [0],
            },
            "paths": {
                "run_dir": str(run_dir),
                "metrics_csv_path": str(metrics_path),
                "prediction_csv_path": str(prediction_path),
                "manifest_path": str(manifest_path),
            },
            "prediction_row_count": 1,
            "metrics": metrics,
        }
        manifest_path.write_text(json.dumps(manifest) + "\n")


class FakeLauncher:
    """Scripted process start for ``hparam_runtime.launch_hparam_runs(..., hooks=launcher.hooks())`` in tests.

    Each start records ``(execution, command)`` in ``starts`` and returns the next outcome, repeating the last one:
    a status such as ``"launched"``, ``"pending"`` or ``"launch_failed"``, an exception to raise, or a callable that
    takes ``(execution, command)`` and returns a status. ``verify_target=False`` also skips the frozen
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

    def start_process(self, execution: dict, command: str) -> str:
        self.starts.append((execution, command))
        outcome = self.outcomes[min(len(self.starts), len(self.outcomes)) - 1]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome(execution, command) if callable(outcome) else outcome

    def assert_not_started(self) -> None:
        assert self.starts == [], f"Unexpected process starts: {self.commands}"
