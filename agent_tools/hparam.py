from __future__ import annotations

import calendar
import csv
import importlib
from itertools import combinations
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
from typing import Any

import pandas as pd
import yaml

from .manifests import write_text
from .models import REPO_ROOT, module_for_variant
from .progress import read_progress

SSH_TIMEOUT_SECONDS = 10
LAUNCH_TIMEOUT_SECONDS = 60


def launch_hparam_trials(plan_dir: str | Path, *, dry_run: bool = True) -> Path:
    run_dir = Path(plan_dir).expanduser()
    if not run_dir.is_absolute():
        run_dir = run_dir.resolve()
    plan = _read_plan(run_dir)
    recipe = plan.get("recipe") if isinstance(plan.get("recipe"), dict) else {}
    execution = recipe.get("execution") if isinstance(recipe.get("execution"), dict) else {}
    trials = plan.get("trials") if isinstance(plan.get("trials"), list) else []
    target = str(execution.get("target", "local") or "local")
    log_dir = _artifact_dir(run_dir, execution, "log_dir", "logs")
    pid_dir = _artifact_dir(run_dir, execution, "pid_dir", "pids")
    if target != "ssh":
        Path(str(log_dir)).mkdir(parents=True, exist_ok=True)
        Path(str(pid_dir)).mkdir(parents=True, exist_ok=True)
    max_concurrent = int(execution.get("max_concurrent") or len(trials) or 1)
    rows = []
    for index, trial in enumerate(trials):
        trial_id = str(trial["trial_id"])
        script = run_dir / str(trial["script"])
        gpus = _assigned_gpus(recipe, index)
        log_path = _join_artifact_path(log_dir, f"{trial_id}.log", target=target)
        pid_path = _join_artifact_path(pid_dir, f"{trial_id}.pid", target=target)
        command = _launch_command(execution, script, log_path, pid_path, gpus)
        status = "planned"
        if not dry_run and index < max_concurrent:
            status = _start_process(execution, command)
        elif not dry_run:
            status = "pending"
        rows.append(
            {
                "trial_id": trial_id,
                "version": trial.get("version") or f"{recipe.get('name')}-{trial_id}",
                "config": trial.get("config"),
                "script": str(script),
                "target": execution.get("target", "local"),
                "host": execution.get("host", ""),
                "workdir": execution.get("workdir") or str(run_dir),
                "gpus": ",".join(str(item) for item in gpus),
                "log_path": str(log_path),
                "pid_path": str(pid_path),
                "command": command,
                "status": status,
                "launched_at": "" if dry_run or status == "pending" else _now(),
            }
        )
    manifest = run_dir / "launch_manifest.tsv"
    _write_rows(manifest, rows)
    _write_rows(run_dir / "trial_status.tsv", rows)
    return manifest


def monitor_hparam_trials(run_dir: str | Path, *, once: bool = True, health: bool = False) -> Path:
    root = Path(run_dir)
    manifest = _read_rows(root / "launch_manifest.tsv")
    previous_rows = {row.get("trial_id"): row for row in _read_rows(root / "trial_status.tsv")}
    rows = [_status_row(root, row, previous_rows.get(row.get("trial_id"), {}), health=health) for row in manifest]
    rows, manifest = _launch_pending_trials(root, manifest, rows)
    out = root / "trial_status.tsv"
    _write_rows(root / "launch_manifest.tsv", manifest)
    _write_rows(out, rows)
    if not once:
        print(f"wrote {out}")
    return out


def _launch_pending_trials(
    run_dir: Path,
    manifest: list[dict[str, str]],
    status_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    plan_path = run_dir / "plan.json"
    if not plan_path.exists():
        return status_rows, manifest
    plan = json.loads(plan_path.read_text())
    recipe = plan.get("recipe") if isinstance(plan.get("recipe"), dict) else {}
    execution = recipe.get("execution") if isinstance(recipe.get("execution"), dict) else {}
    max_concurrent = int(execution.get("max_concurrent") or len(manifest) or 1)
    active_statuses = {"launched", "running", "unknown_remote"}
    active = sum(row.get("status") in active_statuses for row in status_rows)
    slots = max(max_concurrent - active, 0)
    if slots <= 0:
        return status_rows, manifest

    status_by_trial = {row.get("trial_id"): row for row in status_rows}
    for row in manifest:
        if slots <= 0:
            break
        trial_id = row.get("trial_id")
        current = status_by_trial.get(trial_id, row)
        if current.get("status") != "pending" and row.get("status") != "pending":
            continue
        command = row.get("command")
        if not command:
            row["status"] = "launch_failed"
            current["status"] = "launch_failed"
            continue
        status = _start_process(execution, command)
        launched_at = _now() if status == "launched" else ""
        row["status"] = status
        row["launched_at"] = launched_at
        current["status"] = status
        current["launched_at"] = launched_at
        slots -= 1
    return status_rows, manifest


def stop_hparam_trial(run_dir: str | Path, trial_id: str) -> Path:
    root = Path(run_dir)
    rows = _read_rows(root / "launch_manifest.tsv")
    matched = [row for row in rows if row.get("trial_id") == trial_id]
    if not matched:
        raise ValueError(f"Unknown trial_id: {trial_id}")
    row = matched[0]
    pid = _read_pid(row.get("pid_path"), row)
    if pid is None:
        raise ValueError(f"No recorded PID for trial_id: {trial_id}")
    if row.get("target") == "ssh":
        subprocess.run(
            ["ssh", row["host"], f"kill -TERM {pid}"],
            check=False,
            timeout=SSH_TIMEOUT_SECONDS,
        )
    else:
        os.kill(pid, signal.SIGTERM)
    status_path = root / "trial_status.tsv"
    status_rows = _read_rows(status_path) if status_path.exists() else rows
    for item in status_rows:
        if item.get("trial_id") == trial_id:
            item["status"] = "stopped"
            item["stopped_at"] = _now()
    _write_rows(status_path, status_rows)
    return status_path


def select_hparam_candidates(run_dir: str | Path, metric: str, mode: str) -> Path:
    root = Path(run_dir)
    plan = _read_plan(root)
    recipe = plan.get("recipe") if isinstance(plan.get("recipe"), dict) else {}
    rows = []
    for trial in plan.get("trials", []):
        version = trial.get("version") or f"{recipe.get('name')}-{trial.get('trial_id')}"
        manifest_path = _find_run_manifest(root, str(version), recipe)
        manifest = _read_json(manifest_path) if manifest_path else {}
        score = _metric_value(manifest, metric)
        ckpt = _fixed_checkpoint_path(manifest, manifest_path)
        rows.append(
            {
                "trial_id": trial.get("trial_id"),
                "version": version,
                "metric": metric,
                "score": score,
                "config": trial.get("config"),
                "checkpoint_path": ckpt,
                "run_manifest": str(manifest_path or ""),
                "status": manifest.get("status", ""),
            }
        )
    reverse = mode == "max"
    ranked = sorted(
        rows,
        key=lambda row: _sortable_score(row.get("score"), reverse),
        reverse=reverse,
    )
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
    out = root / "candidate_ranking.csv"
    _write_rows(out, ranked)
    return out


def generate_external_eval(
    run_dir: str | Path,
    selected_csv: str | Path,
    *,
    unlock_final_test: bool,
    kaldi_data_root: str | None = None,
    kaldi_manifest: str | None = None,
    finetune_data_index: str | None = None,
    eval_split: str = "test",
    top_k: int = 1,
    all_candidates: bool = False,
) -> Path:
    if not unlock_final_test:
        raise ValueError("hparam-external-eval requires --unlock-final-test.")
    root = Path(run_dir)
    plan = _read_plan(root)
    recipe = plan.get("recipe") if isinstance(plan.get("recipe"), dict) else {}
    base_recipe = recipe.get("_base_recipe") if isinstance(recipe.get("_base_recipe"), dict) else {}
    base_inputs = base_recipe.get("inputs") if isinstance(base_recipe.get("inputs"), dict) else {}
    rows = _selected_candidate_rows(_read_rows(selected_csv), top_k=top_k, all_candidates=all_candidates)
    config_dir = root / "external_eval_configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    commands = []
    manifest_rows = []
    for index, row in enumerate(rows, start=1):
        source_config = Path(str(row["config"]))
        target_config = config_dir / f"{_candidate_id(row)}_{index:03d}_external.yaml"
        _copy_config_with_data_paths(
            source_config,
            target_config,
            kaldi_data_root=kaldi_data_root,
            kaldi_manifest=kaldi_manifest,
            finetune_data_index=finetune_data_index,
        )
        checkpoint_path = _first_value(row, ["checkpoint_path", "fixed_checkpoint_path", "ckpt_path"])
        if not checkpoint_path:
            raise ValueError(f"Selected row is missing checkpoint_path: {_candidate_id(row)}")
        command = _render_command(
            [
                "python",
                "-m",
                module_for_variant(str(recipe.get("variant")), "infer"),
                "--config",
                str(target_config),
                "--ckpt-path",
                checkpoint_path,
                "--label-name",
                base_inputs.get("label_name") or (recipe.get("inputs") or {}).get("label_name"),
                "--eval-split",
                eval_split,
            ]
        )
        commands.append(command)
        manifest_rows.append({**row, "external_config": str(target_config), "external_command": command})
    _write_rows(root / "external_eval_manifest.tsv", manifest_rows)
    write_text(
        root / "external_eval.sh",
        "\n".join(_script_lines(commands)) + "\n",
        executable=True,
    )
    return root / "external_eval.sh"


def export_hparam_logits(
    run_dir: str | Path,
    selected_csv: str | Path,
    *,
    unlock_final_test: bool,
    val_split: str = "val",
    test_split: str = "test",
    skip_test: bool = False,
    label_name: str | None = None,
    val_kaldi_data_root: str | None = None,
    val_kaldi_manifest: str | None = None,
    val_finetune_data_index: str | None = None,
    test_kaldi_data_root: str | None = None,
    test_kaldi_manifest: str | None = None,
    test_finetune_data_index: str | None = None,
    batch_size: int = 12,
    num_workers: int = 8,
    devices: list[int] | None = None,
    accelerator: str = "gpu",
    device: str = "cuda",
    precision: str = "bf16-mixed",
    seed: int = 4523,
    top_k: int = 1,
    all_candidates: bool = False,
    execute: bool = False,
) -> Path:
    if not skip_test and not unlock_final_test:
        raise ValueError("hparam-export-logits requires --unlock-final-test unless --skip-test is used.")  # noqa: E501

    root = Path(run_dir)
    plan = _read_plan(root)
    recipe = plan.get("recipe") if isinstance(plan.get("recipe"), dict) else {}
    base_recipe = recipe.get("_base_recipe") if isinstance(recipe.get("_base_recipe"), dict) else {}
    base_inputs = base_recipe.get("inputs") if isinstance(base_recipe.get("inputs"), dict) else {}
    resolved_label = label_name or base_inputs.get("label_name") or (recipe.get("inputs") or {}).get("label_name")
    if not resolved_label:
        raise ValueError(
            "hparam-export-logits requires --label-name when the hparam plan has no base label_name."  # noqa: E501
        )

    rows = _selected_candidate_rows(_read_rows(selected_csv), top_k=top_k, all_candidates=all_candidates)
    config_dir = root / "logits_export_configs"
    output_dir = root / "logits_exports"
    config_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    for index, row in enumerate(rows, start=1):
        checkpoint_path = _first_value(row, ["checkpoint_path", "fixed_checkpoint_path", "ckpt_path"])
        if not checkpoint_path:
            raise ValueError(f"Selected row is missing checkpoint_path: {_candidate_id(row)}")

        candidate = _candidate_id(row)
        source_config = Path(str(row["config"]))
        val_config = config_dir / f"{candidate}_{index:03d}_{val_split}.yaml"
        _copy_config_with_data_paths(
            source_config,
            val_config,
            kaldi_data_root=val_kaldi_data_root,
            kaldi_manifest=val_kaldi_manifest,
            finetune_data_index=val_finetune_data_index,
        )
        val_logits_path = output_dir / f"{candidate}_{index:03d}_{val_split}_logits.csv"
        manifest_row = {
            **row,
            "checkpoint_path": checkpoint_path,
            "label_name": resolved_label,
            "val_split": val_split,
            "val_config": str(val_config),
            "val_logits_path": str(val_logits_path),
            "val_infer_command": _infer_command(
                recipe,
                val_config,
                checkpoint_path,
                resolved_label,
                val_split,
                batch_size=batch_size,
                num_workers=num_workers,
                devices=devices,
                accelerator=accelerator,
                device=device,
                precision=precision,
                seed=seed,
            ),
        }

        if not skip_test:
            test_config = config_dir / f"{candidate}_{index:03d}_{test_split}.yaml"
            _copy_config_with_data_paths(
                source_config,
                test_config,
                kaldi_data_root=test_kaldi_data_root,
                kaldi_manifest=test_kaldi_manifest,
                finetune_data_index=test_finetune_data_index,
            )
            test_logits_path = output_dir / f"{candidate}_{index:03d}_{test_split}_logits.csv"
            manifest_row.update(
                {
                    "test_split": test_split,
                    "test_config": str(test_config),
                    "test_logits_path": str(test_logits_path),
                    "test_infer_command": _infer_command(
                        recipe,
                        test_config,
                        checkpoint_path,
                        resolved_label,
                        test_split,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        devices=devices,
                        accelerator=accelerator,
                        device=device,
                        precision=precision,
                        seed=seed,
                    ),
                }
            )
        manifest_rows.append(manifest_row)

    manifest = root / "logits_export_manifest.tsv"
    _write_rows(manifest, manifest_rows)
    if execute:
        _execute_logit_exports(
            recipe,
            manifest_rows,
            batch_size=batch_size,
            num_workers=num_workers,
            devices=devices,
            accelerator=accelerator,
            device=device,
            precision=precision,
            seed=seed,
            skip_test=skip_test,
        )
    return manifest


def scan_hparam_checkpoints(run_dir: str | Path, metric: str, mode: str, *, top_k: int | None = None) -> Path:
    root = Path(run_dir)
    plan = _read_plan(root)
    recipe = plan.get("recipe") if isinstance(plan.get("recipe"), dict) else {}
    rows = []
    for trial in plan.get("trials", []):
        version = trial.get("version") or f"{recipe.get('name')}-{trial.get('trial_id')}"
        manifest_path = _find_run_manifest(root, str(version), recipe)
        manifest = _read_json(manifest_path) if manifest_path else {}
        rows.extend(_checkpoint_scan_rows(trial, str(version), metric, manifest_path, manifest))
    reverse = mode == "max"
    ranked = sorted(
        rows,
        key=lambda row: _sortable_score(row.get("score"), reverse),
        reverse=reverse,
    )
    if top_k is not None:
        ranked = ranked[:top_k]
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
    out = root / "checkpoint_ranking.csv"
    _write_rows(out, ranked)
    return out


def threshold_hparam_outputs(run_dir: str | Path, selected_csv: str | Path) -> Path:
    rows = []
    for row in _read_rows(selected_csv):
        val_path = _first_value(row, ["val_predictions_path", "val_logits_path"])
        test_path = _first_value(row, ["test_predictions_path", "test_logits_path"])
        if not val_path or not test_path:
            continue
        label_name = _first_value(row, ["label_name", "target_name", "target_label"])
        val = _read_binary_predictions(val_path, label_name=label_name)
        test = _read_binary_predictions(test_path, label_name=label_name)
        threshold = _best_f1_threshold(val["y"], val["p"])
        val_metrics = _binary_metrics(val["y"], val["p"], threshold)
        test_metrics = _binary_metrics(test["y"], test["p"], threshold)
        rows.append(
            {
                "trial_id": row.get("trial_id"),
                "threshold": threshold,
                **{f"val_{key}": value for key, value in val_metrics.items()},
                **{f"test_{key}": value for key, value in test_metrics.items()},
            }
        )
    out = Path(run_dir) / "threshold_summary.csv"
    _write_rows(out, rows)
    return out


def ensemble_hparam_outputs(
    run_dir: str | Path,
    candidates_csv: str | Path,
    *,
    search_combinations: bool = False,
    max_size: int | None = None,
    metric: str = "exploratory_test_auroc",
    mode: str = "max",
    top_k: int | None = None,
) -> Path:
    rows = _read_rows(candidates_csv)
    usable = [row for row in rows if _first_value(row, ["val_predictions_path", "val_logits_path"])]
    summary = []
    if usable and search_combinations:
        largest = min(max_size or len(usable), len(usable))
        for size in range(1, largest + 1):
            for combo in combinations(usable, size):
                summary.append(_ensemble_summary_row(list(combo)))
        reverse = mode == "max"
        summary = sorted(
            summary,
            key=lambda row: _sortable_score(row.get(metric), reverse),
            reverse=reverse,
        )
        if top_k is not None:
            summary = summary[:top_k]
        for rank, row in enumerate(summary, start=1):
            row["rank"] = rank
            row["rank_metric"] = metric
    elif usable:
        summary.append(_ensemble_summary_row(usable))
    out = Path(run_dir) / "ensemble_summary.csv"
    _write_rows(out, summary)
    return out


def _read_plan(run_dir: Path) -> dict[str, Any]:
    plan_path = run_dir / "plan.json"
    if not plan_path.exists():
        raise FileNotFoundError(f"Missing hparam plan: {plan_path}")
    return json.loads(plan_path.read_text())


def _selected_candidate_rows(
    rows: list[dict[str, str]], *, top_k: int = 1, all_candidates: bool = False
) -> list[dict[str, str]]:
    if all_candidates:
        return rows
    selected = []
    for row in rows:
        rank = row.get("rank")
        if rank in (None, ""):
            selected.append(row)
            continue
        try:
            if int(float(rank)) <= top_k:
                selected.append(row)
        except ValueError:
            continue
    return selected


def _checkpoint_scan_rows(
    trial: dict[str, Any],
    version: str,
    metric: str,
    manifest_path: Path | None,
    manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    if manifest_path:
        for epoch, score in _history_metric_rows(manifest_path.parent, metric):
            checkpoint = _checkpoint_for_epoch(manifest_path, epoch)
            if checkpoint:
                rows.append(
                    {
                        "trial_id": trial.get("trial_id"),
                        "version": version,
                        "config": trial.get("config"),
                        "metric": metric,
                        "score": score,
                        "epoch": epoch,
                        "checkpoint_path": str(checkpoint),
                        "run_manifest": str(manifest_path),
                        "source": "history",
                    }
                )
    if rows:
        return rows
    score = _metric_value(manifest, metric)
    checkpoint = _fixed_checkpoint_path(manifest, manifest_path)
    if score not in ("", None) and checkpoint:
        rows.append(
            {
                "trial_id": trial.get("trial_id"),
                "version": version,
                "config": trial.get("config"),
                "metric": metric,
                "score": score,
                "epoch": manifest.get("epoch") or _epoch_from_checkpoint_name(Path(checkpoint).name),
                "checkpoint_path": checkpoint,
                "run_manifest": str(manifest_path or ""),
                "source": "manifest",
            }
        )
    return rows


def _history_metric_rows(run_dir: Path, metric: str) -> list[tuple[int, float]]:
    by_epoch: dict[int, float] = {}
    for record in _history_records(run_dir):
        if metric not in record:
            continue
        epoch = _history_epoch(record)
        score = _float_or_none(record.get(metric))
        if epoch is not None and score is not None:
            by_epoch[epoch] = score
    return sorted(by_epoch.items())


def _history_records(run_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(run_dir.glob("wandb/**/wandb-history*.jsonl")):
        for line in path.read_text(errors="replace").splitlines():
            if line.strip():
                records.append(json.loads(line))
    for path in sorted(run_dir.glob("wandb/**/wandb-history*.csv")):
        with path.open(newline="") as file_obj:
            records.extend(csv.DictReader(file_obj))
    history = (
        _read_json(run_dir / "run_manifest.json").get("history") if (run_dir / "run_manifest.json").exists() else None
    )
    if isinstance(history, list):
        records.extend(row for row in history if isinstance(row, dict))
    return records


def _history_epoch(record: dict[str, Any]) -> int | None:
    for key in ("epoch", "trainer/epoch", "current_epoch", "global_epoch"):
        value = _float_or_none(record.get(key))
        if value is not None:
            return int(value)
    return None


def _checkpoint_for_epoch(manifest_path: Path, epoch: int) -> Path | None:
    return _checkpoint_for_epoch_in_dir(manifest_path.parent / "checkpoints", epoch)


def _checkpoint_for_epoch_in_dir(ckpt_dir: Path, epoch: int | None) -> Path | None:
    if epoch is None:
        return None
    for path in sorted(ckpt_dir.glob("epoch=*.ckpt")):
        if not path.name.startswith("best-") and _epoch_number_from_checkpoint_name(path.name) == epoch:
            return path
    return None


def _epoch_from_checkpoint_name(name: str) -> str:
    if not name.startswith("epoch="):
        return ""
    return name.split("=", 1)[1].split("-", 1)[0].split(".", 1)[0]


def _epoch_number(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(str(value)))
    except ValueError:
        return None


def _epoch_number_from_checkpoint_name(name: str) -> int | None:
    return _epoch_number(_epoch_from_checkpoint_name(name))


def _local_dir(run_dir: Path, raw: Any, default_name: str) -> Path:
    if raw in (None, ""):
        return run_dir / default_name
    path = Path(str(raw))
    return path if path.is_absolute() else run_dir / path


def _artifact_dir(run_dir: Path, execution: dict[str, Any], key: str, default_name: str) -> str | Path:
    raw = execution.get(key)
    if execution.get("target", "local") == "ssh":
        base = str(execution.get("workdir") or run_dir)
        value = str(raw or default_name)
        if value.startswith("/"):
            return value
        return f"{base.rstrip('/')}/{value}"
    return _local_dir(run_dir, raw, default_name)


def _join_artifact_path(directory: str | Path, name: str, *, target: str) -> str | Path:
    if target == "ssh":
        return f"{str(directory).rstrip('/')}/{name}"
    return Path(str(directory)) / name


def _assigned_gpus(recipe: dict[str, Any], trial_index: int) -> list[Any]:
    execution = recipe.get("execution") if isinstance(recipe.get("execution"), dict) else {}
    base = recipe.get("_base_recipe") if isinstance(recipe.get("_base_recipe"), dict) else {}
    base_runtime = base.get("runtime") if isinstance(base.get("runtime"), dict) else {}
    base_devices = _as_list(base_runtime.get("devices"))
    pool = _as_list(execution.get("gpu_pool")) or base_devices
    if not pool:
        return []
    per_trial = int(execution.get("gpus_per_trial") or len(base_devices) or 1)
    start = (trial_index * per_trial) % len(pool)
    return [pool[(start + offset) % len(pool)] for offset in range(per_trial)]


def _as_list(value: Any) -> list[Any]:
    if value in (None, "", "ASK_USER"):
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _launch_command(
    execution: dict[str, Any],
    script: Path,
    log_path: str | Path,
    pid_path: str | Path,
    gpus: list[Any],
) -> str:
    env = dict(execution.get("env") or {})
    if execution.get("wandb_project"):
        env["WANDB_PROJECT"] = execution["wandb_project"]
    if execution.get("wandb_group"):
        env["WANDB_RUN_GROUP"] = execution["wandb_group"]
        env["WANDB_GROUP"] = execution["wandb_group"]
    if gpus:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(item) for item in gpus)
    run = ["bash", str(script)]
    if execution.get("conda_env"):
        run = [
            "conda",
            "run",
            "--no-capture-output",
            "-n",
            str(execution["conda_env"]),
            *run,
        ]
    env_prefix = " ".join(f"{key}={_sh(value)}" for key, value in sorted(env.items()))
    run_command = " ".join(_sh(part) for part in run)
    if env_prefix:
        run_command = f"{env_prefix} {run_command}"
    workdir = execution.get("workdir") or str(script.parent)
    if execution.get("target", "local") == "ssh":
        mkdir = f"mkdir -p {_sh(_parent_path(log_path))} {_sh(_parent_path(pid_path))}"
        inner = (
            f"{mkdir} && cd {_sh(workdir)} && "
            f"(nohup {run_command} > {_sh(log_path)} 2>&1 & echo $! > {_sh(pid_path)})"
        )
        return f"ssh {_sh(execution['host'])} {_sh(inner)}"
    inner = (
        f"cd {_sh(workdir)} && (nohup {run_command} > {_sh(log_path)} 2>&1 & echo $! > {_sh(pid_path)})"  # noqa: E501
    )
    return inner


def _parent_path(path: str | Path) -> str:
    text = str(path)
    parent = text.rsplit("/", 1)[0] if "/" in text else "."
    return parent or "/"


def _start_process(execution: dict[str, Any], command: str) -> str:
    try:
        result = subprocess.run(
            ["bash", "-lc", command],
            text=True,
            capture_output=True,
            timeout=LAUNCH_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return "launch_failed"
    return "launched" if result.returncode == 0 else "launch_failed"


def _status_row(
    run_dir: Path,
    row: dict[str, Any],
    previous: dict[str, Any] | None = None,
    *,
    health: bool = False,
) -> dict[str, Any]:
    previous = previous or {}
    pid = _read_pid(row.get("pid_path"), row)
    running_state = _process_running(row, pid) if pid is not None else False
    running = bool(running_state)
    status = row.get("status") or "unknown"
    if pid is None and status == "launched" and _is_remote_row(row):
        status = "unknown_remote"
    elif pid is None and status == "launched":
        status = "missing_pid"
    elif running_state is None:
        status = "unknown_remote"
    elif running:
        status = "running"
    elif status in {"launched", "running"}:
        status = "failed" if _log_has_failure(row.get("log_path"), row) else "finished"
    manifest = _find_run_manifest(run_dir, row.get("version", ""), {"execution": {}})
    checkpoints = _checkpoint_names(manifest)
    output = {
        **row,
        "status": status,
        "pid": pid or "",
        "log_tail": _log_tail(row.get("log_path"), row),
        "run_manifest": str(manifest or ""),
        "checkpoints": ";".join(checkpoints),
        "monitored_at": _now(),
    }
    if health:
        output.update(_health_fields(run_dir, row, previous, pid, running_state, status, checkpoints))
    return output


def _read_pid(path: Any, row: dict[str, Any] | None = None) -> int | None:
    if not path:
        return None
    if _is_remote_row(row):
        result = _run_row_command(row or {}, f"cat {_sh(path)}")
        if result.returncode != 0:
            return None
        text = result.stdout.strip()
    else:
        pid_path = Path(str(path))
        if not pid_path.exists():
            return None
        text = pid_path.read_text().strip()
    try:
        return int(text)
    except ValueError:
        return None


def _process_running(row: dict[str, Any], pid: int | None) -> bool | None:
    if pid is None:
        return False
    if row.get("target") == "ssh" and row.get("host"):
        result = _run_row_command(row, f"ps -p {pid} -o pid=")
        if result.returncode in {124, 255}:
            return None
        return result.returncode == 0 and str(pid) in result.stdout
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _log_has_failure(path: Any, row: dict[str, Any] | None = None) -> bool:
    if not path:
        return False
    if _is_remote_row(row):
        result = _run_row_command(row or {}, f"tail -n 100 {_sh(path)}")
        if result.returncode != 0:
            return False
        tail = result.stdout
    else:
        log_path = Path(str(path))
        if not log_path.exists():
            return False
        tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-100:])
    return any(
        marker in tail
        for marker in [
            "Traceback",
            "RuntimeError",
            "CUDA out of memory",
            "Error executing job",
        ]
    )


def _log_tail(path: Any, row: dict[str, Any] | None = None, lines: int = 8) -> str:
    if not path:
        return ""
    if _is_remote_row(row):
        result = _run_row_command(row or {}, f"tail -n {int(lines)} {_sh(path)}")
        return result.stdout.strip() if result.returncode == 0 else ""
    log_path = Path(str(path))
    if not log_path.exists():
        return ""
    return "\n".join(log_path.read_text(errors="replace").splitlines()[-lines:])


def _health_fields(
    run_dir: Path,
    row: dict[str, Any],
    previous: dict[str, Any],
    pid: int | None,
    running_state: bool | None,
    status: str,
    checkpoints: list[str],
) -> dict[str, Any]:
    progress = _read_trial_progress(run_dir, row)
    io_counts = _proc_io(row, pid)
    read_bytes = io_counts.get("read_bytes")
    write_bytes = io_counts.get("write_bytes")
    read_delta = _delta(read_bytes, previous.get("io_read_bytes"))
    write_delta = _delta(write_bytes, previous.get("io_write_bytes"))
    log_age = _log_age_seconds(row.get("log_path"), row)
    gpu_summary = _gpu_summary(row, pid)
    checkpoint_count = len(checkpoints)
    health_status = _classify_health(
        status=status,
        running_state=running_state,
        gpu_summary=gpu_summary,
        io_read_delta=read_delta,
        io_write_delta=write_delta,
        progress=progress,
        progress_is_fresh=_progress_is_fresh(progress, previous),
        log_age_seconds=log_age,
        checkpoint_count=checkpoint_count,
        previous_checkpoint_count=_to_int(previous.get("checkpoint_count")),
    )
    return {
        "health_status": health_status,
        "gpu_summary": gpu_summary,
        "io_read_bytes": "" if read_bytes is None else read_bytes,
        "io_write_bytes": "" if write_bytes is None else write_bytes,
        "io_read_delta_bytes": "" if read_delta is None else read_delta,
        "io_write_delta_bytes": "" if write_delta is None else write_delta,
        "progress_status": progress.get("status", ""),
        "progress_processed": progress.get("processed", ""),
        "progress_total": progress.get("total", ""),
        "progress_updated_at": progress.get("updated_at", ""),
        "progress_age_seconds": _progress_age_seconds(progress),
        "log_age_seconds": "" if log_age is None else log_age,
        "checkpoint_count": checkpoint_count,
    }


def _read_trial_progress(run_dir: Path, row: dict[str, Any]) -> dict[str, Any]:
    progress_dir = row.get("progress_dir") or row.get("workdir") or run_dir
    try:
        return read_progress(progress_dir, remote=row.get("host") if _is_remote_row(row) else None)
    except Exception as exc:
        return {"status": "unknown", "message": str(exc)}


def _proc_io(row: dict[str, Any], pid: int | None) -> dict[str, int]:
    if pid is None:
        return {}
    result = _run_row_command(row, f"cat /proc/{int(pid)}/io")
    if result.returncode != 0:
        return {}
    counts: dict[str, int] = {}
    for line in result.stdout.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        try:
            counts[key.strip()] = int(value.strip())
        except ValueError:
            pass
    return counts


def _gpu_summary(row: dict[str, Any], pid: int | None) -> str:
    if pid is None:
        return ""
    apps = _run_row_command(
        row,
        "nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader,nounits",  # noqa: E501
    )
    if apps.returncode != 0:
        return ""
    matched = []
    for line in apps.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if parts and parts[0] == str(pid):
            matched.append(line.strip())
    if not matched:
        return ""
    gpu_state = _run_row_command(
        row,
        "nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits",  # noqa: E501
    )
    summary = "; ".join(matched)
    if gpu_state.returncode == 0 and gpu_state.stdout.strip():
        summary = f"{summary} | gpu={gpu_state.stdout.strip().replace(chr(10), '; ')}"
    return summary


def _log_age_seconds(path: Any, row: dict[str, Any]) -> int | None:
    if not path:
        return None
    if _is_remote_row(row):
        result = _run_row_command(
            row,
            f"now=$(date +%s); m=$(stat -c %Y {_sh(path)} 2>/dev/null) || exit 1; echo $((now-m))",  # noqa: E501
        )
        if result.returncode != 0:
            return None
        return _to_int(result.stdout.strip())
    log_path = Path(str(path))
    if not log_path.exists():
        return None
    return int(time.time() - log_path.stat().st_mtime)


def _classify_health(
    *,
    status: str,
    running_state: bool | None,
    gpu_summary: str,
    io_read_delta: int | None,
    io_write_delta: int | None,
    progress: dict[str, Any],
    progress_is_fresh: bool,
    log_age_seconds: int | None,
    checkpoint_count: int,
    previous_checkpoint_count: int | None,
) -> str:
    if status == "unknown_remote" or running_state is None:
        return "unknown_remote"
    if status == "failed":
        return "failed"
    if status == "finished":
        return "finished"
    if not running_state:
        return status
    if gpu_summary:
        return "compute_active"
    if (io_read_delta or 0) > 0 or (io_write_delta or 0) > 0:
        return "data_loading"
    if progress.get("status") == "running" and progress_is_fresh:
        return "healthy_running"
    if log_age_seconds is not None and log_age_seconds < 300:
        return "healthy_running"
    if previous_checkpoint_count is not None and checkpoint_count > previous_checkpoint_count:
        return "healthy_running"
    return "possibly_stalled"


def _delta(current: int | None, previous: Any) -> int | None:
    old = _to_int(previous)
    if current is None or old is None:
        return None
    return max(int(current) - old, 0)


def _to_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _progress_is_fresh(progress: dict[str, Any], previous: dict[str, Any]) -> bool:
    if progress.get("status") != "running":
        return False
    processed = _to_int(progress.get("processed"))
    previous_processed = _to_int(previous.get("progress_processed"))
    if processed is not None and previous_processed is not None and processed > previous_processed:
        return True
    age = _progress_age_seconds(progress)
    return age is not None and age < 300


def _progress_age_seconds(progress: dict[str, Any]) -> int | None:
    updated = progress.get("updated_at")
    if not updated:
        return None
    try:
        parsed = time.strptime(str(updated), "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        return None
    return max(int(time.time() - calendar.timegm(parsed)), 0)


def _run_row_command(row: dict[str, Any], command: str) -> subprocess.CompletedProcess:
    try:
        if _is_remote_row(row):
            return subprocess.run(
                ["ssh", str(row["host"]), command],
                text=True,
                capture_output=True,
                timeout=SSH_TIMEOUT_SECONDS,
            )
        return subprocess.run(
            ["bash", "-lc", command],
            text=True,
            capture_output=True,
            timeout=SSH_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        args = exc.cmd if isinstance(exc.cmd, list) else [str(exc.cmd)]
        return subprocess.CompletedProcess(args, 124, "", f"timed out after {SSH_TIMEOUT_SECONDS}s")


def _is_remote_row(row: dict[str, Any] | None) -> bool:
    return bool(row and row.get("target") == "ssh" and row.get("host"))


def _find_run_manifest(run_dir: Path, version: str, recipe: dict[str, Any]) -> Path | None:
    candidates = [
        run_dir / "log-finetune" / version / "run_manifest.json",
        Path("log-finetune") / version / "run_manifest.json",
        run_dir / version / "run_manifest.json",
    ]
    execution = recipe.get("execution") if isinstance(recipe.get("execution"), dict) else {}
    if execution.get("log_finetune_root"):
        candidates.insert(0, Path(str(execution["log_finetune_root"])) / version / "run_manifest.json")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = list(run_dir.glob(f"**/{version}/run_manifest.json"))
    return matches[0] if matches else None


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    return json.loads(path.read_text())


def _metric_value(manifest: dict[str, Any], metric: str) -> float | str:
    metrics = manifest.get("metrics") if isinstance(manifest.get("metrics"), dict) else {}
    if metric in metrics:
        return metrics[metric]
    if manifest.get("monitor") == metric and manifest.get("best_model_score") is not None:
        return manifest["best_model_score"]
    return ""


def _fixed_checkpoint_path(manifest: dict[str, Any], manifest_path: Path | None) -> str:
    raw = manifest.get("best_model_path") or manifest.get("checkpoint_path") or ""
    if raw:
        path = Path(str(raw))
        if path.name.startswith("best-epoch="):
            fixed = path.with_name(path.name.removeprefix("best-"))
            if fixed.exists():
                return str(fixed)
            matched = _checkpoint_for_epoch_in_dir(path.parent, _epoch_number_from_checkpoint_name(fixed.name))
            if matched:
                return str(matched)
        if path.name.startswith("epoch="):
            return str(path)
        matched = _checkpoint_for_epoch_in_dir(path.parent, _epoch_number(manifest.get("epoch")))
        if matched:
            return str(matched)
        return str(path)
    if manifest_path:
        checkpoints = sorted((manifest_path.parent / "checkpoints").glob("epoch=*.ckpt"))
        if checkpoints:
            return str(checkpoints[-1])
    return ""


def _checkpoint_names(manifest_path: Path | None) -> list[str]:
    if manifest_path is None:
        return []
    ckpt_dir = manifest_path.parent / "checkpoints"
    if not ckpt_dir.exists():
        return []
    return [path.name for path in sorted(ckpt_dir.glob("*.ckpt"))]


def _sortable_score(value: Any, reverse: bool) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return -math.inf if reverse else math.inf
    return score


def _copy_config_with_data_paths(
    source: Path,
    target: Path,
    *,
    kaldi_data_root: str | None,
    kaldi_manifest: str | None,
    finetune_data_index: str | None,
) -> None:
    config = yaml.safe_load(source.read_text())
    data = config.setdefault("data", {})
    uses_kaldi_override = kaldi_data_root is not None or kaldi_manifest is not None
    if uses_kaldi_override:
        data["backend"] = "kaldi"
        data["finetune_data_index"] = None
        data["finetune_preset_path"] = None
    if kaldi_data_root is not None:
        data["kaldi_data_root"] = kaldi_data_root
    if kaldi_manifest is not None:
        data["kaldi_manifest"] = kaldi_manifest
    if finetune_data_index is not None and not uses_kaldi_override:
        data["finetune_data_index"] = finetune_data_index
        data["finetune_preset_path"] = None
    target.write_text(yaml.safe_dump(config, sort_keys=False))


def _infer_command(
    recipe: dict[str, Any],
    config: Path,
    checkpoint_path: str,
    label_name: str,
    eval_split: str,
    *,
    batch_size: int,
    num_workers: int,
    devices: list[int] | None,
    accelerator: str,
    device: str,
    precision: str,
    seed: int,
) -> str:
    command = [
        "python",
        "-m",
        module_for_variant(str(recipe.get("variant")), "infer"),
        "--config",
        str(config),
        "--ckpt-path",
        checkpoint_path,
        "--label-name",
        label_name,
        "--eval-split",
        eval_split,
        "--batch-size",
        batch_size,
        "--num-workers",
        num_workers,
        "--accelerator",
        accelerator,
        "--device",
        device,
        "--precision",
        precision,
        "--seed",
        seed,
    ]
    if devices:
        command.extend(["--devices", *devices])
    return _render_command(command)


def _execute_logit_exports(
    recipe: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    batch_size: int,
    num_workers: int,
    devices: list[int] | None,
    accelerator: str,
    device: str,
    precision: str,
    seed: int,
    skip_test: bool,
) -> None:
    for row in rows:
        _run_logit_export(
            recipe,
            config=Path(str(row["val_config"])),
            checkpoint_path=str(row["checkpoint_path"]),
            label_name=str(row["label_name"]),
            eval_split=str(row["val_split"]),
            output_path=Path(str(row["val_logits_path"])),
            batch_size=batch_size,
            num_workers=num_workers,
            devices=devices,
            accelerator=accelerator,
            device=device,
            precision=precision,
            seed=seed,
        )
        if not skip_test:
            _run_logit_export(
                recipe,
                config=Path(str(row["test_config"])),
                checkpoint_path=str(row["checkpoint_path"]),
                label_name=str(row["label_name"]),
                eval_split=str(row["test_split"]),
                output_path=Path(str(row["test_logits_path"])),
                batch_size=batch_size,
                num_workers=num_workers,
                devices=devices,
                accelerator=accelerator,
                device=device,
                precision=precision,
                seed=seed,
            )


def _run_logit_export(
    recipe: dict[str, Any],
    *,
    config: Path,
    checkpoint_path: str,
    label_name: str,
    eval_split: str,
    output_path: Path,
    batch_size: int,
    num_workers: int,
    devices: list[int] | None,
    accelerator: str,
    device: str,
    precision: str,
    seed: int,
) -> None:
    module_name = module_for_variant(str(recipe.get("variant")), "infer")
    infer_mod = importlib.import_module(module_name)
    args = _infer_args(
        config=config,
        checkpoint_path=checkpoint_path,
        label_name=label_name,
        eval_split=eval_split,
        batch_size=batch_size,
        num_workers=num_workers,
        devices=devices or [0],
        accelerator=accelerator,
        device=device,
        precision=precision,
        seed=seed,
    )
    infer_mod.run_inference(args)
    prediction_path = Path(str(args.inference_prediction_csv_path))
    _copy_logits_csv(prediction_path, output_path)


def _infer_args(
    *,
    config: Path,
    checkpoint_path: str,
    label_name: str,
    eval_split: str,
    batch_size: int,
    num_workers: int,
    devices: list[int],
    accelerator: str,
    device: str,
    precision: str,
    seed: int,
) -> Any:
    return type(
        "InferenceArgs",
        (),
        {
            "config": config,
            "ckpt_path": checkpoint_path,
            "label_name": label_name,
            "eval_split": eval_split,
            "batch_size": int(batch_size),
            "num_workers": int(num_workers),
            "devices": [int(item) for item in devices],
            "accelerator": accelerator,
            "device": device,
            "lr": 1e-6,
            "weight_decay": 1e-5,
            "override_dataset_names": None,
            "inference_preset_path": None,
            "precision": precision,
            "avg_ckpts": 1,
            "avg_ckpt_dir": None,
            "seed": int(seed),
            "pretrained_backbone_path": None,
            "wandb": False,
            "wandb_project": None,
            "wandb_name": None,
            "wandb_entity": None,
            "wandb_group": None,
            "wandb_id": None,
            "wandb_mode": None,
        },
    )()


def _copy_logits_csv(prediction_path: Path, output_path: Path) -> None:
    if not prediction_path.exists():
        raise FileNotFoundError(f"Inference prediction CSV was not written: {prediction_path}")
    df = pd.read_csv(prediction_path)
    score_columns = ["score", "prob_1", "prob", "pred_prob", "positive_prob", "logit"]
    if _first_column(df, score_columns) is None:
        raise ValueError(f"Inference prediction CSV has no supported score column: {prediction_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(prediction_path, output_path)


def _read_binary_predictions(path: str | Path, *, label_name: str | None = None) -> dict[str, list[Any]]:
    df = pd.read_csv(path)
    label_col = _first_column(df, _prediction_label_columns(label_name))
    score_col = _first_column(df, ["score", "prob_1", "prob", "pred_prob", "positive_prob", "logit"])
    if label_col is None or score_col is None:
        raise ValueError(f"Prediction file must contain label and score columns: {path}")

    ids: list[str] = []
    y: list[int] = []
    raw: list[float] = []
    for row_index, row in df.iterrows():
        labels = _prediction_values(row[label_col])
        scores = _prediction_values(row[score_col])
        if len(labels) != len(scores):
            raise ValueError(
                f"Prediction file has mismatched label/score lengths at row {row_index}: "  # noqa: E501
                f"{label_col} has {len(labels)}, {score_col} has {len(scores)} ({path})"
            )
        ids.extend(_prediction_ids(row, row_index, len(labels)))
        y.extend(int(float(value)) for value in labels)
        raw.extend(float(value) for value in scores)

    if raw and (min(raw) < 0 or max(raw) > 1):
        raw = [1 / (1 + math.exp(-value)) for value in raw]
    return {"id": ids, "y": y, "p": raw}


def _prediction_label_columns(label_name: str | None) -> list[str]:
    columns = []
    if label_name:
        columns.append(label_name)
    columns.extend(["label", "true", "y_true", "target", "groundtruth"])
    return list(dict.fromkeys(columns))


def _prediction_values(value: Any) -> list[float]:
    if value is None:
        return []
    if isinstance(value, float) and math.isnan(value):
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() == "nan":
            return []
        if text.startswith("["):
            loaded = json.loads(text)
            if not isinstance(loaded, list):
                raise ValueError(f"Expected a JSON list prediction value, got: {value!r}")
            return [float(item) for item in loaded]
        return [float(text)]
    if isinstance(value, list):
        return [float(item) for item in value]
    return [float(value)]


def _prediction_ids(row: pd.Series, row_index: int, count: int) -> list[str]:
    sample = _prediction_sample_id(row, row_index)
    tokens = _prediction_token_ids(row, count)
    return [f"{sample}|{token}" for token in tokens]


def _prediction_sample_id(row: pd.Series, row_index: int) -> str:
    for column in ("path", "sample_id", "patient_id", "record_key", "eid"):
        if column in row and row[column] not in (None, "") and not pd.isna(row[column]):
            return f"{column}={row[column]}"
    return f"row={row_index}"


def _prediction_token_ids(row: pd.Series, count: int) -> list[str]:
    if "token_start" in row and row["token_start"] not in (None, "") and not pd.isna(row["token_start"]):
        start = int(float(row["token_start"]))
        return [f"token_start={start + offset}" for offset in range(count)]
    if "token_starts" in row and row["token_starts"] not in (None, "") and not pd.isna(row["token_starts"]):
        starts = _prediction_values(row["token_starts"])
        if len(starts) == count:
            return [f"token_start={int(value)}" for value in starts]
        if len(starts) == 1:
            start = int(starts[0])
            return [f"token_start={start + offset}" for offset in range(count)]
        encoded = ",".join(str(int(value)) for value in starts)
        return [f"token_starts={encoded};offset={offset}" for offset in range(count)]
    return [f"offset={offset}" for offset in range(count)]


def _ensemble_summary_row(rows: list[dict[str, str]]) -> dict[str, Any]:
    ids = [_candidate_id(row) for row in rows]
    val_sets = [
        _read_binary_predictions(
            _first_value(row, ["val_predictions_path", "val_logits_path"]),
            label_name=_first_value(row, ["label_name", "target_name", "target_label"]),
        )
        for row in rows
    ]
    y_val, p_val = _average_binary_predictions(val_sets)
    threshold = _best_f1_threshold(y_val, p_val)
    val_metrics = _binary_metrics(y_val, p_val, threshold)
    summary = {
        "ensemble_id": "+".join(ids),
        "n_models": len(rows),
        "member_checkpoint_paths": ";".join(_first_value(row, ["checkpoint_path", "ckpt_path"]) for row in rows),
        "threshold": threshold,
        **{f"val_{key}": value for key, value in val_metrics.items()},
    }
    test_paths = [_first_value(row, ["test_predictions_path", "test_logits_path"]) for row in rows]
    if all(test_paths):
        y_test, p_test = _average_binary_predictions(
            [
                _read_binary_predictions(
                    path,
                    label_name=_first_value(row, ["label_name", "target_name", "target_label"]),
                )
                for row, path in zip(rows, test_paths, strict=True)
            ]
        )
        test_metrics = _binary_metrics(y_test, p_test, threshold)
        summary.update({f"exploratory_test_{key}": value for key, value in test_metrics.items()})
    return summary


def _average_binary_predictions(
    items: list[dict[str, list[Any]]],
) -> tuple[list[int], list[float]]:
    order = list(items[0]["id"])
    y = list(items[0]["y"])
    _ensure_unique_prediction_ids(order)
    aligned_scores = [list(items[0]["p"])]
    for item in items[1:]:
        _ensure_unique_prediction_ids(item["id"])
        lookup = {sample_id: (label, score) for sample_id, label, score in zip(item["id"], item["y"], item["p"])}
        if set(lookup) != set(order):
            raise ValueError("Prediction files must contain the same sample identifiers for ensembling.")  # noqa: E501
        scores = []
        for sample_id, label in zip(order, y):
            other_label, score = lookup[sample_id]
            if other_label != label:
                raise ValueError("Prediction files must have aligned labels for ensembling.")
            scores.append(score)
        aligned_scores.append(scores)
    p = [sum(values) / len(values) for values in zip(*aligned_scores)]
    return y, p


def _ensure_unique_prediction_ids(ids: list[Any]) -> None:
    if len(ids) != len(set(ids)):
        raise ValueError("Prediction files must not contain duplicate sample identifiers for ensembling.")  # noqa: E501


def _first_column(df: pd.DataFrame, names: list[str]) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    return None


def _first_value(row: dict[str, Any], names: list[str]) -> str:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return str(value)
    return ""


def _candidate_id(row: dict[str, Any]) -> str:
    return str(
        row.get("trial_id") or row.get("version") or Path(_first_value(row, ["checkpoint_path"])).stem or "candidate"
    )


def _float_or_none(value: Any) -> float | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return score if math.isfinite(score) else None


def _best_f1_threshold(y_true: list[int], prob: list[float]) -> float:
    thresholds = sorted(set(prob))
    if not thresholds:
        return 0.5
    best = max(
        ((_binary_metrics(y_true, prob, threshold)["f1"], threshold) for threshold in thresholds),
        key=lambda x: x[0],
    )
    return float(best[1])


def _binary_metrics(y_true: list[int], prob: list[float], threshold: float) -> dict[str, float]:
    pred = [1 if value >= threshold else 0 for value in prob]
    tp = sum(1 for y, p in zip(y_true, pred) if y == 1 and p == 1)
    tn = sum(1 for y, p in zip(y_true, pred) if y == 0 and p == 0)
    fp = sum(1 for y, p in zip(y_true, pred) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(y_true, pred) if y == 1 and p == 0)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    accuracy = (tp + tn) / len(y_true) if y_true else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "auroc": _auroc(y_true, prob),
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def _auroc(y_true: list[int], prob: list[float]) -> float:
    positives = sum(y_true)
    negatives = len(y_true) - positives
    if positives == 0 or negatives == 0:
        return math.nan
    order = sorted(range(len(prob)), key=lambda idx: prob[idx])
    ranks = [0.0] * len(prob)
    index = 0
    while index < len(order):
        end = index
        while end + 1 < len(order) and prob[order[end + 1]] == prob[order[index]]:
            end += 1
        avg_rank = (index + end + 2) / 2
        for pos in range(index, end + 1):
            ranks[order[pos]] = avg_rank
        index = end + 1
    pos_rank_sum = sum(rank for rank, label in zip(ranks, y_true) if label == 1)
    return (pos_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def _read_rows(path: str | Path) -> list[dict[str, str]]:
    table = Path(path)
    if not table.exists():
        return []
    delimiter = "\t" if table.suffix == ".tsv" else ","
    with table.open(newline="") as file_obj:
        return list(csv.DictReader(file_obj, delimiter=delimiter))


def _write_rows(path: str | Path, rows: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row}) if rows else ["trial_id"]
    delimiter = "\t" if target.suffix == ".tsv" else ","
    with target.open("w", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames, delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)


def _script_lines(commands: list[str]) -> list[str]:
    return [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        f"cd {_sh(REPO_ROOT)}",
        f"export PYTHONPATH={_sh(REPO_ROOT)}${{PYTHONPATH:+:$PYTHONPATH}}",
        "",
        "# External test evaluation was explicitly unlocked.",
        *commands,
    ]


def _render_command(parts: list[Any]) -> str:
    return " ".join(_sh(part) for part in parts if part not in (None, ""))


def _sh(value: Any) -> str:
    import shlex

    return shlex.quote(str(value))


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
