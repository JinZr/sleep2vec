from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time
from typing import Any

import pandas as pd
import yaml

from .manifests import write_text
from .models import module_for_variant


def launch_hparam_trials(plan_dir: str | Path, *, dry_run: bool = True) -> Path:
    run_dir = Path(plan_dir)
    plan = _read_plan(run_dir)
    recipe = plan.get("recipe") if isinstance(plan.get("recipe"), dict) else {}
    execution = recipe.get("execution") if isinstance(recipe.get("execution"), dict) else {}
    trials = plan.get("trials") if isinstance(plan.get("trials"), list) else []
    log_dir = _local_dir(run_dir, execution.get("log_dir"), "logs")
    pid_dir = _local_dir(run_dir, execution.get("pid_dir"), "pids")
    log_dir.mkdir(parents=True, exist_ok=True)
    pid_dir.mkdir(parents=True, exist_ok=True)
    max_concurrent = int(execution.get("max_concurrent") or len(trials) or 1)
    rows = []
    for index, trial in enumerate(trials):
        trial_id = str(trial["trial_id"])
        script = run_dir / str(trial["script"])
        gpus = _assigned_gpus(recipe, index)
        log_path = log_dir / f"{trial_id}.log"
        pid_path = pid_dir / f"{trial_id}.pid"
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


def monitor_hparam_trials(run_dir: str | Path, *, once: bool = True) -> Path:
    root = Path(run_dir)
    manifest = _read_rows(root / "launch_manifest.tsv")
    rows = [_status_row(root, row) for row in manifest]
    out = root / "trial_status.tsv"
    _write_rows(out, rows)
    if not once:
        print(f"wrote {out}")
    return out


def stop_hparam_trial(run_dir: str | Path, trial_id: str) -> Path:
    root = Path(run_dir)
    rows = _read_rows(root / "launch_manifest.tsv")
    matched = [row for row in rows if row.get("trial_id") == trial_id]
    if not matched:
        raise ValueError(f"Unknown trial_id: {trial_id}")
    row = matched[0]
    pid = _read_pid(row.get("pid_path"))
    if pid is None:
        raise ValueError(f"No recorded PID for trial_id: {trial_id}")
    if row.get("target") == "ssh":
        subprocess.run(["ssh", row["host"], f"kill -TERM {pid}"], check=False)
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
    ranked = sorted(rows, key=lambda row: _sortable_score(row.get("score"), reverse), reverse=reverse)
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
) -> Path:
    if not unlock_final_test:
        raise ValueError("hparam-external-eval requires --unlock-final-test.")
    root = Path(run_dir)
    plan = _read_plan(root)
    recipe = plan.get("recipe") if isinstance(plan.get("recipe"), dict) else {}
    base_recipe = recipe.get("_base_recipe") if isinstance(recipe.get("_base_recipe"), dict) else {}
    base_inputs = base_recipe.get("inputs") if isinstance(base_recipe.get("inputs"), dict) else {}
    rows = _read_rows(selected_csv)
    config_dir = root / "external_eval_configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    commands = []
    manifest_rows = []
    for row in rows:
        if str(row.get("rank", "1")) not in {"", "1"}:
            continue
        source_config = Path(str(row["config"]))
        target_config = config_dir / f"{row['trial_id']}_external.yaml"
        _copy_config_with_data_paths(
            source_config,
            target_config,
            kaldi_data_root=kaldi_data_root,
            kaldi_manifest=kaldi_manifest,
            finetune_data_index=finetune_data_index,
        )
        command = _render_command(
            [
                "python",
                "-m",
                module_for_variant(str(recipe.get("variant")), "infer"),
                "--config",
                str(target_config),
                "--ckpt-path",
                row["checkpoint_path"],
                "--label-name",
                base_inputs.get("label_name") or (recipe.get("inputs") or {}).get("label_name"),
                "--eval-split",
                eval_split,
            ]
        )
        commands.append(command)
        manifest_rows.append({**row, "external_config": str(target_config), "external_command": command})
    _write_rows(root / "external_eval_manifest.tsv", manifest_rows)
    write_text(root / "external_eval.sh", "\n".join(_script_lines(commands)) + "\n", executable=True)
    return root / "external_eval.sh"


def threshold_hparam_outputs(run_dir: str | Path, selected_csv: str | Path) -> Path:
    rows = []
    for row in _read_rows(selected_csv):
        val_path = _first_value(row, ["val_predictions_path", "val_logits_path"])
        test_path = _first_value(row, ["test_predictions_path", "test_logits_path"])
        if not val_path or not test_path:
            continue
        val = _read_binary_predictions(val_path)
        test = _read_binary_predictions(test_path)
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


def ensemble_hparam_outputs(run_dir: str | Path, candidates_csv: str | Path) -> Path:
    rows = _read_rows(candidates_csv)
    val_sets = []
    test_sets = []
    trial_ids = []
    for row in rows:
        val_path = _first_value(row, ["val_predictions_path", "val_logits_path"])
        test_path = _first_value(row, ["test_predictions_path", "test_logits_path"])
        if not val_path:
            continue
        val_sets.append(_read_binary_predictions(val_path))
        if test_path:
            test_sets.append(_read_binary_predictions(test_path))
        trial_ids.append(str(row.get("trial_id")))
    summary = []
    if val_sets:
        y_val = val_sets[0]["y"]
        p_val = [sum(items) / len(items) for items in zip(*(item["p"] for item in val_sets))]
        threshold = _best_f1_threshold(y_val, p_val)
        val_metrics = _binary_metrics(y_val, p_val, threshold)
        row = {
            "ensemble_id": "+".join(trial_ids),
            "n_models": len(val_sets),
            "threshold": threshold,
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        if test_sets:
            y_test = test_sets[0]["y"]
            p_test = [sum(items) / len(items) for items in zip(*(item["p"] for item in test_sets))]
            test_metrics = _binary_metrics(y_test, p_test, threshold)
            row.update({f"exploratory_test_{key}": value for key, value in test_metrics.items()})
        summary.append(row)
    out = Path(run_dir) / "ensemble_summary.csv"
    _write_rows(out, summary)
    return out


def _read_plan(run_dir: Path) -> dict[str, Any]:
    plan_path = run_dir / "plan.json"
    if not plan_path.exists():
        raise FileNotFoundError(f"Missing hparam plan: {plan_path}")
    return json.loads(plan_path.read_text())


def _local_dir(run_dir: Path, raw: Any, default_name: str) -> Path:
    if raw in (None, ""):
        return run_dir / default_name
    path = Path(str(raw))
    return path if path.is_absolute() else run_dir / path


def _assigned_gpus(recipe: dict[str, Any], trial_index: int) -> list[Any]:
    execution = recipe.get("execution") if isinstance(recipe.get("execution"), dict) else {}
    base = recipe.get("_base_recipe") if isinstance(recipe.get("_base_recipe"), dict) else {}
    base_runtime = base.get("runtime") if isinstance(base.get("runtime"), dict) else {}
    pool = execution.get("gpu_pool") or base_runtime.get("devices") or []
    if not isinstance(pool, list) or not pool:
        return []
    per_trial = int(execution.get("gpus_per_trial") or len(base_runtime.get("devices") or [pool[0]]))
    start = (trial_index * per_trial) % len(pool)
    return [pool[(start + offset) % len(pool)] for offset in range(per_trial)]


def _launch_command(execution: dict[str, Any], script: Path, log_path: Path, pid_path: Path, gpus: list[Any]) -> str:
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
        run = ["conda", "run", "--no-capture-output", "-n", str(execution["conda_env"]), *run]
    env_prefix = " ".join(f"{key}={_sh(value)}" for key, value in sorted(env.items()))
    run_command = " ".join(_sh(part) for part in run)
    if env_prefix:
        run_command = f"{env_prefix} {run_command}"
    workdir = execution.get("workdir") or str(script.parent)
    inner = f"cd {_sh(workdir)} && nohup {run_command} > {_sh(log_path)} 2>&1 & echo $! > {_sh(pid_path)}"
    if execution.get("target", "local") == "ssh":
        return f"ssh {_sh(execution['host'])} {_sh(inner)}"
    return inner


def _start_process(execution: dict[str, Any], command: str) -> str:
    result = subprocess.run(["bash", "-lc", command], text=True, capture_output=True)
    return "launched" if result.returncode == 0 else "launch_failed"


def _status_row(run_dir: Path, row: dict[str, Any]) -> dict[str, Any]:
    pid = _read_pid(row.get("pid_path"))
    running = _process_running(row, pid) if pid is not None else False
    status = row.get("status") or "unknown"
    if pid is None and status == "launched":
        status = "missing_pid"
    elif running:
        status = "running"
    elif status in {"launched", "running"}:
        status = "failed" if _log_has_failure(row.get("log_path")) else "finished"
    manifest = _find_run_manifest(run_dir, row.get("version", ""), {"execution": {}})
    checkpoints = _checkpoint_names(manifest)
    return {
        **row,
        "status": status,
        "pid": pid or "",
        "log_tail": _log_tail(row.get("log_path")),
        "run_manifest": str(manifest or ""),
        "checkpoints": ";".join(checkpoints),
        "monitored_at": _now(),
    }


def _read_pid(path: Any) -> int | None:
    if not path:
        return None
    pid_path = Path(str(path))
    if not pid_path.exists():
        return None
    try:
        return int(pid_path.read_text().strip())
    except ValueError:
        return None


def _process_running(row: dict[str, Any], pid: int | None) -> bool:
    if pid is None:
        return False
    if row.get("target") == "ssh" and row.get("host"):
        result = subprocess.run(["ssh", row["host"], f"ps -p {pid} -o pid="], text=True, capture_output=True)
        return result.returncode == 0 and str(pid) in result.stdout
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _log_has_failure(path: Any) -> bool:
    if not path:
        return False
    log_path = Path(str(path))
    if not log_path.exists():
        return False
    tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-100:])
    return any(marker in tail for marker in ["Traceback", "RuntimeError", "CUDA out of memory", "Error executing job"])


def _log_tail(path: Any, lines: int = 8) -> str:
    if not path:
        return ""
    log_path = Path(str(path))
    if not log_path.exists():
        return ""
    return "\n".join(log_path.read_text(errors="replace").splitlines()[-lines:])


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
        if path.name.startswith("epoch="):
            return str(path)
        epoch = manifest.get("epoch")
        if epoch is not None:
            fixed = path.with_name(f"epoch={epoch}.ckpt")
            if fixed.exists() or str(path):
                return str(fixed)
        if path.name.startswith("best-epoch="):
            fixed = path.with_name(path.name.removeprefix("best-"))
            return str(fixed)
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
    if kaldi_data_root is not None:
        data["kaldi_data_root"] = kaldi_data_root
    if kaldi_manifest is not None:
        data["kaldi_manifest"] = kaldi_manifest
    if finetune_data_index is not None:
        data["finetune_data_index"] = finetune_data_index
    target.write_text(yaml.safe_dump(config, sort_keys=False))


def _read_binary_predictions(path: str | Path) -> dict[str, list[float]]:
    df = pd.read_csv(path)
    label_col = _first_column(df, ["label", "true", "y_true", "target", "src_isDep"])
    score_col = _first_column(df, ["score", "prob", "prob_1", "pred_prob", "positive_prob", "logit"])
    if label_col is None or score_col is None:
        raise ValueError(f"Prediction file must contain label and score columns: {path}")
    y = [int(value) for value in pd.to_numeric(df[label_col], errors="coerce").fillna(0)]
    raw = [float(value) for value in pd.to_numeric(df[score_col], errors="coerce").fillna(0)]
    if raw and (min(raw) < 0 or max(raw) > 1):
        raw = [1 / (1 + math.exp(-value)) for value in raw]
    return {"y": y, "p": raw}


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


def _best_f1_threshold(y_true: list[int], prob: list[float]) -> float:
    thresholds = sorted(set(prob))
    if not thresholds:
        return 0.5
    best = max(
        ((_binary_metrics(y_true, prob, threshold)["f1"], threshold) for threshold in thresholds), key=lambda x: x[0]
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
