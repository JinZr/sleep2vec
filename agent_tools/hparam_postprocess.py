from __future__ import annotations

import importlib
from itertools import combinations
import json
import math
from pathlib import Path
import shlex
import shutil
from types import SimpleNamespace
from typing import Any

import pandas as pd
import yaml

from . import experiment_io as exp_io, run_artifacts as artifacts
from .experiment_workspace import (
    canonical_local_experiment_root,
    experiment_root,
    managed_run_key,
    managed_run_parameters,
    read_run_manifest,
    validate_frozen_run_update,
    validate_managed_run_rows,
)
from .manifests import read_rows, write_rows, write_text
from .models import REPO_ROOT, module_for_variant
from .plan_rendering import infer_runtime_cli_args, render_command


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
    root = canonical_local_experiment_root(run_dir, Path.cwd())
    plan = artifacts.read_hparam_plan(root)
    rows, owner_plans = _selected_candidate_rows(
        read_rows(selected_csv, require_managed_identity=True),
        plan=plan,
        top_k=top_k,
        all_candidates=all_candidates,
    )
    config_dir = root / "external_eval_configs"
    config_paths = []
    checkpoint_paths = []
    for index, row in enumerate(rows, start=1):
        checkpoint_path = _first_value(row, ["checkpoint_path", "fixed_checkpoint_path", "ckpt_path"])
        if not checkpoint_path:
            raise ValueError(f"Selected row is missing checkpoint_path: {_candidate_id(row)}")
        config_paths.append(config_dir / f"{_candidate_id(row)}_{index:03d}_external.yaml")
        checkpoint_paths.append(checkpoint_path)
    manifest_path = root / "external_eval_manifest.tsv"
    script_path = root / "external_eval.sh"
    # Validate the whole output topology before a later alias can leave a partial export.
    exp_io.validate_managed_output_paths(root, [*config_paths, manifest_path, script_path])
    config_dir.mkdir(parents=True, exist_ok=True)
    commands = []
    manifest_rows = []
    for row, target_config, checkpoint_path in zip(rows, config_paths, checkpoint_paths, strict=True):
        owner_plan = owner_plans[managed_run_key(row)]
        recipe = owner_plan.get("recipe") if isinstance(owner_plan.get("recipe"), dict) else {}
        inputs = recipe.get("inputs") if isinstance(recipe.get("inputs"), dict) else {}
        runtime_defaults = recipe.get("runtime") if isinstance(recipe.get("runtime"), dict) else {}
        variant = str(recipe.get("variant"))
        source_config = Path(str(row["config"]))
        _copy_config_with_data_paths(
            source_config,
            target_config,
            kaldi_data_root=kaldi_data_root,
            kaldi_manifest=kaldi_manifest,
            finetune_data_index=finetune_data_index,
        )
        runtime = dict(runtime_defaults)
        for key, value in row.items():
            if key.startswith("runtime.") and value not in (None, ""):
                runtime[key.removeprefix("runtime.")] = value
        command = render_command(
            [
                "python",
                "-m",
                module_for_variant(variant, "infer"),
                "--config",
                str(target_config),
                "--ckpt-path",
                checkpoint_path,
                "--label-name",
                inputs.get("label_name"),
                "--eval-split",
                eval_split,
                *infer_runtime_cli_args(runtime),
            ]
        )
        execution = recipe.get("execution") if isinstance(recipe.get("execution"), dict) else {}
        run_cwd = Path(str(execution.get("workdir") or REPO_ROOT))
        command_root = shlex.quote(str(run_cwd))
        script_command = (
            f"(cd {command_root} && export PYTHONPATH={command_root}${{PYTHONPATH:+:$PYTHONPATH}} && " f"{command})"
        )
        commands.append(script_command)
        manifest_rows.append({**row, "external_config": str(target_config), "external_command": command})
    write_rows(manifest_path, manifest_rows)
    first_owner = owner_plans[managed_run_key(rows[0])]
    first_recipe = first_owner.get("recipe") if isinstance(first_owner.get("recipe"), dict) else {}
    first_execution = first_recipe.get("execution") if isinstance(first_recipe.get("execution"), dict) else {}
    run_cwd = Path(str(first_execution.get("workdir") or REPO_ROOT))
    write_text(
        script_path,
        "\n".join(_external_script_lines(commands, run_cwd)) + "\n",
        executable=True,
    )
    return script_path


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
        raise ValueError("hparam-export-logits requires --unlock-final-test unless --skip-test is used.")

    root = canonical_local_experiment_root(run_dir, Path.cwd())
    plan = artifacts.read_hparam_plan(root)
    rows, owner_plans = _selected_candidate_rows(
        read_rows(selected_csv, require_managed_identity=True),
        plan=plan,
        top_k=top_k,
        all_candidates=all_candidates,
    )
    config_dir = root / "logits_export_configs"
    output_dir = root / "logits_exports"
    manifest = root / "logits_export_manifest.tsv"
    output_paths = [manifest]
    if not execute:
        # Dry-run writes a replay script alongside the manifest.
        output_paths.append(root / "logits_export.sh")
    prepared = []
    for index, row in enumerate(rows, start=1):
        checkpoint_path = _first_value(row, ["checkpoint_path", "fixed_checkpoint_path", "ckpt_path"])
        if not checkpoint_path:
            raise ValueError(f"Selected row is missing checkpoint_path: {_candidate_id(row)}")
        owner_plan = owner_plans[managed_run_key(row)]
        recipe = owner_plan.get("recipe") if isinstance(owner_plan.get("recipe"), dict) else {}
        inputs = recipe.get("inputs") if isinstance(recipe.get("inputs"), dict) else {}
        resolved_label = label_name or inputs.get("label_name")
        if not resolved_label:
            raise ValueError(
                f"hparam-export-logits requires --label-name when the owning hparam plan has no label_name: "
                f"{row['step_id']} / {row['run_id']}"
            )
        candidate = _candidate_id(row)
        paths = {
            "val_config": config_dir / f"{candidate}_{index:03d}_{val_split}.yaml",
            "val_logits_path": output_dir / f"{candidate}_{index:03d}_{val_split}_logits.csv",
        }
        if not skip_test:
            paths.update(
                {
                    "test_config": config_dir / f"{candidate}_{index:03d}_{test_split}.yaml",
                    "test_logits_path": output_dir / f"{candidate}_{index:03d}_{test_split}_logits.csv",
                }
            )
        prepared.append((row, checkpoint_path, paths, recipe, resolved_label))
        output_paths.extend(paths.values())
    # Configs, manifests, and inference targets are one managed mutation boundary.
    exp_io.validate_managed_output_paths(root, output_paths)
    config_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    for row, checkpoint_path, paths, recipe, resolved_label in prepared:
        source_config = Path(str(row["config"]))
        val_config = paths["val_config"]
        _copy_config_with_data_paths(
            source_config,
            val_config,
            kaldi_data_root=val_kaldi_data_root,
            kaldi_manifest=val_kaldi_manifest,
            finetune_data_index=val_finetune_data_index,
        )
        val_logits_path = paths["val_logits_path"]
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
            test_config = paths["test_config"]
            _copy_config_with_data_paths(
                source_config,
                test_config,
                kaldi_data_root=test_kaldi_data_root,
                kaldi_manifest=test_kaldi_manifest,
                finetune_data_index=test_finetune_data_index,
            )
            test_logits_path = paths["test_logits_path"]
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

    if execute:
        _execute_logit_exports(
            manifest_rows,
            owner_plans,
            batch_size=batch_size,
            num_workers=num_workers,
            devices=devices,
            accelerator=accelerator,
            device=device,
            precision=precision,
            seed=seed,
            skip_test=skip_test,
        )
    else:
        command = [
            "python",
            "-m",
            "agent_tools",
            "hparam-export-logits",
            "--run-dir",
            str(root),
            "--selected",
            str(Path(selected_csv).expanduser().resolve()),
            "--val-split",
            val_split,
            "--test-split",
            test_split,
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
            "--top-k",
            top_k,
            "--execute",
        ]
        if unlock_final_test:
            command.append("--unlock-final-test")
        if skip_test:
            command.append("--skip-test")
        if label_name:
            command.extend(["--label-name", label_name])
        for flag, value in (
            ("--val-kaldi-data-root", val_kaldi_data_root),
            ("--val-kaldi-manifest", val_kaldi_manifest),
            ("--val-finetune-data-index", val_finetune_data_index),
            ("--test-kaldi-data-root", test_kaldi_data_root),
            ("--test-kaldi-manifest", test_kaldi_manifest),
            ("--test-finetune-data-index", test_finetune_data_index),
        ):
            if value:
                command.extend([flag, value])
        if devices:
            command.append("--devices")
            command.extend(devices)
        if all_candidates:
            command.append("--all-candidates")
        repo_root = shlex.quote(str(REPO_ROOT))
        write_text(
            root / "logits_export.sh",
            "\n".join(
                [
                    "#!/usr/bin/env bash",
                    "set -euo pipefail",
                    "",
                    f"cd {repo_root}",
                    f"export PYTHONPATH={repo_root}${{PYTHONPATH:+:$PYTHONPATH}}",
                    "",
                    render_command(command),
                ]
            )
            + "\n",
            executable=True,
        )
    write_rows(manifest, manifest_rows)
    return manifest


def threshold_hparam_outputs(run_dir: str | Path, selected_csv: str | Path) -> Path:
    root = canonical_local_experiment_root(run_dir, Path.cwd())
    plan = artifacts.read_hparam_plan(root)
    selected_rows, _owner_plans = _selected_candidate_rows(
        read_rows(selected_csv, require_managed_identity=True), plan=plan, all_candidates=True
    )
    out = root / "threshold_summary.csv"
    exp_io.validate_managed_output_paths(root, [out])
    rows = []
    for row in selected_rows:
        val_path = _first_value(row, ["val_predictions_path", "val_logits_path"])
        test_path = _first_value(row, ["test_predictions_path", "test_logits_path"])
        if not val_path or not test_path:
            raise ValueError(
                f"Selected candidate must define validation and test predictions/logits: "
                f"{row['step_id']} / {row['run_id']}"
            )
        label_name = _first_value(row, ["label_name", "target_name", "target_label"])
        val = _read_binary_predictions(val_path, label_name=label_name)
        test = _read_binary_predictions(test_path, label_name=label_name)
        if not val["y"] or not test["y"]:
            raise ValueError(
                f"Selected candidate validation and test predictions/logits must contain samples: "
                f"{row['step_id']} / {row['run_id']}"
            )
        threshold = _best_f1_threshold(val["y"], val["p"])
        val_metrics = _binary_metrics(val["y"], val["p"], threshold)
        test_metrics = _binary_metrics(test["y"], test["p"], threshold)
        rows.append(
            {
                **{
                    field: row.get(field, "")
                    for field in ("experiment_id", "step_id", "run_id")
                    if row.get(field) not in (None, "")
                },
                "threshold": threshold,
                **{f"val_{key}": value for key, value in val_metrics.items()},
                **{f"test_{key}": value for key, value in test_metrics.items()},
            }
        )
    write_rows(out, rows)
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
    root = canonical_local_experiment_root(run_dir, Path.cwd())
    plan = artifacts.read_hparam_plan(root)
    rows, _owner_plans = _selected_candidate_rows(
        read_rows(candidates_csv, require_managed_identity=True), plan=plan, all_candidates=True
    )
    out = root / "ensemble_summary.csv"
    exp_io.validate_managed_output_paths(root, [out])
    usable = [row for row in rows if _first_value(row, ["val_predictions_path", "val_logits_path"])]
    summary = []
    if usable and search_combinations:
        largest = min(max_size or len(usable), len(usable))
        for size in range(1, largest + 1):
            for combo in combinations(usable, size):
                summary.append(_ensemble_summary_row(list(combo)))
        reverse = mode == "max"
        summary = artifacts.assign_ranks(summary, key=metric, reverse=reverse, top_k=top_k, rank_metric=metric)
    elif usable:
        summary.append(_ensemble_summary_row(usable))
    write_rows(out, summary)
    return out


def _selected_candidate_rows(
    rows: list[dict[str, str]],
    *,
    plan: dict[str, Any],
    top_k: int = 1,
    all_candidates: bool = False,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    validate_managed_run_rows(rows, source="selected candidates", cardinality="many_per_run")
    recipe = plan.get("recipe") if isinstance(plan.get("recipe"), dict) else {}
    evaluation = recipe.get("evaluation_policy") if isinstance(recipe.get("evaluation_policy"), dict) else {}
    selection_metric = evaluation.get("selection_metric")
    selection_mode = evaluation.get("selection_mode")
    workspace = experiment_root(recipe)
    if workspace is None:
        raise ValueError("Selected candidates require a managed experiment workspace.")
    step = recipe.get("step") if isinstance(recipe.get("step"), dict) else {}
    step_id = str(step.get("id") or "")
    workspace_by_key = {managed_run_key(run): run for run in read_run_manifest(workspace)}

    owner_runs_by_key = {}
    owner_plans_by_key = {}
    for registered_root, owner_plan in artifacts.iter_registered_hparam_plans(
        workspace, step_id, selection_metric=selection_metric, selection_mode=selection_mode
    ):
        for run in owner_plan["runs"]:
            key = managed_run_key(run)
            if key in owner_runs_by_key:
                raise ValueError(f"Managed run is owned by multiple registered hparam plans: {key[0]} / {key[1]}")
            owner_runs_by_key[key] = run
            owner_plans_by_key[key] = owner_plan

    for row in rows:
        key = managed_run_key(row)
        managed = workspace_by_key.get(key)
        if managed is None:
            if str(row.get("step_id") or "") == step_id:
                raise ValueError(
                    f"Selected candidate is not managed by a registered hparam plan for the current step: "
                    f"{row['step_id']} / {row['run_id']}"
                )
            raise ValueError(
                f"Selected candidate is not managed by the experiment workspace: {row['step_id']} / {row['run_id']}"
            )
        candidate_parameters = managed_run_parameters(row)
        ownership_evidence = (
            {field: value for field, value in row.items() if field not in candidate_parameters}
            if key in owner_runs_by_key
            else row
        )
        # A selected checkpoint is external evidence and must belong to its frozen managed run.
        validate_frozen_run_update(managed, ownership_evidence, require_checkpoint_ownership=True)
    rows = [row for row in rows if str(row.get("step_id") or "") == step_id]
    if not rows:
        raise ValueError(f"No selected candidates match the current hparam step: {step_id}")
    managed_rows = []
    for row in rows:
        key = managed_run_key(row)
        run = owner_runs_by_key.get(key)
        if run is None:
            raise ValueError(
                f"Selected candidate is not managed by a registered hparam plan for the current step: "
                f"{key[0]} / {key[1]}"
            )
        candidate_parameters = managed_run_parameters(row)
        plan_parameters = managed_run_parameters(run)
        extra_parameters = sorted(set(candidate_parameters) - set(plan_parameters))
        if extra_parameters:
            raise ValueError(
                f"Selected candidate defines parameters outside the managed plan: {', '.join(extra_parameters)}"
            )
        for field, value in candidate_parameters.items():
            expected = "" if plan_parameters[field] is None else str(plan_parameters[field])
            actual = "" if value is None else str(value)
            if actual != expected:
                raise ValueError(f"Selected candidate parameter differs from the managed plan: {field}")
        derived = {field: value for field, value in row.items() if field not in candidate_parameters}
        validate_frozen_run_update(run, derived, require_checkpoint_ownership=True)
        managed_rows.append({**derived, **run})
    if not all_candidates and (type(top_k) is not int or top_k <= 0):
        raise ValueError("top_k must be a positive integer.")
    ranked_rows = []
    for index, row in enumerate(managed_rows):
        rank = row.get("rank")
        try:
            numeric_rank = float(rank)
        except (TypeError, ValueError):
            numeric_rank = math.nan
        if (
            isinstance(rank, bool)
            or not math.isfinite(numeric_rank)
            or not numeric_rank.is_integer()
            or numeric_rank <= 0
        ):
            raise ValueError(f"Selected candidate rank must be a positive integer: {row['step_id']} / {row['run_id']}")
        ranked_rows.append((int(numeric_rank), index, row))
    if all_candidates:
        return managed_rows, owner_plans_by_key
    selected = [row for _rank, _index, row in sorted(ranked_rows)[:top_k]]
    if not selected:
        raise ValueError("No selected candidates remain after rank/top_k filtering.")
    return selected, owner_plans_by_key


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
    variant = str(recipe.get("variant"))
    command = [
        "python",
        "-m",
        module_for_variant(variant, "infer"),
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
    return render_command(command)


def _execute_logit_exports(
    rows: list[dict[str, Any]],
    owner_plans: dict[tuple[str, str], dict[str, Any]],
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
    splits = ["val"] if skip_test else ["val", "test"]
    for row in rows:
        owner_plan = owner_plans[managed_run_key(row)]
        recipe = owner_plan.get("recipe") if isinstance(owner_plan.get("recipe"), dict) else {}
        for split in splits:
            _run_logit_export(
                recipe,
                config=Path(str(row[f"{split}_config"])),
                checkpoint_path=str(row["checkpoint_path"]),
                label_name=str(row["label_name"]),
                eval_split=str(row[f"{split}_split"]),
                output_path=Path(str(row[f"{split}_logits_path"])),
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
    variant = str(recipe.get("variant"))
    infer_mod = importlib.import_module(module_for_variant(variant, "infer"))
    args = SimpleNamespace(
        config=config,
        ckpt_path=checkpoint_path,
        label_name=label_name,
        eval_split=eval_split,
        batch_size=int(batch_size),
        num_workers=int(num_workers),
        devices=[int(item) for item in (devices or [0])],
        accelerator=accelerator,
        device=device,
        lr=1e-6,
        weight_decay=1e-5,
        override_dataset_names=None,
        inference_preset_path=None,
        precision=precision,
        avg_ckpts=1,
        avg_ckpt_dir=None,
        seed=int(seed),
        pretrained_backbone_path=None,
        wandb=False,
        wandb_project=None,
        wandb_name=None,
        wandb_entity=None,
        wandb_group=None,
        wandb_id=None,
        wandb_mode=None,
    )
    infer_mod.run_inference(args)
    _copy_logits_csv(Path(str(args.inference_prediction_csv_path)), output_path)


def _copy_logits_csv(prediction_path: Path, output_path: Path) -> None:
    if not prediction_path.exists():
        raise FileNotFoundError(f"Inference prediction CSV was not written: {prediction_path}")
    df = pd.read_csv(prediction_path)
    if _first_column(df, ["score", "prob_1", "prob", "pred_prob", "positive_prob", "logit"]) is None:
        raise ValueError(f"Inference prediction CSV has no supported score column: {prediction_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(prediction_path, output_path)


def _read_binary_predictions(path: str | Path, *, label_name: str | None = None) -> dict[str, list[Any]]:
    df = pd.read_csv(path)
    label_columns = [label_name] if label_name else []
    label_columns.extend(["label", "true", "y_true", "target", "groundtruth"])
    label_col = _first_column(df, list(dict.fromkeys(label_columns)))
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
                f"Prediction file has mismatched label/score lengths at row {row_index}: "
                f"{label_col} has {len(labels)}, {score_col} has {len(scores)} ({path})"
            )
        ids.extend(_prediction_ids(row, row_index, len(labels)))
        y.extend(int(float(value)) for value in labels)
        raw.extend(float(value) for value in scores)

    if raw and (min(raw) < 0 or max(raw) > 1):
        raw = [1 / (1 + math.exp(-value)) for value in raw]
    return {"id": ids, "y": y, "p": raw}


def _prediction_values(value: Any) -> list[float]:
    if value is None or isinstance(value, float) and math.isnan(value):
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
    sample = f"row={row_index}"
    for column in ("path", "sample_id", "patient_id", "record_key", "eid"):
        if column in row and row[column] not in (None, "") and not pd.isna(row[column]):
            sample = f"{column}={row[column]}"
            break
    if "token_start" in row and row["token_start"] not in (None, "") and not pd.isna(row["token_start"]):
        start = int(float(row["token_start"]))
        tokens = [f"token_start={start + offset}" for offset in range(count)]
    elif "token_starts" in row and row["token_starts"] not in (None, "") and not pd.isna(row["token_starts"]):
        starts = _prediction_values(row["token_starts"])
        if len(starts) == count:
            tokens = [f"token_start={int(value)}" for value in starts]
        elif len(starts) == 1:
            start = int(starts[0])
            tokens = [f"token_start={start + offset}" for offset in range(count)]
        else:
            encoded = ",".join(str(int(value)) for value in starts)
            tokens = [f"token_starts={encoded};offset={offset}" for offset in range(count)]
    else:
        tokens = [f"offset={offset}" for offset in range(count)]
    return [f"{sample}|{token}" for token in tokens]


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


def _average_binary_predictions(items: list[dict[str, list[Any]]]) -> tuple[list[int], list[float]]:
    order = list(items[0]["id"])
    y = list(items[0]["y"])
    if len(order) != len(set(order)):
        raise ValueError("Prediction files must not contain duplicate sample identifiers for ensembling.")
    aligned_scores = [list(items[0]["p"])]
    for item in items[1:]:
        if len(item["id"]) != len(set(item["id"])):
            raise ValueError("Prediction files must not contain duplicate sample identifiers for ensembling.")
        lookup = {sample_id: (label, score) for sample_id, label, score in zip(item["id"], item["y"], item["p"])}
        if set(lookup) != set(order):
            raise ValueError("Prediction files must contain the same sample identifiers for ensembling.")
        scores = []
        for sample_id, label in zip(order, y):
            other_label, score = lookup[sample_id]
            if other_label != label:
                raise ValueError("Prediction files must have aligned labels for ensembling.")
            scores.append(score)
        aligned_scores.append(scores)
    return y, [sum(values) / len(values) for values in zip(*aligned_scores)]


def _first_column(df: pd.DataFrame, names: list[str]) -> str | None:
    return next((name for name in names if name in df.columns), None)


def _first_value(row: dict[str, Any], names: list[str]) -> str:
    return next((str(row[name]) for name in names if row.get(name) not in (None, "")), "")


def _candidate_id(row: dict[str, Any]) -> str:
    return str(
        row.get("run_id") or row.get("version") or Path(_first_value(row, ["checkpoint_path"])).stem or "candidate"
    )


def _best_f1_threshold(y_true: list[int], prob: list[float]) -> float:
    thresholds = sorted(set(prob))
    if not thresholds:
        return 0.5
    best = max(
        ((_binary_metrics(y_true, prob, threshold)["f1"], threshold) for threshold in thresholds),
        key=lambda item: item[0],
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


def _external_script_lines(commands: list[str], run_cwd: str | Path = REPO_ROOT) -> list[str]:
    root = shlex.quote(str(run_cwd))
    return [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        f"cd {root}",
        f"export PYTHONPATH={root}${{PYTHONPATH:+:$PYTHONPATH}}",
        "",
        "# External test evaluation was explicitly unlocked.",
        *commands,
    ]
