from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from agent_tool_test_helpers import write_finetune_recipe, write_yaml
import pytest
import yaml

from agent_tools import experiment_tracking, experiments, hparam_selection, run_artifacts, run_evidence
from agent_tools.experiment_workspace import merge_run_manifest, read_run_manifest
from agent_tools.manifests import read_rows, write_rows
from agent_tools.models import REPO_ROOT

_RUNTIME_COMMIT = subprocess.run(
    ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, text=True, capture_output=True
).stdout.strip()


def _run(*args: str) -> subprocess.CompletedProcess:
    runner = Path(__file__).with_name("agent_tools_cli_stub.py")
    return subprocess.run([sys.executable, str(runner), *args], text=True, capture_output=True)


def _hparam_recipe(
    tmp_path: Path,
    *,
    execution: dict | None = None,
    selection_metric: str = "val_ahi_pearson",
    selection_mode: str = "max",
    selection_split: str = "val",
    config_monitor: str | None = None,
    max_runs: int = 1,
) -> Path:
    base = write_finetune_recipe(tmp_path)
    base_payload = yaml.safe_load(base.read_text())
    config_path = Path(base_payload["inputs"]["config"])
    config_payload = yaml.safe_load(config_path.read_text())
    monitor_metric = config_monitor or selection_metric
    config_payload["finetune"]["task"]["monitor"] = monitor_metric
    config_payload["finetune"]["task"]["monitor_mod"] = selection_mode
    write_yaml(config_path, config_payload)
    base_payload["evaluation_policy"]["selection_metric"] = monitor_metric
    base_payload["evaluation_policy"]["selection_mode"] = selection_mode
    write_yaml(base, base_payload)
    execution_payload = dict(execution) if execution is not None else {"workdir": str(tmp_path)}
    manager_runtime = (
        str(execution_payload.get("target", "local") or "local") == "local"
        and execution_payload.get("workdir") in (None, "", str(REPO_ROOT))
        and execution_payload.get("conda_env") in (None, "")
    )
    if not manager_runtime:
        execution_payload.setdefault("python", sys.executable)
        execution_payload.setdefault("runtime_commit", _RUNTIME_COMMIT)
    return write_yaml(
        tmp_path / "tune.yaml",
        {
            "name": "unit_hparam",
            "task": "hparam_tune",
            "variant": "sleep2vec",
            "base_recipe": str(base),
            "search": {
                "method": "grid",
                "max_runs": max_runs,
                "parameters": {"runtime.lr": [float(index + 1) * 1e-6 for index in range(max_runs)]},
            },
            "execution": execution_payload,
            "evaluation_policy": {
                "selection_metric": selection_metric,
                "selection_mode": selection_mode,
                "selection_split": selection_split,
                "external_test_locked": selection_split != "test",
                "test_after_fit": selection_split == "test",
                "final_eval_split": "test",
                "final_test_unlocked": False,
                "require_manual_unlock_for_final_test": True,
            },
            "decisions": {
                "task": {"value": "hparam_tune", "source": "explicit_recipe"},
                "label_name": {"value": "ahi", "source": "explicit_recipe"},
                "external_test_locked": {
                    "value": selection_split != "test",
                    "source": "explicit_recipe",
                },
                "train_val_test_policy": {
                    "value": selection_split,
                    "source": "explicit_recipe",
                },
                "overwrite_policy": {"value": False, "source": "explicit_recipe"},
                "final_eval_unlock": {"value": False, "source": "explicit_recipe"},
            },
        },
    )


def _read_table(path: Path) -> list[dict[str, str]]:
    delimiter = "\t" if path.suffix == ".tsv" else ","
    with path.open(newline="") as file_obj:
        return list(csv.DictReader(file_obj, delimiter=delimiter))


def _first_run(plan_dir: Path) -> dict:
    plan = json.loads((plan_dir / "plan.json").read_text())
    recipe = plan["recipe"]
    workspace = Path(recipe["experiment"]["root"])
    merge_run_manifest(
        workspace,
        [
            {
                "step_id": run["step_id"],
                "run_id": run["run_id"],
                "run_name": run["run_name"],
                "status": "completed",
            }
            for run in plan["runs"]
        ],
    )
    return plan["runs"][0]


def _ranking_path(plan_dir: Path) -> Path:
    recipe = json.loads((plan_dir / "plan.json").read_text())["recipe"]
    return Path(recipe["experiment"]["root"]) / "reports" / "ranking.csv"


def _prepare_two_hparam_steps(tmp_path: Path) -> tuple[dict, Path, Path]:
    first_recipe = _hparam_recipe(tmp_path)
    first_plan = tmp_path / "plan-1"
    assert _run("plan", "--recipe", str(first_recipe), "--output-dir", str(first_plan)).returncode == 0
    first_run = _first_run(first_plan)
    first_checkpoint = Path(first_run["checkpoint_dir"]) / "epoch=1.ckpt"
    first_checkpoint.parent.mkdir(parents=True)
    first_checkpoint.write_text("checkpoint")
    (Path(first_run["runtime_dir"]) / "run_manifest.json").write_text(
        json.dumps(
            {
                "metrics": {"val_ahi_pearson": 0.9},
                "best_model_path": str(first_checkpoint),
                "epoch": 1,
            }
        )
    )
    hparam_selection.select_hparam_candidates(first_plan)

    second_recipe = _hparam_recipe(tmp_path)
    second_payload = yaml.safe_load(second_recipe.read_text())
    second_payload["step"] = {
        "id": "second-tune",
        "phase": "train",
        "purpose": "Exercise a second hparam selection step.",
    }
    write_yaml(second_recipe, second_payload)
    second_plan = tmp_path / "plan-2"
    assert _run("plan", "--recipe", str(second_recipe), "--output-dir", str(second_plan)).returncode == 0
    second_run = _first_run(second_plan)
    second_checkpoint = Path(second_run["checkpoint_dir"]) / "epoch=2.ckpt"
    second_checkpoint.parent.mkdir(parents=True)
    second_checkpoint.write_text("checkpoint")
    (Path(second_run["runtime_dir"]) / "run_manifest.json").write_text(
        json.dumps(
            {
                "metrics": {"val_ahi_pearson": 0.8},
                "best_model_path": str(second_checkpoint),
                "epoch": 2,
            }
        )
    )
    return first_run, first_plan, second_plan


def _prepare_two_test_selected_steps(tmp_path: Path) -> list[Path]:
    plans = []
    for step_id, score in (("a-tune", 0.9), ("b-tune", 0.8)):
        recipe = _hparam_recipe(
            tmp_path,
            selection_metric="test_ahi_pearson",
            selection_split="test",
            config_monitor="val_ahi_pearson",
        )
        payload = yaml.safe_load(recipe.read_text())
        payload["step"] = {"id": step_id, "phase": "train", "purpose": f"Select {step_id}."}
        write_yaml(recipe, payload)
        plan_dir = tmp_path / f"plan-{step_id}"
        assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
        run = _first_run(plan_dir)
        checkpoint = Path(run["checkpoint_dir"]) / "epoch=1.ckpt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_text(f"checkpoint-{step_id}")
        (Path(run["runtime_dir"]) / "run_manifest.json").write_text(
            json.dumps(
                {
                    "test_all_checkpoints_after_fit": True,
                    "checkpoint_test_results": [
                        {
                            "checkpoint_path": str(checkpoint),
                            "epoch": 1,
                            "metrics": {"test_ahi_pearson": score},
                        }
                    ],
                }
            )
        )
        hparam_selection.select_hparam_candidates(plan_dir)
        plans.append(plan_dir)
    return plans


def _prepare_test_selected_plan_with_two_checkpoints(tmp_path: Path) -> Path:
    recipe = _hparam_recipe(
        tmp_path,
        selection_metric="test_ahi_pearson",
        selection_split="test",
        config_monitor="val_ahi_pearson",
    )
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    run = _first_run(plan_dir)
    checkpoint_dir = Path(run["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True)
    checkpoints = [checkpoint_dir / "epoch=1.ckpt", checkpoint_dir / "epoch=2.ckpt"]
    for checkpoint in checkpoints:
        checkpoint.write_text(checkpoint.name)
    (Path(run["runtime_dir"]) / "run_manifest.json").write_text(
        json.dumps(
            {
                "test_all_checkpoints_after_fit": True,
                "checkpoint_test_results": [
                    {
                        "checkpoint_path": str(checkpoint),
                        "epoch": epoch,
                        "metrics": {"test_ahi_pearson": score},
                    }
                    for checkpoint, epoch, score in zip(checkpoints, (1, 2), (0.9, 0.8))
                ],
            }
        )
    )
    hparam_selection.select_hparam_candidates(plan_dir)
    return plan_dir


def test_hparam_select_uses_fixed_epoch_checkpoint_not_best_alias(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    run = _first_run(plan_dir)
    version = run["version"]
    run_dir = Path(run["runtime_dir"])
    ckpt_dir = Path(run["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "epoch=11.ckpt").write_text("fixed")
    (ckpt_dir / "best-epoch=11.ckpt").write_text("alias")
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "version": version,
                "monitor": "val_ahi_pearson",
                "best_model_score": 0.71,
                "best_model_path": str(ckpt_dir / "best-epoch=11.ckpt"),
                "epoch": 11,
                "metrics": {"val_ahi_pearson": 0.71},
            }
        )
    )

    result = _run(
        "hparam-select",
        "--run-dir",
        str(plan_dir),
        "--metric",
        "val_ahi_pearson",
        "--mode",
        "max",
    )

    assert result.returncode == 0, result.stderr
    rows = _read_table(_ranking_path(plan_dir))
    assert rows[0]["checkpoint_path"].endswith("epoch=11.ckpt")
    assert "best-epoch" not in rows[0]["checkpoint_path"]
    selected = next(
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text().splitlines()
        if json.loads(line)["event_type"] == "candidate_selected"
    )
    assert selected["step_id"] == "unit-hparam-tune"
    assert selected["selected_run_id"] == "run-000"
    selection_report = tmp_path / "reports" / "hparam_selection.md"
    report_text = selection_report.read_text()
    canonical = read_run_manifest(tmp_path)[0]
    assert "Selection metric: `val_ahi_pearson`" in report_text
    assert "Selection split: `val`" in report_text
    assert "Evaluated candidates: `1/1`" in report_text
    assert "Winner run: `run-000`" in report_text
    assert "Search overrides:" in report_text
    assert f"Frozen config: `{canonical['config']}`" in report_text
    assert f"Frozen config SHA-256: `{canonical['config_sha256']}`" in report_text
    assert f"Frozen script: `{canonical['script']}`" in report_text
    assert f"Frozen script SHA-256: `{canonical['script_sha256']}`" in report_text
    assert "global optimum" in report_text
    assert selected["selection_report"] == str(selection_report)
    assert selected["selection_report_sha256"] == hashlib.sha256(report_text.encode()).hexdigest()
    assert canonical["selection_mode"] == "max"
    assert canonical["selection_split"] == "val"
    assert canonical["selection_report"] == str(selection_report)
    assert canonical["selection_report_sha256"] == selected["selection_report_sha256"]


def test_hparam_select_rejects_completed_experiment_without_mutation(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    run = _first_run(plan_dir)
    checkpoint = Path(run["checkpoint_dir"]) / "epoch=1.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text("checkpoint")
    runtime_manifest = Path(run["runtime_dir"]) / "run_manifest.json"
    runtime_manifest.write_text(
        json.dumps(
            {
                "metrics": {"val_ahi_pearson": 0.8},
                "best_model_path": str(checkpoint),
                "epoch": 1,
            }
        )
    )
    hparam_selection.select_hparam_candidates(plan_dir)
    selection_report = tmp_path / "reports" / "hparam_selection.md"
    experiments.finalize_experiment(tmp_path, selection_report)
    runtime_manifest.write_text(
        json.dumps(
            {
                "metrics": {"val_ahi_pearson": 0.9},
                "best_model_path": str(checkpoint),
                "epoch": 1,
            }
        )
    )
    before = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}

    with pytest.raises(ValueError, match="completed"):
        hparam_selection.select_hparam_candidates(plan_dir)

    assert {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before


def test_hparam_select_clears_stale_result_evidence_from_unscored_candidates(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path, max_runs=2)
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    first_run = _first_run(plan_dir)
    runs = json.loads((plan_dir / "plan.json").read_text())["runs"]
    checkpoint = Path(first_run["checkpoint_dir"]) / "epoch=1.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text("checkpoint")
    Path(first_run["runtime_dir"], "run_manifest.json").write_text(
        json.dumps(
            {
                "metrics": {"val_ahi_pearson": 0.8},
                "best_model_path": str(checkpoint),
                "epoch": 1,
            }
        )
    )
    losing_run = runs[1]
    failed_manifest = Path(losing_run["runtime_dir"]) / "run_manifest.json"
    failed_manifest.parent.mkdir(parents=True)
    failed_manifest.write_text(json.dumps({"health_status": "failed"}))
    merge_run_manifest(
        tmp_path,
        [
            {
                "step_id": losing_run["step_id"],
                "run_id": losing_run["run_id"],
                "status": "failed",
                "score": "0.7",
                "rank": "2",
                "checkpoint_path": str(Path(losing_run["checkpoint_dir"]) / "epoch=2.ckpt"),
                "checkpoint_sha256": "b" * 64,
                "run_manifest": str(failed_manifest),
                "epoch": "2",
                "checkpoint_rank": "2",
                "source": "stale-selection",
            }
        ],
    )

    hparam_selection.select_hparam_candidates(plan_dir)

    canonical = next(row for row in read_run_manifest(tmp_path) if row["run_id"] == losing_run["run_id"])
    assert canonical["selection_task"] == "hparam_tune"
    assert all(canonical.get(field) in (None, "") for field in hparam_selection.tracking.HPARAM_SELECTION_RESULT_FIELDS)
    assert canonical["run_manifest"] == str(failed_manifest)


@pytest.mark.parametrize(("field", "value"), [("status", "running"), ("unexpected_terminal", "value")])
def test_hparam_select_rejects_invalid_active_experiment_owner_without_mutation(tmp_path: Path, field: str, value: str):
    recipe = _hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    manifest_path = tmp_path / "experiment.yaml"
    manifest = yaml.safe_load(manifest_path.read_text())
    manifest["experiment"][field] = value
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False))
    before = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}

    with pytest.raises(ValueError, match="Invalid active experiment owner"):
        hparam_selection.select_hparam_candidates(plan_dir)

    assert {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before


@pytest.mark.parametrize("alias_kind", ["symlink", "hardlink"])
def test_hparam_select_rejects_aliased_active_experiment_owner_before_writing(tmp_path: Path, alias_kind: str):
    recipe = _hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    run = _first_run(plan_dir)
    checkpoint = Path(run["checkpoint_dir"]) / "epoch=1.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text("checkpoint")
    (Path(run["runtime_dir"]) / "run_manifest.json").write_text(
        json.dumps(
            {
                "metrics": {"val_ahi_pearson": 0.8},
                "best_model_path": str(checkpoint),
                "epoch": 1,
            }
        )
    )
    manifest_path = tmp_path / "experiment.yaml"
    outside = tmp_path.parent / f"{tmp_path.name}-foreign-experiment.yaml"
    outside.write_bytes(manifest_path.read_bytes())
    manifest_path.unlink()
    if alias_kind == "symlink":
        manifest_path.symlink_to(outside)
    else:
        manifest_path.hardlink_to(outside)
    before = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    outside_before = outside.read_bytes()

    with pytest.raises(ValueError, match="missing or aliased"):
        hparam_selection.select_hparam_candidates(plan_dir)

    assert {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before
    assert outside.read_bytes() == outside_before
    if alias_kind == "symlink":
        assert manifest_path.is_symlink()
    else:
        assert manifest_path.stat().st_ino == outside.stat().st_ino


@pytest.mark.parametrize("ranking_present", [True, False])
def test_hparam_select_preserves_frozen_val_selection_when_runtime_evidence_drifts(
    tmp_path: Path, ranking_present: bool
):
    recipe = _hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    run = _first_run(plan_dir)
    checkpoint = Path(run["checkpoint_dir"]) / "epoch=1.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text("checkpoint")
    runtime_manifest = Path(run["runtime_dir"]) / "run_manifest.json"
    runtime_manifest.write_text(
        json.dumps(
            {
                "metrics": {"val_ahi_pearson": 0.8},
                "best_model_path": str(checkpoint),
                "epoch": 1,
            }
        )
    )
    hparam_selection.select_hparam_candidates(plan_dir)
    merge_run_manifest(
        tmp_path,
        [{"step_id": run["step_id"], "run_id": run["run_id"], "epoch": "1"}],
    )
    hparam_selection.select_hparam_candidates(plan_dir)
    if not ranking_present:
        _ranking_path(plan_dir).unlink()
    runtime_manifest.write_text(
        json.dumps(
            {
                "metrics": {"val_ahi_pearson": 0.9},
                "best_model_path": str(checkpoint),
                "epoch": 1,
            }
        )
    )
    before = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}

    with pytest.raises(ValueError, match="Frozen canonical hparam selection"):
        hparam_selection.select_hparam_candidates(plan_dir)

    assert {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before


def test_test_selected_plan_exports_frozen_checkpoint_path_spelling(tmp_path: Path):
    (tmp_path / "lexical").mkdir()
    workdir = tmp_path / "lexical" / ".."
    recipe = _hparam_recipe(
        tmp_path,
        execution={"workdir": str(workdir)},
        selection_metric="test_ahi_pearson",
        selection_split="test",
        config_monitor="val_ahi_pearson",
    )
    plan_dir = tmp_path / "plan"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir))

    assert result.returncode == 0, result.stderr or result.stdout
    run = _first_run(plan_dir)
    expected_checkpoint_dir = workdir / "log-finetune" / run["version"] / "checkpoints"
    assert run["checkpoint_dir"] == str(expected_checkpoint_dir)
    assert f"export _SLEEP2VEC_FROZEN_CHECKPOINT_DIR={expected_checkpoint_dir}" in Path(run["script"]).read_text()


def test_hparam_select_globally_ranks_every_saved_checkpoint_by_test_metric(tmp_path: Path):
    recipe = _hparam_recipe(
        tmp_path,
        selection_metric="test_ahi_pearson",
        selection_split="test",
        config_monitor="val_ahi_pearson",
        max_runs=2,
    )
    plan_dir = tmp_path / "plan"
    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir))
    assert result.returncode == 0, result.stderr or result.stdout
    plan = json.loads((plan_dir / "plan.json").read_text())

    checkpoint_scores = ((0.91, 0.75), (0.79, 0.75))
    for run, val_score, test_scores in zip(plan["runs"], (0.81, 0.72), checkpoint_scores):
        runtime_dir = Path(run["runtime_dir"])
        checkpoint_dir = Path(run["checkpoint_dir"])
        checkpoint_dir.mkdir(parents=True)
        checkpoints = [checkpoint_dir / "epoch=00.ckpt", checkpoint_dir / "epoch=3.ckpt"]
        for checkpoint in checkpoints:
            checkpoint.write_text(f"{run['run_id']}:{checkpoint.name}")
        (checkpoint_dir / "best-epoch=3.ckpt").write_text("mutable best alias")
        (checkpoint_dir / "last.ckpt").write_text("mutable last alias")
        (runtime_dir / "run_manifest.json").write_text(
            json.dumps(
                {
                    "monitor": "val_ahi_pearson",
                    "monitor_mode": "max",
                    "best_model_score": val_score,
                    "best_model_path": str(checkpoints[1]),
                    "epoch": 3,
                    "metrics": {
                        "val_ahi_pearson": val_score,
                        "test_ahi_pearson": test_scores[1],
                    },
                    "test_all_checkpoints_after_fit": True,
                    "checkpoint_test_results": [
                        {
                            "checkpoint_path": str(checkpoint),
                            "epoch": epoch,
                            "metrics": {"test_ahi_pearson": score},
                        }
                        for checkpoint, epoch, score in zip(checkpoints, (0, 3), test_scores)
                    ],
                }
            )
        )
    merge_run_manifest(
        tmp_path,
        [{"step_id": run["step_id"], "run_id": run["run_id"], "status": "completed"} for run in plan["runs"]],
    )

    ranking = hparam_selection.select_hparam_candidates(plan_dir)

    rows = _read_table(ranking)
    assert [(row["run_id"], row["epoch"], row["score"]) for row in rows] == [
        ("run-000", "0", "0.91"),
        ("run-001", "0", "0.79"),
    ]
    checkpoint_rows = _read_table(plan_dir / "checkpoint_test_ranking.csv")
    assert [(row["run_id"], row["epoch"], row["score"]) for row in checkpoint_rows] == [
        ("run-000", "0", "0.91"),
        ("run-001", "0", "0.79"),
        ("run-000", "3", "0.75"),
        ("run-001", "3", "0.75"),
    ]
    assert all(len(row["checkpoint_sha256"]) == 64 for row in checkpoint_rows)
    assert all("best-epoch" not in row["checkpoint_path"] for row in checkpoint_rows)
    assert all(not row["checkpoint_path"].endswith("last.ckpt") for row in checkpoint_rows)
    selected = next(
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text().splitlines()
        if json.loads(line)["event_type"] == "candidate_selected"
    )
    assert selected["selected_run_id"] == "run-000"
    assert selected["selected_checkpoint_path"] == str(Path(plan["runs"][0]["checkpoint_dir"]) / "epoch=00.ckpt")
    assert len(selected["selected_checkpoint_sha256"]) == 64
    canonical = {row["run_id"]: row for row in read_run_manifest(tmp_path)}
    for row in rows:
        assert {
            field: canonical[row["run_id"]][field]
            for field in ("metric", "score", "epoch", "checkpoint_rank", "source", "run_manifest", "status")
        } == {
            field: row[field]
            for field in ("metric", "score", "epoch", "checkpoint_rank", "source", "run_manifest", "status")
        }

    hparam_selection.select_hparam_candidates(plan_dir)

    selections = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text().splitlines()
        if json.loads(line)["event_type"] == "candidate_selected"
    ]
    assert len(selections) == 1

    ranking_before = ranking.read_bytes()
    checkpoint_ranking_before = (plan_dir / "checkpoint_test_ranking.csv").read_bytes()
    Path(selected["selected_checkpoint_path"]).write_text("drifted checkpoint")
    with pytest.raises(ValueError, match="Frozen checkpoint SHA-256 differs"):
        hparam_selection.select_hparam_candidates(plan_dir)
    assert ranking.read_bytes() == ranking_before
    assert (plan_dir / "checkpoint_test_ranking.csv").read_bytes() == checkpoint_ranking_before


@pytest.mark.parametrize("field", ["checkpoint_rank", "epoch", "run_manifest", "source", "status"])
@pytest.mark.parametrize("mutation", ["tamper", "remove"])
def test_experiment_status_rejects_test_ranking_optional_provenance_drift(tmp_path: Path, field: str, mutation: str):
    recipe = _hparam_recipe(
        tmp_path,
        selection_metric="test_ahi_pearson",
        selection_split="test",
        config_monitor="val_ahi_pearson",
    )
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    run = _first_run(plan_dir)
    checkpoint = Path(run["checkpoint_dir"]) / "epoch=1.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text("checkpoint")
    (Path(run["runtime_dir"]) / "run_manifest.json").write_text(
        json.dumps(
            {
                "test_all_checkpoints_after_fit": True,
                "checkpoint_test_results": [
                    {
                        "checkpoint_path": str(checkpoint),
                        "epoch": 1,
                        "metrics": {"test_ahi_pearson": 0.8},
                    }
                ],
            }
        )
    )
    ranking = hparam_selection.select_hparam_candidates(plan_dir)
    selection_report = tmp_path / "reports" / "hparam_selection.md"
    rows = read_rows(ranking, require_managed_identity=True)
    if mutation == "remove":
        rows[0].pop(field)
    else:
        rows[0][field] = "tampered"
    write_rows(ranking, rows)

    snapshot = experiments.experiment_status(tmp_path)

    assert snapshot["summary"]["state"] == "ready_to_report"
    with pytest.raises(ValueError, match="selection report is missing or differs"):
        experiments.finalize_experiment(tmp_path, selection_report)


@pytest.mark.parametrize("field", ["run_manifest", "status"])
@pytest.mark.parametrize("mutation", ["tamper", "remove"])
def test_experiment_status_rejects_val_ranking_optional_provenance_drift(tmp_path: Path, field: str, mutation: str):
    recipe = _hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    run = _first_run(plan_dir)
    checkpoint = Path(run["checkpoint_dir"]) / "epoch=1.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text("checkpoint")
    (Path(run["runtime_dir"]) / "run_manifest.json").write_text(
        json.dumps(
            {
                "metrics": {"val_ahi_pearson": 0.8},
                "best_model_path": str(checkpoint),
                "epoch": 1,
            }
        )
    )
    ranking = hparam_selection.select_hparam_candidates(plan_dir)
    selection_report = tmp_path / "reports" / "hparam_selection.md"
    rows = read_rows(ranking, require_managed_identity=True)
    if mutation == "remove":
        rows[0].pop(field)
    else:
        rows[0][field] = "tampered"
    write_rows(ranking, rows)

    assert experiments.experiment_status(tmp_path)["summary"]["state"] == "ready_to_report"
    with pytest.raises(ValueError, match="selection report is missing or differs"):
        experiments.finalize_experiment(tmp_path, selection_report)


def test_experiment_status_rejects_val_ranking_with_unowned_optional_column(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    run = _first_run(plan_dir)
    checkpoint = Path(run["checkpoint_dir"]) / "epoch=1.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text("checkpoint")
    (Path(run["runtime_dir"]) / "run_manifest.json").write_text(
        json.dumps(
            {
                "metrics": {"val_ahi_pearson": 0.8},
                "best_model_path": str(checkpoint),
                "epoch": 1,
            }
        )
    )
    ranking = hparam_selection.select_hparam_candidates(plan_dir)
    selection_report = tmp_path / "reports" / "hparam_selection.md"
    rows = read_rows(ranking, require_managed_identity=True)
    rows[0]["source"] = "forged_source"
    write_rows(ranking, rows)

    assert experiments.experiment_status(tmp_path)["summary"]["state"] == "ready_to_report"
    with pytest.raises(ValueError, match="selection report is missing or differs"):
        experiments.finalize_experiment(tmp_path, selection_report)


def test_checkpoint_hash_uses_ssh_execution_target(monkeypatch):
    checkpoint = "/remote/runtime/checkpoints/epoch=1.ckpt"
    digest = "a" * 64
    calls = []

    def fake_run(row, command):
        calls.append((row, command))
        return subprocess.CompletedProcess(command, 0, digest, "")

    monkeypatch.setattr(run_evidence, "run_row_command", fake_run)

    observed = run_evidence.checkpoint_file_sha256(
        {"target": "ssh", "host": "unit-host"},
        checkpoint,
    )

    assert observed == digest
    assert calls[0][0] == {"target": "ssh", "host": "unit-host"}
    assert checkpoint in calls[0][1]
    assert "hashlib.sha256" in calls[0][1]


@pytest.mark.parametrize("selection_split", ["val", "test"])
def test_hparam_select_uses_ssh_manifest_inventory_and_hash_evidence(
    tmp_path: Path,
    monkeypatch,
    selection_split: str,
):
    metric = "test_ahi_pearson" if selection_split == "test" else "val_ahi_pearson"
    recipe = _hparam_recipe(
        tmp_path,
        execution={
            "target": "ssh",
            "host": "unit-host",
            "workdir": "/remote/repository",
            "path_context": "remote",
            "path_validation": "defer",
        },
        selection_metric=metric,
        selection_split=selection_split,
        config_monitor="val_ahi_pearson",
    )
    plan_dir = tmp_path / "plan"
    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir))
    assert result.returncode == 0, result.stderr or result.stdout
    run = _first_run(plan_dir)
    checkpoint = str(Path(run["checkpoint_dir"]) / "epoch=1.ckpt")
    manifest = {
        "epoch": 1,
        "best_model_path": str(Path(run["checkpoint_dir"]) / "best-epoch=1.ckpt"),
        "metrics": {metric: 0.8},
    }
    if selection_split == "test":
        manifest.update(
            {
                "test_all_checkpoints_after_fit": True,
                "checkpoint_test_results": [
                    {
                        "checkpoint_path": checkpoint,
                        "epoch": 1,
                        "metrics": {metric: 0.8},
                    }
                ],
            }
        )
    runtime_calls = []
    hash_calls = []

    def fake_runtime_artifacts(row):
        runtime_calls.append(row)
        return str(Path(run["runtime_dir"]) / "run_manifest.json"), manifest, ["epoch=1.ckpt"]

    def fake_checkpoint_hash(row, path):
        hash_calls.append((row, path))
        return "b" * 64

    monkeypatch.setattr(hparam_selection.evidence, "runtime_artifacts", fake_runtime_artifacts)
    monkeypatch.setattr(hparam_selection.evidence, "checkpoint_file_sha256", fake_checkpoint_hash)
    monkeypatch.setattr(hparam_selection.tracking, "validate_checkpoint_evidence_rows", lambda *_args: None)
    merge_run_manifest(
        tmp_path,
        [{"step_id": run["step_id"], "run_id": run["run_id"], "status": "completed"}],
    )

    ranking = hparam_selection.select_hparam_candidates(plan_dir)

    row = _read_table(ranking)[0]
    assert row["score"] == "0.8"
    assert row["checkpoint_path"] == checkpoint
    assert row["run_manifest"] == str(Path(run["runtime_dir"]) / "run_manifest.json")
    assert runtime_calls and runtime_calls[0]["target"] == "ssh"
    assert runtime_calls[0]["host"] == "unit-host"
    assert row["checkpoint_sha256"] == "b" * 64
    assert [(call[0]["target"], call[0]["host"], call[1]) for call in hash_calls] == [("ssh", "unit-host", checkpoint)]


@pytest.mark.parametrize("selection_split", ["val", "test"])
def test_hparam_select_fails_before_partial_ranking_when_successful_ssh_evidence_is_unavailable(
    tmp_path: Path,
    monkeypatch,
    selection_split: str,
):
    metric = "test_ahi_pearson" if selection_split == "test" else "val_ahi_pearson"
    recipe = _hparam_recipe(
        tmp_path,
        execution={
            "target": "ssh",
            "host": "unit-host",
            "workdir": "/remote/repository",
            "path_context": "remote",
            "path_validation": "defer",
        },
        selection_metric=metric,
        selection_split=selection_split,
        config_monitor="val_ahi_pearson",
        max_runs=2,
    )
    plan_dir = tmp_path / "plan"
    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir))
    assert result.returncode == 0, result.stderr or result.stdout
    runs = json.loads((plan_dir / "plan.json").read_text())["runs"]
    checkpoint = str(Path(runs[0]["checkpoint_dir"]) / "epoch=1.ckpt")
    manifest = {
        "epoch": 1,
        "best_model_path": str(Path(runs[0]["checkpoint_dir"]) / "best-epoch=1.ckpt"),
        "metrics": {metric: 0.8},
    }
    if selection_split == "test":
        manifest.update(
            {
                "test_all_checkpoints_after_fit": True,
                "checkpoint_test_results": [
                    {
                        "checkpoint_path": checkpoint,
                        "epoch": 1,
                        "metrics": {metric: 0.8},
                    }
                ],
            }
        )

    def fake_runtime_artifacts(row):
        assert (row["target"], row["host"]) == ("ssh", "unit-host")
        if row["run_id"] == runs[0]["run_id"]:
            return str(Path(row["runtime_dir"]) / "run_manifest.json"), manifest, ["epoch=1.ckpt"]
        return None

    monkeypatch.setattr(hparam_selection.evidence, "runtime_artifacts", fake_runtime_artifacts)
    monkeypatch.setattr(hparam_selection.evidence, "checkpoint_file_sha256", lambda *_args: "b" * 64)
    monkeypatch.setattr(hparam_selection.tracking, "validate_checkpoint_evidence_rows", lambda *_args: None)
    merge_run_manifest(
        tmp_path,
        [
            {
                "step_id": run["step_id"],
                "run_id": run["run_id"],
                "target": "ssh",
                "host": "unit-host",
                "status": "completed",
            }
            for run in runs
        ],
    )
    canonical = tmp_path / "run_manifest.tsv"
    events = tmp_path / "events.jsonl"
    canonical_before = canonical.read_bytes()
    events_before = events.read_bytes()

    with pytest.raises(ValueError, match="Successful SSH hparam run has unavailable runtime artifacts"):
        hparam_selection.select_hparam_candidates(plan_dir)

    assert not _ranking_path(plan_dir).exists()
    assert not (plan_dir / "checkpoint_test_ranking.csv").exists()
    assert canonical.read_bytes() == canonical_before
    assert events.read_bytes() == events_before


@pytest.mark.parametrize("mutated_epoch", [1, 2], ids=["selected-checkpoint", "non-selected-checkpoint"])
def test_hparam_select_ssh_reentry_fails_closed_on_any_checkpoint_hash_drift(
    tmp_path: Path,
    monkeypatch,
    mutated_epoch: int,
):
    recipe = _hparam_recipe(
        tmp_path,
        execution={
            "target": "ssh",
            "host": "unit-host",
            "workdir": "/remote/repository",
            "path_context": "remote",
            "path_validation": "defer",
        },
        selection_metric="test_ahi_pearson",
        selection_split="test",
        config_monitor="val_ahi_pearson",
    )
    plan_dir = tmp_path / "plan"
    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir))
    assert result.returncode == 0, result.stderr or result.stdout
    run = _first_run(plan_dir)
    checkpoints = [str(Path(run["checkpoint_dir"]) / f"epoch={epoch}.ckpt") for epoch in (1, 2)]
    manifest = {
        "test_all_checkpoints_after_fit": True,
        "checkpoint_test_results": [
            {
                "checkpoint_path": checkpoint,
                "epoch": epoch,
                "metrics": {"test_ahi_pearson": score},
            }
            for checkpoint, epoch, score in zip(checkpoints, (1, 2), (0.9, 0.8))
        ],
    }
    current_hashes = {checkpoints[0]: "a" * 64, checkpoints[1]: "b" * 64}
    evidence_calls = []

    def fake_runtime_artifacts(row):
        assert (row["target"], row["host"]) == ("ssh", "unit-host")
        return str(Path(run["runtime_dir"]) / "run_manifest.json"), manifest, ["epoch=1.ckpt", "epoch=2.ckpt"]

    def fake_checkpoint_hash(row, checkpoint_path):
        assert (row["target"], row["host"]) == ("ssh", "unit-host")
        return current_hashes[checkpoint_path]

    def fake_validate_checkpoint_evidence(runs, rows, **_kwargs):
        runs_by_key = {(row["step_id"], row["run_id"]): row for row in runs}
        for row in rows:
            owner = runs_by_key[(row["step_id"], row["run_id"])]
            assert (owner["target"], owner["host"]) == ("ssh", "unit-host")
            evidence_calls.append(row["checkpoint_path"])

    monkeypatch.setattr(hparam_selection.evidence, "runtime_artifacts", fake_runtime_artifacts)
    monkeypatch.setattr(hparam_selection.evidence, "checkpoint_file_sha256", fake_checkpoint_hash)
    monkeypatch.setattr(
        hparam_selection.tracking,
        "validate_checkpoint_evidence_rows",
        fake_validate_checkpoint_evidence,
    )
    merge_run_manifest(
        tmp_path,
        [
            {
                "step_id": run["step_id"],
                "run_id": run["run_id"],
                "target": "ssh",
                "host": "unit-host",
                "status": "completed",
            }
        ],
    )
    canonical = read_run_manifest(tmp_path)[0]
    assert (canonical["target"], canonical["host"]) == ("ssh", "unit-host")

    ranking = hparam_selection.select_hparam_candidates(plan_dir)
    checkpoint_ranking = plan_dir / "checkpoint_test_ranking.csv"
    events = tmp_path / "events.jsonl"
    first_bytes = (ranking.read_bytes(), checkpoint_ranking.read_bytes(), events.read_bytes())

    hparam_selection.select_hparam_candidates(plan_dir)

    assert (ranking.read_bytes(), checkpoint_ranking.read_bytes(), events.read_bytes()) == first_bytes
    assert set(evidence_calls) == set(checkpoints)
    current_hashes[checkpoints[mutated_epoch - 1]] = "c" * 64

    with pytest.raises(ValueError, match="Frozen checkpoint SHA-256 differs"):
        hparam_selection.select_hparam_candidates(plan_dir)

    assert (ranking.read_bytes(), checkpoint_ranking.read_bytes(), events.read_bytes()) == first_bytes


@pytest.mark.parametrize("delete_workspace_ranking", [False, True], ids=["audit-deleted", "both-deleted"])
@pytest.mark.parametrize("mutated_epoch", [1, 2], ids=["selected-checkpoint", "non-selected-checkpoint"])
def test_hparam_select_does_not_rebuild_deleted_frozen_rankings_after_checkpoint_drift(
    tmp_path: Path,
    delete_workspace_ranking: bool,
    mutated_epoch: int,
):
    recipe = _hparam_recipe(
        tmp_path,
        selection_metric="test_ahi_pearson",
        selection_split="test",
        config_monitor="val_ahi_pearson",
    )
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    run = _first_run(plan_dir)
    checkpoint_dir = Path(run["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True)
    checkpoints = [checkpoint_dir / "epoch=1.ckpt", checkpoint_dir / "epoch=2.ckpt"]
    for checkpoint in checkpoints:
        checkpoint.write_text(checkpoint.name)
    (Path(run["runtime_dir"]) / "run_manifest.json").write_text(
        json.dumps(
            {
                "test_all_checkpoints_after_fit": True,
                "checkpoint_test_results": [
                    {
                        "checkpoint_path": str(checkpoint),
                        "epoch": epoch,
                        "metrics": {"test_ahi_pearson": score},
                    }
                    for checkpoint, epoch, score in zip(checkpoints, (1, 2), (0.9, 0.8))
                ],
            }
        )
    )
    merge_run_manifest(
        tmp_path,
        [{"step_id": run["step_id"], "run_id": run["run_id"], "status": "completed"}],
    )
    ranking = hparam_selection.select_hparam_candidates(plan_dir)
    checkpoint_ranking = plan_dir / "checkpoint_test_ranking.csv"
    ranking_before = ranking.read_bytes()
    events = tmp_path / "events.jsonl"
    events_before = events.read_bytes()

    checkpoint_ranking.unlink()
    if delete_workspace_ranking:
        ranking.unlink()
    checkpoints[mutated_epoch - 1].write_text("drifted checkpoint")

    error = (
        "Frozen checkpoint SHA-256 differs"
        if mutated_epoch == 1
        else "Frozen checkpoint test ranking referenced by candidate_selected event is missing"
    )
    with pytest.raises(ValueError, match=error):
        hparam_selection.select_hparam_candidates(plan_dir)

    assert not checkpoint_ranking.exists()
    if delete_workspace_ranking:
        assert not ranking.exists()
    else:
        assert ranking.read_bytes() == ranking_before
    assert events.read_bytes() == events_before


def test_hparam_select_rebuilds_missing_shared_test_ranking_from_canonical_rows(tmp_path: Path):
    recipe = _hparam_recipe(
        tmp_path,
        selection_metric="test_ahi_pearson",
        selection_split="test",
        config_monitor="val_ahi_pearson",
    )
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    run = _first_run(plan_dir)
    checkpoint = Path(run["checkpoint_dir"]) / "epoch=1.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text("checkpoint")
    (Path(run["runtime_dir"]) / "run_manifest.json").write_text(
        json.dumps(
            {
                "test_all_checkpoints_after_fit": True,
                "checkpoint_test_results": [
                    {
                        "checkpoint_path": str(checkpoint),
                        "epoch": 1,
                        "metrics": {"test_ahi_pearson": 0.8},
                    }
                ],
            }
        )
    )
    ranking = hparam_selection.select_hparam_candidates(plan_dir)
    ranking_before = ranking.read_bytes()
    ranking.unlink()
    assert experiments.experiment_status(tmp_path)["summary"]["state"] == "ready_to_report"

    hparam_selection.select_hparam_candidates(plan_dir)

    assert ranking.read_bytes() == ranking_before
    assert experiments.experiment_status(tmp_path)["summary"]["state"] == "ready_to_finalize"


@pytest.mark.parametrize("mutation", ["missing", "wrong_event_path"])
def test_hparam_select_rejects_other_test_step_audit_drift_before_rebuilding_shared_ranking(
    tmp_path: Path,
    mutation: str,
):
    first_plan, second_plan = _prepare_two_test_selected_steps(tmp_path)
    shared_ranking = _ranking_path(first_plan)
    shared_ranking.unlink()
    second_audit = second_plan / "checkpoint_test_ranking.csv"
    if mutation == "missing":
        second_audit.unlink()
    else:
        bogus = tmp_path / "bogus-checkpoint-ranking.csv"
        bogus.write_bytes(second_audit.read_bytes())
        events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
        next(
            event
            for event in events
            if event.get("event_type") == "candidate_selected" and event.get("step_id") == "b-tune"
        )["checkpoint_ranking"] = str(bogus)
        (tmp_path / "events.jsonl").write_text("\n".join(json.dumps(event, sort_keys=True) for event in events) + "\n")
    before = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}

    with pytest.raises(ValueError, match="checkpoint test ranking referenced.*missing or differs"):
        hparam_selection.select_hparam_candidates(first_plan)

    assert {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before


def test_hparam_select_replays_missing_candidate_event_from_canonical_selection(tmp_path: Path):
    first_plan, second_plan = _prepare_two_test_selected_steps(tmp_path)
    events_path = tmp_path / "events.jsonl"
    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    events_path.write_text(
        "\n".join(
            json.dumps(event, sort_keys=True)
            for event in events
            if not (event.get("event_type") == "candidate_selected" and event.get("step_id") == "a-tune")
        )
        + "\n"
    )
    frozen = {
        path: path.read_bytes()
        for path in (
            _ranking_path(first_plan),
            tmp_path / "reports" / "hparam_selection.md",
            first_plan / "checkpoint_test_ranking.csv",
            second_plan / "checkpoint_test_ranking.csv",
        )
    }

    hparam_selection.select_hparam_candidates(first_plan)

    assert {path: path.read_bytes() for path in frozen} == frozen
    selected = [
        json.loads(line)
        for line in events_path.read_text().splitlines()
        if json.loads(line).get("event_type") == "candidate_selected"
    ]
    assert [event["step_id"] for event in selected] == ["b-tune", "a-tune"]


def test_status_and_finalize_require_every_test_selection_plan_audit(tmp_path: Path):
    plans = _prepare_two_test_selected_steps(tmp_path)
    (plans[1] / "checkpoint_test_ranking.csv").unlink()
    selection_report = tmp_path / "reports" / "hparam_selection.md"
    before = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}

    with pytest.raises(ValueError, match="checkpoint test ranking is missing"):
        experiments.experiment_status(tmp_path)
    with pytest.raises(ValueError, match="checkpoint test ranking is missing"):
        experiments.finalize_experiment(tmp_path, selection_report)
    assert {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("checkpoint_rank", "99"),
        ("epoch", "99"),
        ("run_manifest", "forged-runtime-manifest.json"),
        ("source", "forged_source"),
        ("status", "finished"),
    ],
)
def test_status_and_finalize_bind_test_selection_provenance_to_plan_audit(
    tmp_path: Path,
    field: str,
    value: str,
):
    _prepare_two_test_selected_steps(tmp_path)
    canonical = read_rows(tmp_path / "run_manifest.tsv", require_managed_identity=True)
    selected = next(row for row in canonical if row["step_id"] == "a-tune")
    selected[field] = str(tmp_path / value) if field == "run_manifest" else value
    write_rows(tmp_path / "run_manifest.tsv", canonical)
    ranking_path = tmp_path / "reports" / "ranking.csv"
    ranking = read_rows(ranking_path, require_managed_identity=True)
    next(row for row in ranking if row["step_id"] == "a-tune")[field] = selected[field]
    write_rows(ranking_path, ranking)
    selection_report = tmp_path / "reports" / "hparam_selection.md"
    before = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}

    with pytest.raises(ValueError, match="differs from frozen checkpoint test ranking"):
        experiments.experiment_status(tmp_path)
    with pytest.raises(ValueError, match="differs from frozen checkpoint test ranking"):
        experiments.finalize_experiment(tmp_path, selection_report)
    assert {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before


def test_finalize_rechecks_test_selection_audits_before_terminal_commit(tmp_path: Path, monkeypatch):
    plans = _prepare_two_test_selected_steps(tmp_path)
    audit = plans[0] / "checkpoint_test_ranking.csv"
    selection_report = tmp_path / "reports" / "hparam_selection.md"
    manifest = tmp_path / "experiment.yaml"
    manifest_before = manifest.read_bytes()
    original_replace = experiments.exp_io.conditional_atomic_replace_text_at

    def publish_then_tamper(path, text, expected_sha256, *, remote=None, **kwargs):
        committed = original_replace(path, text, expected_sha256, remote=remote, **kwargs)
        if Path(path) == tmp_path / "reports" / "final.md" and committed:
            audit.write_text(audit.read_text().replace("checkpoint_test_results", "forged_source", 1))
        return committed

    monkeypatch.setattr(experiments.exp_io, "conditional_atomic_replace_text_at", publish_then_tamper)

    with pytest.raises(ValueError, match="checkpoint test ranking changed during finalization"):
        experiments.finalize_experiment(tmp_path, selection_report)

    assert manifest.read_bytes() == manifest_before
    assert yaml.safe_load(manifest.read_text())["experiment"].get("status") is None


def test_status_and_finalize_reject_truncated_bound_test_selection_audit(tmp_path: Path):
    plan_dir = _prepare_test_selected_plan_with_two_checkpoints(tmp_path)
    audit = plan_dir / "checkpoint_test_ranking.csv"
    write_rows(audit, read_rows(audit, require_managed_identity=True)[:1])
    selection_report = tmp_path / "reports" / "hparam_selection.md"
    before = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}

    with pytest.raises(ValueError, match="differs from frozen checkpoint test ranking"):
        experiments.experiment_status(tmp_path)
    with pytest.raises(ValueError, match="differs from frozen checkpoint test ranking"):
        experiments.finalize_experiment(tmp_path, selection_report)

    assert {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before


@pytest.mark.parametrize(
    ("field", "value"),
    [("run_manifest", "/forged/nonwinner.json"), ("status", "finished")],
)
def test_test_selection_binds_nonwinning_checkpoint_audit_provenance(
    tmp_path: Path,
    field: str,
    value: str,
):
    plan_dir = _prepare_test_selected_plan_with_two_checkpoints(tmp_path)
    audit = plan_dir / "checkpoint_test_ranking.csv"
    audit_rows = read_rows(audit, require_managed_identity=True)
    audit_rows[1][field] = value
    write_rows(audit, audit_rows)
    canonical = read_rows(tmp_path / "run_manifest.tsv", require_managed_identity=True)
    canonical[0]["checkpoint_ranking_sha256"] = hashlib.sha256(audit.read_bytes()).hexdigest()
    write_rows(tmp_path / "run_manifest.tsv", canonical)
    selection_report = tmp_path / "reports" / "hparam_selection.md"
    before = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}

    with pytest.raises(ValueError, match="differs from current checkpoint test evidence"):
        hparam_selection.select_hparam_candidates(plan_dir)
    with pytest.raises(ValueError, match="differs from frozen checkpoint test ranking"):
        experiments.experiment_status(tmp_path)
    with pytest.raises(ValueError, match="differs from frozen checkpoint test ranking"):
        experiments.finalize_experiment(tmp_path, selection_report)

    assert {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before


def test_finalize_revalidates_every_bound_test_selection_checkpoint(tmp_path: Path):
    plan_dir = _prepare_test_selected_plan_with_two_checkpoints(tmp_path)
    audit_rows = read_rows(plan_dir / "checkpoint_test_ranking.csv", require_managed_identity=True)
    Path(audit_rows[1]["checkpoint_path"]).write_text("tampered losing checkpoint")
    selection_report = tmp_path / "reports" / "hparam_selection.md"
    before = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}

    assert experiments.experiment_status(tmp_path)["summary"]["state"] == "ready_to_finalize"
    with pytest.raises(ValueError, match="Frozen checkpoint SHA-256 differs"):
        experiments.finalize_experiment(tmp_path, selection_report)

    assert {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before


def test_hparam_select_rebuilds_test_ranking_from_new_registered_plan(tmp_path: Path):
    recipe = _hparam_recipe(
        tmp_path,
        selection_metric="test_ahi_pearson",
        selection_split="test",
        config_monitor="val_ahi_pearson",
    )
    plans = []
    first_checkpoint_ranking = b""
    for index, score in enumerate((0.8, 0.9), start=1):
        plan_dir = tmp_path / f"plan-{index}"
        assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
        run = _first_run(plan_dir)
        checkpoint_dir = Path(run["checkpoint_dir"])
        checkpoint_dir.mkdir(parents=True)
        checkpoint = checkpoint_dir / f"epoch={index}.ckpt"
        checkpoint.write_text(f"checkpoint-{index}")
        (Path(run["runtime_dir"]) / "run_manifest.json").write_text(
            json.dumps(
                {
                    "test_all_checkpoints_after_fit": True,
                    "checkpoint_test_results": [
                        {
                            "checkpoint_path": str(checkpoint),
                            "epoch": index,
                            "metrics": {"test_ahi_pearson": score},
                        }
                    ],
                }
            )
        )
        merge_run_manifest(
            tmp_path,
            [{"step_id": run["step_id"], "run_id": run["run_id"], "status": "completed"}],
        )
        hparam_selection.select_hparam_candidates(plan_dir)
        plans.append(plan_dir)
        if index == 1:
            first_checkpoint_ranking = (plan_dir / "checkpoint_test_ranking.csv").read_bytes()

    ranking = _read_table(_ranking_path(plans[-1]))
    assert [(row["run_id"], row["score"]) for row in ranking] == [("run-001", "0.9"), ("run-000", "0.8")]
    checkpoint_ranking = _read_table(plans[-1] / "checkpoint_test_ranking.csv")
    assert [(row["run_id"], row["score"]) for row in checkpoint_ranking] == [("run-001", "0.9")]
    assert len(_read_table(plans[0] / "checkpoint_test_ranking.csv")) == 1

    hparam_selection.select_hparam_candidates(plans[0])

    assert (plans[0] / "checkpoint_test_ranking.csv").read_bytes() == first_checkpoint_ranking
    ranking = _read_table(_ranking_path(plans[0]))
    assert [(row["run_id"], row["score"]) for row in ranking] == [("run-001", "0.9"), ("run-000", "0.8")]
    assert [(row["run_id"], row["checkpoint_rank"]) for row in ranking] == [
        ("run-001", "1"),
        ("run-000", "2"),
    ]
    assert _read_table(plans[0] / "checkpoint_test_ranking.csv")[0]["rank"] == "1"
    selection_events = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text().splitlines()
        if json.loads(line).get("event_type") == "candidate_selected"
    ]
    assert len(selection_events) == 2
    assert selection_events[-1]["checkpoint_ranking"] == str(plans[1] / "checkpoint_test_ranking.csv")
    assert experiments.experiment_status(tmp_path)["summary"]["state"] == "ready_to_finalize"
    report = tmp_path / "reports" / "hparam_selection.md"
    assert experiments.finalize_experiment(tmp_path, report).read_text() == report.read_text()


def test_hparam_select_freezes_and_requires_every_successful_test_plan_audit(tmp_path: Path):
    recipe = _hparam_recipe(
        tmp_path,
        selection_metric="test_ahi_pearson",
        selection_split="test",
        config_monitor="val_ahi_pearson",
    )
    plans = []
    for index, score in enumerate((0.9, 0.8), start=1):
        plan_dir = tmp_path / f"plan-{index}"
        assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
        run = _first_run(plan_dir)
        checkpoint = Path(run["checkpoint_dir"]) / f"epoch={index}.ckpt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_text(f"checkpoint-{index}")
        (Path(run["runtime_dir"]) / "run_manifest.json").write_text(
            json.dumps(
                {
                    "test_all_checkpoints_after_fit": True,
                    "checkpoint_test_results": [
                        {
                            "checkpoint_path": str(checkpoint),
                            "epoch": index,
                            "metrics": {"test_ahi_pearson": score},
                        }
                    ],
                }
            )
        )
        plans.append(plan_dir)

    shared_ranking = hparam_selection.select_hparam_candidates(plans[0])

    assert all((plan_dir / "checkpoint_test_ranking.csv").is_file() for plan_dir in plans)
    shared_ranking.unlink()
    (plans[1] / "checkpoint_test_ranking.csv").unlink()
    before = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}

    with pytest.raises(ValueError, match="checkpoint test ranking referenced.*missing or differs"):
        hparam_selection.select_hparam_candidates(plans[0])

    assert {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before


def test_hparam_select_rejects_a_plan_local_audit_missing_an_owned_run(tmp_path: Path):
    recipe = _hparam_recipe(
        tmp_path,
        selection_metric="test_ahi_pearson",
        selection_split="test",
        config_monitor="val_ahi_pearson",
        max_runs=2,
    )
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    plan = json.loads((plan_dir / "plan.json").read_text())
    for index, run in enumerate(plan["runs"]):
        checkpoint_dir = Path(run["checkpoint_dir"])
        checkpoint_dir.mkdir(parents=True)
        checkpoint = checkpoint_dir / "epoch=1.ckpt"
        checkpoint.write_text(f"checkpoint-{index}")
        (Path(run["runtime_dir"]) / "run_manifest.json").write_text(
            json.dumps(
                {
                    "test_all_checkpoints_after_fit": True,
                    "checkpoint_test_results": [
                        {
                            "checkpoint_path": str(checkpoint),
                            "epoch": 1,
                            "metrics": {"test_ahi_pearson": 0.9 - index * 0.1},
                        }
                    ],
                }
            )
        )
    merge_run_manifest(
        tmp_path,
        [{"step_id": run["step_id"], "run_id": run["run_id"], "status": "completed"} for run in plan["runs"]],
    )
    ranking = hparam_selection.select_hparam_candidates(plan_dir)
    checkpoint_ranking = plan_dir / "checkpoint_test_ranking.csv"
    events = tmp_path / "events.jsonl"
    ranking_before = ranking.read_bytes()
    events_before = events.read_bytes()
    checkpoint_rows = read_rows(checkpoint_ranking, require_managed_identity=True)
    write_rows(checkpoint_ranking, [row for row in checkpoint_rows if row["run_id"] != "run-001"])
    truncated_audit = checkpoint_ranking.read_bytes()

    with pytest.raises(ValueError, match="differs from frozen checkpoint test ranking"):
        hparam_selection.select_hparam_candidates(plan_dir)

    assert checkpoint_ranking.read_bytes() == truncated_audit
    assert ranking.read_bytes() == ranking_before
    assert events.read_bytes() == events_before


def test_hparam_select_rejects_coherent_runtime_and_bound_audit_drift(tmp_path: Path):
    plan_dir = _prepare_test_selected_plan_with_two_checkpoints(tmp_path)
    audit = plan_dir / "checkpoint_test_ranking.csv"
    audit_rows = read_rows(audit, require_managed_identity=True)
    audit_rows[1]["score"] = "0.7"
    write_rows(audit, audit_rows)
    run = _first_run(plan_dir)
    runtime_manifest = Path(run["runtime_dir"]) / "run_manifest.json"
    payload = json.loads(runtime_manifest.read_text())
    payload["checkpoint_test_results"][1]["metrics"]["test_ahi_pearson"] = 0.7
    runtime_manifest.write_text(json.dumps(payload))
    before = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}

    with pytest.raises(ValueError, match="Canonical hparam selection differs from frozen checkpoint test ranking"):
        hparam_selection.select_hparam_candidates(plan_dir)

    assert {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before


def test_hparam_select_rejects_incomplete_checkpoint_test_results_before_writing(tmp_path: Path):
    recipe = _hparam_recipe(
        tmp_path,
        selection_metric="test_ahi_pearson",
        selection_split="test",
        config_monitor="val_ahi_pearson",
    )
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    run = _first_run(plan_dir)
    checkpoint_dir = Path(run["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True)
    first = checkpoint_dir / "epoch=1.ckpt"
    second = checkpoint_dir / "epoch=2.ckpt"
    first.write_text("first")
    second.write_text("second")
    (Path(run["runtime_dir"]) / "run_manifest.json").write_text(
        json.dumps(
            {
                "test_all_checkpoints_after_fit": True,
                "checkpoint_test_results": [
                    {
                        "checkpoint_path": str(first),
                        "epoch": 1,
                        "metrics": {"test_ahi_pearson": 0.8},
                    }
                ],
            }
        )
    )
    merge_run_manifest(
        tmp_path,
        [{"step_id": run["step_id"], "run_id": run["run_id"], "status": "completed"}],
    )

    with pytest.raises(ValueError, match="checkpoint_test_results is incomplete"):
        hparam_selection.select_hparam_candidates(plan_dir)

    assert not (plan_dir / "checkpoint_test_ranking.csv").exists()
    assert not _ranking_path(plan_dir).exists()


@pytest.mark.parametrize("runtime_mode", [False, None])
def test_hparam_select_requires_runtime_all_checkpoint_mode(tmp_path: Path, runtime_mode):
    recipe = _hparam_recipe(
        tmp_path,
        selection_metric="test_ahi_pearson",
        selection_split="test",
        config_monitor="val_ahi_pearson",
    )
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    run = _first_run(plan_dir)
    checkpoint_dir = Path(run["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True)
    checkpoint = checkpoint_dir / "epoch=1.ckpt"
    checkpoint.write_text("checkpoint")
    manifest = {
        "checkpoint_test_results": [
            {
                "checkpoint_path": str(checkpoint),
                "epoch": 1,
                "metrics": {"test_ahi_pearson": 0.8},
            }
        ]
    }
    if runtime_mode is not None:
        manifest["test_all_checkpoints_after_fit"] = runtime_mode
    (Path(run["runtime_dir"]) / "run_manifest.json").write_text(json.dumps(manifest))
    merge_run_manifest(
        tmp_path,
        [{"step_id": run["step_id"], "run_id": run["run_id"], "status": "completed"}],
    )

    with pytest.raises(ValueError, match="did not enable test_all_checkpoints_after_fit"):
        hparam_selection.select_hparam_candidates(plan_dir)

    assert not (plan_dir / "checkpoint_test_ranking.csv").exists()
    assert not _ranking_path(plan_dir).exists()


@pytest.mark.parametrize("score", [True, float("nan"), float("inf"), "not-a-number", None])
def test_hparam_select_requires_finite_checkpoint_test_metric(tmp_path: Path, score):
    recipe = _hparam_recipe(
        tmp_path,
        selection_metric="test_ahi_pearson",
        selection_split="test",
        config_monitor="val_ahi_pearson",
    )
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    run = _first_run(plan_dir)
    checkpoint_dir = Path(run["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True)
    checkpoint = checkpoint_dir / "epoch=1.ckpt"
    checkpoint.write_text("checkpoint")
    (Path(run["runtime_dir"]) / "run_manifest.json").write_text(
        json.dumps(
            {
                "test_all_checkpoints_after_fit": True,
                "checkpoint_test_results": [
                    {
                        "checkpoint_path": str(checkpoint),
                        "epoch": 1,
                        "metrics": {"test_ahi_pearson": score},
                    }
                ],
            }
        )
    )
    merge_run_manifest(
        tmp_path,
        [{"step_id": run["step_id"], "run_id": run["run_id"], "status": "completed"}],
    )

    with pytest.raises(ValueError, match="missing a finite test_ahi_pearson"):
        hparam_selection.select_hparam_candidates(plan_dir)

    assert not (plan_dir / "checkpoint_test_ranking.csv").exists()
    assert not _ranking_path(plan_dir).exists()


def test_hparam_select_rejects_duplicate_numeric_checkpoint_epochs(tmp_path: Path):
    recipe = _hparam_recipe(
        tmp_path,
        selection_metric="test_ahi_pearson",
        selection_split="test",
        config_monitor="val_ahi_pearson",
    )
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    run = _first_run(plan_dir)
    checkpoint_dir = Path(run["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True)
    checkpoints = [checkpoint_dir / "epoch=1.ckpt", checkpoint_dir / "epoch=01.ckpt"]
    for checkpoint in checkpoints:
        checkpoint.write_text(checkpoint.name)
    (Path(run["runtime_dir"]) / "run_manifest.json").write_text(
        json.dumps(
            {
                "test_all_checkpoints_after_fit": True,
                "checkpoint_test_results": [
                    {
                        "checkpoint_path": str(checkpoint),
                        "epoch": 1,
                        "metrics": {"test_ahi_pearson": score},
                    }
                    for checkpoint, score in zip(checkpoints, (0.8, 0.9))
                ],
            }
        )
    )
    merge_run_manifest(
        tmp_path,
        [{"step_id": run["step_id"], "run_id": run["run_id"], "status": "completed"}],
    )

    with pytest.raises(ValueError, match="duplicate epoch"):
        hparam_selection.select_hparam_candidates(plan_dir)

    assert not (plan_dir / "checkpoint_test_ranking.csv").exists()
    assert not _ranking_path(plan_dir).exists()


def test_hparam_select_rejects_malformed_saved_epoch_checkpoint(tmp_path: Path):
    recipe = _hparam_recipe(
        tmp_path,
        selection_metric="test_ahi_pearson",
        selection_split="test",
        config_monitor="val_ahi_pearson",
    )
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    run = _first_run(plan_dir)
    checkpoint_dir = Path(run["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "epoch=invalid.ckpt").write_text("checkpoint")
    (Path(run["runtime_dir"]) / "run_manifest.json").write_text(
        json.dumps(
            {
                "test_all_checkpoints_after_fit": True,
                "checkpoint_test_results": [],
            }
        )
    )
    merge_run_manifest(
        tmp_path,
        [{"step_id": run["step_id"], "run_id": run["run_id"], "status": "completed"}],
    )

    with pytest.raises(ValueError, match="Saved epoch checkpoint has an invalid epoch"):
        hparam_selection.select_hparam_candidates(plan_dir)

    assert not (plan_dir / "checkpoint_test_ranking.csv").exists()
    assert not _ranking_path(plan_dir).exists()


def test_hparam_select_reads_the_user_materialized_effective_metric(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path, selection_metric="val_effective")
    payload = yaml.safe_load(recipe.read_text())
    payload["evaluation_policy"]["selection_metric"] = "val_stale"
    write_yaml(recipe, payload)
    decisions = write_yaml(
        tmp_path / "decisions.yaml",
        {"decisions": {"selection_metric": {"value": "val_effective", "source": "explicit_user"}}},
    )
    plan_dir = tmp_path / "plan"
    result = _run(
        "plan",
        "--recipe",
        str(recipe),
        "--user-decisions",
        str(decisions),
        "--output-dir",
        str(plan_dir),
    )
    assert result.returncode == 0, result.stderr or result.stdout
    run = _first_run(plan_dir)
    runtime_dir = Path(run["runtime_dir"])
    checkpoint_dir = Path(run["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "epoch=1.ckpt").write_text("checkpoint")
    (runtime_dir / "run_manifest.json").write_text(json.dumps({"epoch": 1, "metrics": {"val_effective": 0.7}}))

    ranking = hparam_selection.select_hparam_candidates(plan_dir)

    row = _read_table(ranking)[0]
    assert row["metric"] == "val_effective"
    assert row["score"] == "0.7"


def test_hparam_select_preserves_zero_padded_epoch_checkpoint(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    run = _first_run(plan_dir)
    version = run["version"]
    run_dir = Path(run["runtime_dir"])
    ckpt_dir = Path(run["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True)
    fixed = ckpt_dir / "epoch=09-step=90.ckpt"
    fixed.write_text("fixed")
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "version": version,
                "monitor": "val_ahi_pearson",
                "best_model_score": 0.72,
                "best_model_path": str(ckpt_dir / "best-epoch=09-step=90.ckpt"),
                "epoch": 9,
                "metrics": {"val_ahi_pearson": 0.72},
            }
        )
    )

    result = _run(
        "hparam-select",
        "--run-dir",
        str(plan_dir),
        "--metric",
        "val_ahi_pearson",
        "--mode",
        "max",
    )

    assert result.returncode == 0, result.stderr
    rows = _read_table(_ranking_path(plan_dir))
    assert rows[0]["checkpoint_path"] == str(fixed)


@pytest.mark.parametrize("score", [None, "not-a-number", float("nan"), float("inf"), True])
def test_hparam_select_fails_without_any_valid_score_and_preserves_state(tmp_path: Path, score):
    recipe = _hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    run = _first_run(plan_dir)
    if score is not None:
        runtime_dir = Path(run["runtime_dir"])
        runtime_dir.mkdir(parents=True, exist_ok=True)
        (runtime_dir / "run_manifest.json").write_text(json.dumps({"metrics": {"val_ahi_pearson": score}}))
    canonical = tmp_path / "run_manifest.tsv"
    events = tmp_path / "events.jsonl"
    canonical_before = canonical.read_bytes()
    events_before = events.read_bytes()

    with pytest.raises(ValueError, match="No valid val_ahi_pearson scores"):
        hparam_selection.select_hparam_candidates(plan_dir)

    assert not _ranking_path(plan_dir).exists()
    assert canonical.read_bytes() == canonical_before
    assert events.read_bytes() == events_before


def test_hparam_select_uses_canonical_status_not_runtime_manifest_status(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    run = _first_run(plan_dir)
    run_dir = Path(run["runtime_dir"])
    checkpoint_dir = Path(run["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True)
    checkpoint = checkpoint_dir / "epoch=1.ckpt"
    checkpoint.write_text("checkpoint")
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "epoch": 1,
                "checkpoint_path": str(checkpoint),
                "metrics": {"val_ahi_pearson": 0.7},
            }
        )
    )
    canonical = read_rows(tmp_path / "run_manifest.tsv")
    canonical[0]["status"] = "failed"
    write_rows(tmp_path / "run_manifest.tsv", canonical)

    result = _run("hparam-select", "--run-dir", str(plan_dir))

    assert result.returncode == 1
    assert "No valid val_ahi_pearson scores" in result.stderr
    assert not _ranking_path(plan_dir).exists()


def test_hparam_select_requires_terminal_canonical_runs_for_val_selection(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    run = json.loads((plan_dir / "plan.json").read_text())["runs"][0]
    checkpoint = Path(run["checkpoint_dir"]) / "epoch=1.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text("checkpoint")
    (Path(run["runtime_dir"]) / "run_manifest.json").write_text(
        json.dumps({"metrics": {"val_ahi_pearson": 0.7}, "checkpoint_path": str(checkpoint), "epoch": 1})
    )
    before = (tmp_path / "run_manifest.tsv").read_bytes()

    with pytest.raises(ValueError, match="requires every managed hparam run to be terminal"):
        hparam_selection.select_hparam_candidates(plan_dir)

    assert (tmp_path / "run_manifest.tsv").read_bytes() == before
    assert not _ranking_path(plan_dir).exists()


def test_hparam_checkpoint_scan_ranks_history_fixed_epoch_checkpoints(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    run = _first_run(plan_dir)
    version = run["version"]
    run_dir = Path(run["runtime_dir"])
    ckpt_dir = Path(run["checkpoint_dir"])
    history_dir = run_dir / "wandb" / "run-1" / "files"
    ckpt_dir.mkdir(parents=True)
    history_dir.mkdir(parents=True)
    (ckpt_dir / "epoch=13.ckpt").write_text("fixed13")
    (ckpt_dir / "epoch=20.ckpt").write_text("fixed20")
    (ckpt_dir / "best-epoch=20.ckpt").write_text("alias")
    (history_dir / "wandb-history.jsonl").write_text(
        json.dumps({"epoch": 2, "val_auroc": 0.99})
        + "\n"
        + json.dumps({"epoch": 13, "val_auroc": 0.72})
        + "\n"
        + json.dumps({"epoch": 20, "val_auroc": 0.81})
        + "\n"
    )
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "version": version,
                "best_model_path": str(ckpt_dir / "best-epoch=20.ckpt"),
                "epoch": 20,
                "metrics": {"val_auroc": 0.5},
            }
        )
    )

    result = _run(
        "hparam-checkpoint-scan",
        "--run-dir",
        str(plan_dir),
        "--metric",
        "val_auroc",
        "--mode",
        "max",
    )

    assert result.returncode == 0, result.stderr
    rows = _read_table(plan_dir / "checkpoint_ranking.csv")
    assert rows[0]["epoch"] == "20"
    assert rows[0]["score"] == "0.81"
    assert rows[0]["checkpoint_path"].endswith("epoch=20.ckpt")
    assert "best-epoch" not in rows[0]["checkpoint_path"]
    assert rows[0]["source"] == "history"
    assert {row["epoch"] for row in rows} == {"13", "20"}
    assert rows[0]["runtime.lr"] == "1e-06"
    first_output = (plan_dir / "checkpoint_ranking.csv").read_text()

    repeated = _run(
        "hparam-checkpoint-scan",
        "--run-dir",
        str(plan_dir),
        "--metric",
        "val_auroc",
        "--mode",
        "max",
    )

    assert repeated.returncode == 0, repeated.stderr
    assert (plan_dir / "checkpoint_ranking.csv").read_text() == first_output


@pytest.mark.parametrize(
    "history_row",
    [
        {"epoch": 1, "val_auroc": True},
        {"epoch": 1.5, "val_auroc": 0.8},
    ],
)
def test_hparam_checkpoint_scan_excludes_invalid_history_score_or_epoch(tmp_path: Path, history_row: dict):
    recipe = _hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    run = _first_run(plan_dir)
    runtime_dir = Path(run["runtime_dir"])
    checkpoint_dir = Path(run["checkpoint_dir"])
    history_dir = runtime_dir / "wandb" / "run-1" / "files"
    checkpoint_dir.mkdir(parents=True)
    history_dir.mkdir(parents=True)
    (checkpoint_dir / "epoch=1.ckpt").write_text("checkpoint")
    (history_dir / "wandb-history.jsonl").write_text(json.dumps(history_row) + "\n")
    (runtime_dir / "run_manifest.json").write_text("{}\n")

    ranking = hparam_selection.scan_hparam_checkpoints(plan_dir, "val_auroc", "max")

    assert ranking.read_text() == "step_id,run_id\n"


def test_hparam_checkpoint_scan_empty_output_remains_readable(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0

    first = hparam_selection.scan_hparam_checkpoints(plan_dir, "val_auroc", "max")
    second = hparam_selection.scan_hparam_checkpoints(plan_dir, "val_auroc", "max")

    assert first == second
    assert first.read_text() == "step_id,run_id\n"


@pytest.mark.parametrize("score", ["not-a-number", float("nan"), float("inf"), True])
def test_hparam_checkpoint_scan_excludes_invalid_manifest_scores(tmp_path: Path, score):
    recipe = _hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    run = _first_run(plan_dir)
    runtime_dir = Path(run["runtime_dir"])
    checkpoint_dir = Path(run["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "epoch=1.ckpt").write_text("checkpoint")
    (runtime_dir / "run_manifest.json").write_text(json.dumps({"epoch": 1, "metrics": {"val_auroc": score}}))

    ranking = hparam_selection.scan_hparam_checkpoints(plan_dir, "val_auroc", "max")

    assert ranking.read_text() == "step_id,run_id\n"


def test_hparam_select_does_not_scan_unmanaged_runtime_directories(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    run = _first_run(plan_dir)
    decoy_dir = plan_dir / "unmanaged" / run["version"]
    decoy_checkpoint_dir = decoy_dir / "checkpoints"
    decoy_checkpoint_dir.mkdir(parents=True)
    decoy_checkpoint = decoy_checkpoint_dir / "epoch=99.ckpt"
    decoy_checkpoint.write_text("decoy")
    (decoy_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "monitor": "val_ahi_pearson",
                "best_model_score": 0.99,
                "best_model_path": str(decoy_checkpoint),
                "metrics": {"val_ahi_pearson": 0.99},
            }
        )
    )
    managed_runtime = Path(run["runtime_dir"])
    managed_runtime.mkdir(parents=True, exist_ok=True)
    managed_manifest = managed_runtime / "run_manifest.json"
    managed_checkpoint_dir = Path(run["checkpoint_dir"])
    managed_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    managed_checkpoint = managed_checkpoint_dir / "epoch=1.ckpt"
    managed_checkpoint.write_text("managed")
    managed_manifest.write_text(
        json.dumps(
            {
                "epoch": 1,
                "metrics": {"val_ahi_pearson": 0.7},
                "checkpoint_path": str(managed_checkpoint),
            }
        )
    )

    result = _run("hparam-select", "--run-dir", str(plan_dir))

    assert result.returncode == 0, result.stderr
    row = _read_table(_ranking_path(plan_dir))[0]
    assert row["score"] == "0.7"
    assert row["run_manifest"] == str(managed_manifest)
    assert row["checkpoint_path"] == str(managed_checkpoint)


def test_hparam_select_requires_checkpoint_evidence_for_finite_score(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    run = _first_run(plan_dir)
    runtime_dir = Path(run["runtime_dir"])
    runtime_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(run["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "epoch=1.ckpt").write_text("unbound checkpoint")
    (runtime_dir / "run_manifest.json").write_text(json.dumps({"metrics": {"val_ahi_pearson": 0.7}}))

    with pytest.raises(ValueError, match="No valid val_ahi_pearson scores"):
        hparam_selection.select_hparam_candidates(plan_dir)

    assert not _ranking_path(plan_dir).exists()
    assert not any(
        json.loads(line)["event_type"] == "candidate_selected"
        for line in (tmp_path / "events.jsonl").read_text().splitlines()
    )


def test_hparam_select_rejects_hardlinked_checkpoint_evidence(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    run = _first_run(plan_dir)
    runtime_dir = Path(run["runtime_dir"])
    checkpoint_dir = Path(run["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    foreign = tmp_path / "foreign.ckpt"
    foreign.write_text("checkpoint")
    checkpoint = checkpoint_dir / "epoch=1.ckpt"
    checkpoint.hardlink_to(foreign)
    (runtime_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "epoch": 1,
                "metrics": {"val_ahi_pearson": 0.7},
                "checkpoint_path": str(checkpoint),
            }
        )
    )

    with pytest.raises(ValueError, match="independent regular files"):
        hparam_selection.select_hparam_candidates(plan_dir)

    assert not _ranking_path(plan_dir).exists()


def test_fixed_checkpoint_does_not_escape_frozen_checkpoint_dir(tmp_path: Path):
    checkpoint_dir = tmp_path / "managed" / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    unmanaged = tmp_path / "unmanaged" / "epoch=07.ckpt"
    unmanaged.parent.mkdir()
    unmanaged.write_text("unmanaged")

    path = run_artifacts.fixed_checkpoint_path({"best_model_path": str(unmanaged), "epoch": 7}, checkpoint_dir)

    assert path == ""


def test_fixed_checkpoint_requires_the_manifest_epoch_locally(tmp_path: Path):
    checkpoint_dir = tmp_path / "managed" / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "epoch=2.ckpt").write_text("wrong epoch")

    path = run_artifacts.fixed_checkpoint_path({"epoch": 1}, checkpoint_dir)

    assert path == ""


def test_fixed_checkpoint_requires_the_manifest_epoch_from_remote_names(tmp_path: Path):
    checkpoint_dir = tmp_path / "remote" / "checkpoints"

    path = run_artifacts.fixed_checkpoint_path_from_names(
        {"epoch": 1},
        checkpoint_dir,
        ["epoch=2.ckpt", "last.ckpt"],
    )

    assert path == ""


def test_fixed_checkpoint_rejects_unbound_local_and_remote_epochs(tmp_path: Path):
    checkpoint_dir = tmp_path / "managed" / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    for epoch in (1, 2):
        (checkpoint_dir / f"epoch={epoch}.ckpt").write_text("checkpoint")
    names = ["epoch=1.ckpt", "epoch=2.ckpt", "last.ckpt"]

    assert run_artifacts.fixed_checkpoint_path({}, checkpoint_dir) == ""
    assert run_artifacts.fixed_checkpoint_path_from_names({}, checkpoint_dir, names) == ""


def test_fixed_checkpoint_accepts_same_epoch_best_only_locally_and_remotely(tmp_path: Path):
    checkpoint_dir = tmp_path / "managed" / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    checkpoint = checkpoint_dir / "best-epoch=03.ckpt"
    checkpoint.write_text("checkpoint")
    manifest = {"best_model_path": str(checkpoint), "epoch": 3}

    assert run_artifacts.fixed_checkpoint_path(manifest, checkpoint_dir) == str(checkpoint)
    assert run_artifacts.fixed_checkpoint_path_from_names(manifest, checkpoint_dir, [checkpoint.name]) == str(
        checkpoint
    )


def test_fixed_checkpoint_rejects_best_only_symlink(tmp_path: Path):
    foreign = tmp_path / "foreign" / "best-epoch=03.ckpt"
    foreign.parent.mkdir()
    foreign.write_text("foreign")
    checkpoint_dir = tmp_path / "managed" / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    checkpoint = checkpoint_dir / foreign.name
    checkpoint.symlink_to(foreign)

    assert run_artifacts.fixed_checkpoint_path({"best_model_path": str(checkpoint), "epoch": 3}, checkpoint_dir) == ""


def test_fixed_checkpoint_rejects_fractional_manifest_epoch_locally_and_remotely(tmp_path: Path):
    checkpoint_dir = tmp_path / "managed" / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    checkpoint = checkpoint_dir / "epoch=2.ckpt"
    checkpoint.write_text("checkpoint")
    manifest = {"epoch": 2.5}

    assert run_artifacts.fixed_checkpoint_path(manifest, checkpoint_dir) == ""
    assert run_artifacts.fixed_checkpoint_path_from_names(manifest, checkpoint_dir, [checkpoint.name]) == ""


@pytest.mark.parametrize("value", [2, 2.0, "2", "2.0"])
def test_epoch_number_accepts_integer_values(value):
    assert run_artifacts.epoch_number(value) == 2


@pytest.mark.parametrize("value", [2.5, "2.5", float("nan"), float("inf"), True, "not-a-number"])
def test_epoch_number_rejects_non_integer_values(value):
    assert run_artifacts.epoch_number(value) is None


@pytest.mark.parametrize("alias_kind", ["checkpoint", "checkpoint_dir"])
def test_fixed_checkpoint_rejects_filesystem_aliases(tmp_path: Path, alias_kind: str):
    foreign_dir = tmp_path / "foreign"
    foreign_dir.mkdir()
    foreign = foreign_dir / "epoch=07.ckpt"
    foreign.write_text("foreign")
    checkpoint_dir = tmp_path / "managed" / "checkpoints"
    checkpoint_dir.parent.mkdir()
    if alias_kind == "checkpoint_dir":
        checkpoint_dir.symlink_to(foreign_dir, target_is_directory=True)
    else:
        checkpoint_dir.mkdir()
        (checkpoint_dir / foreign.name).symlink_to(foreign)

    path = run_artifacts.fixed_checkpoint_path(
        {"best_model_path": str(checkpoint_dir / foreign.name), "epoch": 7},
        checkpoint_dir,
    )

    assert path == ""


def test_hparam_select_rejects_legacy_plan_without_rewriting_outputs(tmp_path: Path):
    (tmp_path / "plan.json").write_text(json.dumps({"trials": [{"trial_id": "trial_000"}], "recipe": {}}))
    ranking = tmp_path / "candidate_ranking.csv"
    ranking.write_text("sentinel\n")

    with pytest.raises(ValueError, match="Legacy hparam plan"):
        hparam_selection.select_hparam_candidates(tmp_path)

    assert ranking.read_text() == "sentinel\n"


def test_hparam_select_rejects_historical_workspace_before_writing_ranking(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    (tmp_path / "run_manifest.tsv").write_text("trial_id\tstatus\ntrial_000\tfinished\n")
    ranking = _ranking_path(plan_dir)

    with pytest.raises(ValueError, match="read-only"):
        hparam_selection.select_hparam_candidates(plan_dir)

    assert not ranking.exists()


@pytest.mark.parametrize("manifest_state", ["missing", "current_key_absent"])
def test_hparam_select_requires_registered_run_manifest_before_writing(tmp_path: Path, manifest_state: str):
    recipe = _hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    manifest = tmp_path / "run_manifest.tsv"
    if manifest_state == "missing":
        manifest.unlink()
    else:
        manifest.write_text("experiment_id\tstep_id\trun_id\tstatus\nunit-experiment\tother-step\trun-000\tplanned\n")
    ranking = _ranking_path(plan_dir)
    before = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}

    with pytest.raises((FileNotFoundError, ValueError), match="missing"):
        hparam_selection.select_hparam_candidates(plan_dir)

    after = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    assert after == before
    assert not ranking.exists()


def test_hparam_select_rejects_unmanaged_existing_ranking_before_preserving_other_steps(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    ranking = _ranking_path(plan_dir)
    ranking.parent.mkdir(parents=True, exist_ok=True)
    write_rows(
        ranking,
        [
            {
                "experiment_id": "foreign-experiment",
                "step_id": "other-step",
                "run_id": "run-999",
                "version": "foreign-version",
                "config": str(tmp_path / "foreign.yaml"),
                "checkpoint_path": str(tmp_path / "foreign.ckpt"),
                "rank": 1,
            }
        ],
    )
    before = ranking.read_bytes()
    events_before = (tmp_path / "events.jsonl").read_bytes()

    with pytest.raises(ValueError, match="outside the canonical manifest"):
        hparam_selection.select_hparam_candidates(plan_dir)

    assert ranking.read_bytes() == before
    assert (tmp_path / "events.jsonl").read_bytes() == events_before


def test_hparam_select_drops_unselected_canonical_other_step_ranking(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    run = _first_run(plan_dir)
    runtime_dir = Path(run["runtime_dir"])
    checkpoint_dir = Path(run["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_dir / "epoch=1.ckpt"
    checkpoint.write_text("checkpoint")
    (runtime_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "epoch": 1,
                "checkpoint_path": str(checkpoint),
                "metrics": {"val_ahi_pearson": 0.7},
            }
        )
    )
    other_config = tmp_path / "other.yaml"
    other_config.write_text("model: other\n")
    other_checkpoint_dir = tmp_path / "other-checkpoints"
    other_checkpoint_dir.mkdir()
    merge_run_manifest(
        tmp_path,
        [
            {
                "experiment_id": "unit-experiment",
                "step_id": "other-step",
                "run_id": "run-999",
                "version": "other-version",
                "config": str(other_config),
                "checkpoint_dir": str(other_checkpoint_dir),
                "status": "completed",
            }
        ],
    )
    ranking = _ranking_path(plan_dir)
    other_checkpoint = other_checkpoint_dir / "epoch=1.ckpt"
    other_checkpoint.write_text("checkpoint")
    write_rows(
        ranking,
        [
            {
                "experiment_id": "unit-experiment",
                "step_id": "other-step",
                "run_id": "run-999",
                "version": "other-version",
                "config": str(other_config),
                "checkpoint_path": str(other_checkpoint),
                "metric": "val_other",
                "score": 0.5,
                "rank": 1,
            }
        ],
    )

    hparam_selection.select_hparam_candidates(plan_dir)

    rows = read_rows(ranking)
    assert {(row["step_id"], row["run_id"]) for row in rows} == {(run["step_id"], run["run_id"])}


def test_hparam_select_rejects_hardlinked_preserved_checkpoint_before_writing(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    other_config = tmp_path / "other.yaml"
    other_config.write_text("model: other\n")
    other_checkpoint_dir = tmp_path / "other-checkpoints"
    other_checkpoint_dir.mkdir()
    foreign_checkpoint = tmp_path / "foreign.ckpt"
    foreign_checkpoint.write_text("checkpoint")
    other_checkpoint = other_checkpoint_dir / "epoch=1.ckpt"
    other_checkpoint.hardlink_to(foreign_checkpoint)
    merge_run_manifest(
        tmp_path,
        [
            {
                "experiment_id": "unit-experiment",
                "step_id": "other-step",
                "run_id": "run-999",
                "version": "other-version",
                "config": str(other_config),
                "checkpoint_dir": str(other_checkpoint_dir),
                "status": "completed",
            }
        ],
    )
    ranking = _ranking_path(plan_dir)
    write_rows(
        ranking,
        [
            {
                "experiment_id": "unit-experiment",
                "step_id": "other-step",
                "run_id": "run-999",
                "version": "other-version",
                "config": str(other_config),
                "checkpoint_path": str(other_checkpoint),
                "metric": "val_other",
                "score": 0.5,
                "rank": 1,
            }
        ],
    )
    canonical = tmp_path / "run_manifest.tsv"
    events = tmp_path / "events.jsonl"
    ranking_before = ranking.read_bytes()
    canonical_before = canonical.read_bytes()
    events_before = events.read_bytes()

    with pytest.raises(ValueError, match="independent regular files"):
        hparam_selection.select_hparam_candidates(plan_dir)

    assert ranking.read_bytes() == ranking_before
    assert canonical.read_bytes() == canonical_before
    assert events.read_bytes() == events_before


def test_hparam_select_rejects_empty_preserved_checkpoint_before_writing(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    other_checkpoint_dir = tmp_path / "other-checkpoints"
    merge_run_manifest(
        tmp_path,
        [
            {
                "experiment_id": "unit-experiment",
                "step_id": "other-step",
                "run_id": "run-999",
                "version": "other-version",
                "checkpoint_dir": str(other_checkpoint_dir),
                "status": "completed",
            }
        ],
    )
    ranking = _ranking_path(plan_dir)
    write_rows(
        ranking,
        [
            {
                "experiment_id": "unit-experiment",
                "step_id": "other-step",
                "run_id": "run-999",
                "version": "other-version",
                "checkpoint_path": "",
                "metric": "val_other",
                "score": 0.5,
                "rank": 1,
            }
        ],
    )
    canonical = tmp_path / "run_manifest.tsv"
    events = tmp_path / "events.jsonl"
    ranking_before = ranking.read_bytes()
    canonical_before = canonical.read_bytes()
    events_before = events.read_bytes()

    with pytest.raises(ValueError, match="finite score lacks checkpoint evidence"):
        hparam_selection.select_hparam_candidates(plan_dir)

    assert ranking.read_bytes() == ranking_before
    assert canonical.read_bytes() == canonical_before
    assert events.read_bytes() == events_before


@pytest.mark.parametrize("score", ["", "not-a-number", float("nan"), float("inf"), True])
def test_hparam_select_rejects_invalid_other_step_score_without_mutation(tmp_path: Path, score):
    recipe = _hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    run = _first_run(plan_dir)
    runtime_dir = Path(run["runtime_dir"])
    runtime_dir.mkdir(parents=True, exist_ok=True)
    (runtime_dir / "run_manifest.json").write_text(json.dumps({"metrics": {"val_ahi_pearson": 0.7}}))
    other_checkpoint_dir = tmp_path / "other-checkpoints"
    merge_run_manifest(
        tmp_path,
        [
            {
                "experiment_id": "unit-experiment",
                "step_id": "other-step",
                "run_id": "run-999",
                "version": "other-version",
                "checkpoint_dir": str(other_checkpoint_dir),
                "status": "completed",
            }
        ],
    )
    ranking = _ranking_path(plan_dir)
    other_checkpoint_dir.mkdir()
    (other_checkpoint_dir / "epoch=1.ckpt").write_text("checkpoint")
    write_rows(
        ranking,
        [
            {
                "experiment_id": "unit-experiment",
                "step_id": "other-step",
                "run_id": "run-999",
                "version": "other-version",
                "checkpoint_path": str(other_checkpoint_dir / "epoch=1.ckpt"),
                "metric": "val_other",
                "score": score,
                "rank": 1,
            }
        ],
    )
    canonical = tmp_path / "run_manifest.tsv"
    events = tmp_path / "events.jsonl"
    ranking_before = ranking.read_bytes()
    canonical_before = canonical.read_bytes()
    events_before = events.read_bytes()

    with pytest.raises(ValueError, match="another step has an invalid score"):
        hparam_selection.select_hparam_candidates(plan_dir)

    assert ranking.read_bytes() == ranking_before
    assert canonical.read_bytes() == canonical_before
    assert events.read_bytes() == events_before


def test_hparam_select_rejects_unowned_preserved_checkpoint_before_writing(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    other_checkpoint_dir = tmp_path / "other-checkpoints"
    merge_run_manifest(
        tmp_path,
        [
            {
                "experiment_id": "unit-experiment",
                "step_id": "other-step",
                "run_id": "run-999",
                "version": "other-version",
                "checkpoint_dir": str(other_checkpoint_dir),
                "status": "completed",
            }
        ],
    )
    ranking = _ranking_path(plan_dir)
    write_rows(
        ranking,
        [
            {
                "experiment_id": "unit-experiment",
                "step_id": "other-step",
                "run_id": "run-999",
                "version": "other-version",
                "checkpoint_path": str(tmp_path / "foreign" / "epoch=1.ckpt"),
                "rank": 1,
            }
        ],
    )
    before = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}

    with pytest.raises(ValueError, match="checkpoint_path is outside the frozen checkpoint_dir"):
        hparam_selection.select_hparam_candidates(plan_dir)

    assert {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before


def test_hparam_select_rejects_invalid_owner_target_before_ranking_write(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    ranking = _ranking_path(plan_dir)
    matrix = tmp_path / "run_matrix.csv"
    matrix.unlink()
    matrix.hardlink_to(tmp_path / "run_manifest.tsv")
    before = {path.relative_to(tmp_path): path.read_bytes() if path.is_file() else None for path in tmp_path.rglob("*")}

    with pytest.raises(ValueError, match="Managed file is missing or aliased"):
        hparam_selection.select_hparam_candidates(plan_dir)

    assert not ranking.exists()
    assert {
        path.relative_to(tmp_path): path.read_bytes() if path.is_file() else None for path in tmp_path.rglob("*")
    } == before


def test_hparam_select_preserves_and_reranks_previous_plans_for_same_step(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path)
    plans = []
    for index, score in enumerate((0.9, 0.8), start=1):
        plan_dir = tmp_path / f"plan-{index}"
        assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
        run = _first_run(plan_dir)
        runtime_dir = Path(run["runtime_dir"])
        checkpoint_dir = Path(run["checkpoint_dir"])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = checkpoint_dir / f"epoch={index}.ckpt"
        checkpoint.write_text("checkpoint")
        (runtime_dir / "run_manifest.json").write_text(
            json.dumps(
                {
                    "metrics": {"val_ahi_pearson": score},
                    "best_model_path": str(checkpoint),
                    "epoch": index,
                }
            )
        )
        hparam_selection.select_hparam_candidates(plan_dir)
        plans.append((plan_dir, run))

    ranking = read_rows(_ranking_path(plans[-1][0]))
    assert [(row["run_id"], row["score"], row["rank"]) for row in ranking] == [
        ("run-000", "0.9", "1"),
        ("run-001", "0.8", "2"),
    ]
    canonical = read_rows(tmp_path / "run_manifest.tsv")
    assert [(row["run_id"], row["score"], row["rank"]) for row in canonical] == [
        ("run-000", "0.9", "1"),
        ("run-001", "0.8", "2"),
    ]
    selections = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text().splitlines()
        if json.loads(line)["event_type"] == "candidate_selected"
    ]
    assert selections[-1]["selected_run_id"] == "run-000"


def test_hparam_select_rebuilds_ranking_from_all_registered_plans(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path)
    plans = []
    for index, score in enumerate((0.9, 0.8), start=1):
        plan_dir = tmp_path / f"plan-{index}"
        assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
        run = _first_run(plan_dir)
        checkpoint_dir = Path(run["checkpoint_dir"])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = checkpoint_dir / f"epoch={index}.ckpt"
        checkpoint.write_text("checkpoint")
        (Path(run["runtime_dir"]) / "run_manifest.json").write_text(
            json.dumps(
                {
                    "metrics": {"val_ahi_pearson": score},
                    "best_model_path": str(checkpoint),
                    "epoch": index,
                }
            )
        )
        plans.append((plan_dir, run))

    hparam_selection.select_hparam_candidates(plans[1][0])

    ranking = read_rows(_ranking_path(plans[1][0]))
    assert [(row["run_id"], row["score"], row["rank"]) for row in ranking] == [
        ("run-000", "0.9", "1"),
        ("run-001", "0.8", "2"),
    ]
    canonical = read_rows(tmp_path / "run_manifest.tsv")
    assert [(row["run_id"], row["score"], row["rank"]) for row in canonical] == [
        ("run-000", "0.9", "1"),
        ("run-001", "0.8", "2"),
    ]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("checkpoint_sha256", "", "selection evidence is invalid"),
        ("selection_mode", "min", "selection mode differs"),
    ],
)
def test_hparam_select_validates_other_selected_steps_before_writing(
    tmp_path: Path, field: str, value: str, message: str
):
    first_run, _first_plan, second_plan = _prepare_two_hparam_steps(tmp_path)
    canonical = read_run_manifest(tmp_path)
    first_row = next(row for row in canonical if row["step_id"] == first_run["step_id"])
    first_row[field] = value
    write_rows(tmp_path / "run_manifest.tsv", canonical)
    before = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}

    with pytest.raises(ValueError, match=message):
        hparam_selection.select_hparam_candidates(second_plan)

    assert {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before


def test_hparam_select_rebuilds_missing_prior_step_ranking_from_canonical_rows(tmp_path: Path):
    first_run, first_plan, second_plan = _prepare_two_hparam_steps(tmp_path)
    ranking = _ranking_path(second_plan)
    ranking.write_text("step_id,run_id\n")

    hparam_selection.select_hparam_candidates(second_plan)

    ranking_rows = read_rows(ranking, require_managed_identity=True)
    assert {row["step_id"] for row in ranking_rows} == {first_run["step_id"], "second-tune"}
    assert experiments.experiment_status(tmp_path)["summary"]["state"] == "ready_to_finalize"
    ranking_bytes = ranking.read_bytes()

    hparam_selection.select_hparam_candidates(first_plan)

    assert ranking.read_bytes() == ranking_bytes


def test_experiment_status_rejects_reordered_hparam_ranking_rows(tmp_path: Path):
    _first_run, _first_plan, second_plan = _prepare_two_hparam_steps(tmp_path)
    ranking = hparam_selection.select_hparam_candidates(second_plan)
    ranking_rows = read_rows(ranking, require_managed_identity=True)
    write_rows(ranking, list(reversed(ranking_rows)))

    snapshot = experiments.experiment_status(tmp_path)

    assert snapshot["summary"]["state"] == "ready_to_report"
    with pytest.raises(ValueError, match="selection report is missing or differs"):
        experiments.finalize_experiment(tmp_path, tmp_path / "reports" / "hparam_selection.md")


def test_hparam_select_only_preflights_registered_plans_that_own_preserved_rankings(tmp_path: Path, monkeypatch):
    recipe = _hparam_recipe(tmp_path)
    first_plan = tmp_path / "plan-1"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(first_plan)).returncode == 0
    first_run = _first_run(first_plan)
    first_runtime = Path(first_run["runtime_dir"])
    first_checkpoint_dir = Path(first_run["checkpoint_dir"])
    first_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    first_checkpoint = first_checkpoint_dir / "epoch=1.ckpt"
    first_checkpoint.write_text("checkpoint")
    (first_runtime / "run_manifest.json").write_text(
        json.dumps(
            {
                "epoch": 1,
                "checkpoint_path": str(first_checkpoint),
                "metrics": {"val_ahi_pearson": 0.9},
            }
        )
    )
    hparam_selection.select_hparam_candidates(first_plan)

    finetune_recipe = write_finetune_recipe(tmp_path)
    finetune_payload = yaml.safe_load(finetune_recipe.read_text())
    finetune_payload["step"] = json.loads((first_plan / "plan.json").read_text())["recipe"]["step"]
    finetune_recipe = write_yaml(tmp_path / "non-hparam.yaml", finetune_payload)
    non_hparam_plan = tmp_path / "non-hparam-plan"
    assert _run("plan", "--recipe", str(finetune_recipe), "--output-dir", str(non_hparam_plan)).returncode == 0
    non_hparam_run = json.loads((non_hparam_plan / "plan.json").read_text())["runs"][0]
    merge_run_manifest(
        tmp_path,
        [
            {
                "step_id": non_hparam_run["step_id"],
                "run_id": non_hparam_run["run_id"],
                "status": "completed",
            }
        ],
    )

    recipe = _hparam_recipe(tmp_path)
    second_plan = tmp_path / "plan-2"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(second_plan)).returncode == 0
    second_run = _first_run(second_plan)
    second_runtime = Path(second_run["runtime_dir"])
    second_checkpoint_dir = Path(second_run["checkpoint_dir"])
    second_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    second_checkpoint = second_checkpoint_dir / "epoch=1.ckpt"
    second_checkpoint.write_text("checkpoint")
    (second_runtime / "run_manifest.json").write_text(
        json.dumps(
            {
                "epoch": 1,
                "checkpoint_path": str(second_checkpoint),
                "metrics": {"val_ahi_pearson": 0.8},
            }
        )
    )
    strict_reads = []
    original_read_hparam_plan = hparam_selection.artifacts.read_hparam_plan

    def tracked_read_hparam_plan(path):
        strict_reads.append(Path(path))
        return original_read_hparam_plan(path)

    monkeypatch.setattr(hparam_selection.artifacts, "read_hparam_plan", tracked_read_hparam_plan)

    hparam_selection.select_hparam_candidates(second_plan)

    assert non_hparam_plan not in strict_reads
    assert {row["run_id"] for row in read_rows(_ranking_path(second_plan))} == {"run-000", "run-002"}
    canonical = read_run_manifest(tmp_path)
    non_hparam_row = next(
        row
        for row in canonical
        if (row["step_id"], row["run_id"]) == (non_hparam_run["step_id"], non_hparam_run["run_id"])
    )
    assert all(
        non_hparam_row.get(field) in (None, "") for field in experiment_tracking.HPARAM_SELECTION_METADATA_FIELDS
    )
    assert experiments.experiment_status(tmp_path)["summary"]["state"] == "ready_to_report"
    combined = tmp_path / "combined.md"
    combined.write_text("# Combined hparam and finetune report\n")
    assert experiments.finalize_experiment(tmp_path, combined).read_text() == combined.read_text()


def test_hparam_select_rejects_same_step_non_hparam_selection_metadata_without_writing(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path)
    plan_dir = tmp_path / "hparam-plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    run = _first_run(plan_dir)
    checkpoint = Path(run["checkpoint_dir"]) / "epoch=1.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text("checkpoint")
    (Path(run["runtime_dir"]) / "run_manifest.json").write_text(
        json.dumps({"metrics": {"val_ahi_pearson": 0.9}, "best_model_path": str(checkpoint), "epoch": 1})
    )

    finetune_recipe = write_finetune_recipe(tmp_path)
    finetune_payload = yaml.safe_load(finetune_recipe.read_text())
    finetune_payload["step"] = json.loads((plan_dir / "plan.json").read_text())["recipe"]["step"]
    finetune_recipe = write_yaml(tmp_path / "non-hparam.yaml", finetune_payload)
    non_hparam_plan = tmp_path / "non-hparam-plan"
    assert _run("plan", "--recipe", str(finetune_recipe), "--output-dir", str(non_hparam_plan)).returncode == 0
    non_hparam_run = json.loads((non_hparam_plan / "plan.json").read_text())["runs"][0]
    merge_run_manifest(
        tmp_path,
        [
            {
                "step_id": non_hparam_run["step_id"],
                "run_id": non_hparam_run["run_id"],
                "selection_task": "hparam_tune",
                "status": "completed",
            }
        ],
    )
    before = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}

    with pytest.raises(ValueError, match="not owned by a registered hparam plan"):
        hparam_selection.select_hparam_candidates(plan_dir)

    assert {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before


def test_hparam_select_skips_registered_blocked_plan_after_successful_retry(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    payload["decisions"]["overwrite_policy"]["value"] = "ASK_USER"
    write_yaml(recipe, payload)
    blocked_plan = tmp_path / "blocked-plan"

    blocked = _run("plan", "--recipe", str(recipe), "--output-dir", str(blocked_plan))

    assert blocked.returncode == 2
    assert (blocked_plan / "plan.blocked.md").exists()
    assert not (blocked_plan / "plan.json").exists()
    payload["decisions"]["overwrite_policy"]["value"] = False
    write_yaml(recipe, payload)
    current_plan = tmp_path / "current-plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(current_plan)).returncode == 0
    run = _first_run(current_plan)
    checkpoint = Path(run["checkpoint_dir"]) / "epoch=1.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text("checkpoint")
    (Path(run["runtime_dir"]) / "run_manifest.json").write_text(
        json.dumps(
            {
                "metrics": {"val_ahi_pearson": 0.8},
                "best_model_path": str(checkpoint),
                "epoch": 1,
            }
        )
    )

    out = hparam_selection.select_hparam_candidates(current_plan)

    assert out == _ranking_path(current_plan)


def test_hparam_select_rejects_registered_plan_task_drift_before_writing(tmp_path: Path):
    first_recipe = _hparam_recipe(tmp_path)
    first_plan = tmp_path / "plan-1"
    assert _run("plan", "--recipe", str(first_recipe), "--output-dir", str(first_plan)).returncode == 0
    first_plan_path = first_plan / "plan.json"
    first_payload = json.loads(first_plan_path.read_text())
    first_payload["recipe"]["task"] = "finetune"
    first_plan_path.write_text(json.dumps(first_payload))

    second_recipe = _hparam_recipe(tmp_path, selection_mode="min")
    second_plan = tmp_path / "plan-2"
    assert _run("plan", "--recipe", str(second_recipe), "--output-dir", str(second_plan)).returncode == 0
    run = _first_run(second_plan)
    checkpoint = Path(run["checkpoint_dir"]) / "epoch=2.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text("checkpoint")
    (Path(run["runtime_dir"]) / "run_manifest.json").write_text(
        json.dumps(
            {
                "metrics": {"val_ahi_pearson": 0.7},
                "best_model_path": str(checkpoint),
                "epoch": 2,
            }
        )
    )
    before = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}

    with pytest.raises(ValueError, match="task|recipe.resolved.yaml"):
        hparam_selection.select_hparam_candidates(second_plan)

    assert {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before


def test_hparam_select_rejects_foreign_step_plan_before_writing(tmp_path: Path):
    current_recipe = _hparam_recipe(tmp_path)
    current_plan = tmp_path / "current-plan"
    assert _run("plan", "--recipe", str(current_recipe), "--output-dir", str(current_plan)).returncode == 0

    foreign_payload = yaml.safe_load(current_recipe.read_text())
    foreign_payload["step"] = {
        "id": "foreign-hparam-step",
        "phase": "train",
        "purpose": "Exercise a different hparam step.",
    }
    foreign_recipe = write_yaml(tmp_path / "foreign-tune.yaml", foreign_payload)
    foreign_plan = tmp_path / "foreign-plan"
    assert _run("plan", "--recipe", str(foreign_recipe), "--output-dir", str(foreign_plan)).returncode == 0

    for plan_dir, score in ((current_plan, 0.9), (foreign_plan, 0.8)):
        run = _first_run(plan_dir)
        runtime_dir = Path(run["runtime_dir"])
        runtime_dir.mkdir(parents=True, exist_ok=True)
        (runtime_dir / "run_manifest.json").write_text(json.dumps({"metrics": {"val_ahi_pearson": score}}))

    step_manifest_path = tmp_path / "steps" / "unit-hparam-tune" / "step.yaml"
    step_manifest = yaml.safe_load(step_manifest_path.read_text())
    step_manifest["plans"].append(str(foreign_plan.resolve()))
    write_yaml(step_manifest_path, step_manifest)
    ranking = _ranking_path(current_plan)
    before = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}

    with pytest.raises(ValueError, match="different step"):
        hparam_selection.select_hparam_candidates(current_plan)

    assert not ranking.exists()
    assert {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before


@pytest.mark.parametrize(("registered_split", "invoking_split"), [("val", "test"), ("test", "val")])
def test_hparam_select_rejects_selection_split_drift_across_registered_plans_before_writing(
    tmp_path: Path,
    registered_split: str,
    invoking_split: str,
):
    registered_recipe = _hparam_recipe(
        tmp_path,
        selection_metric="test_ahi_pearson",
        selection_split=registered_split,
        config_monitor="test_ahi_pearson",
    )
    registered_plan = tmp_path / "plan-1"
    assert _run("plan", "--recipe", str(registered_recipe), "--output-dir", str(registered_plan)).returncode == 0

    invoking_recipe = _hparam_recipe(
        tmp_path,
        selection_metric="test_ahi_pearson",
        selection_split=invoking_split,
        config_monitor="test_ahi_pearson",
    )
    invoking_plan = tmp_path / "plan-2"
    assert _run("plan", "--recipe", str(invoking_recipe), "--output-dir", str(invoking_plan)).returncode == 0
    before = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}

    with pytest.raises(ValueError, match="selection split differs"):
        hparam_selection.select_hparam_candidates(invoking_plan)

    assert {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before


@pytest.mark.parametrize(
    ("selection_metric", "selection_mode", "expected_message"),
    [
        ("val_loss", "max", "metric"),
        ("val_ahi_pearson", "min", "mode"),
    ],
)
@pytest.mark.parametrize("select_first_plan", [False, True])
def test_hparam_select_rejects_selection_contract_drift_across_plans_before_writing(
    tmp_path: Path,
    selection_metric: str,
    selection_mode: str,
    expected_message: str,
    select_first_plan: bool,
):
    first_recipe = _hparam_recipe(tmp_path)
    first_plan = tmp_path / "plan-1"
    assert _run("plan", "--recipe", str(first_recipe), "--output-dir", str(first_plan)).returncode == 0
    first_run = _first_run(first_plan)
    first_runtime = Path(first_run["runtime_dir"])
    first_checkpoint_dir = Path(first_run["checkpoint_dir"])
    first_checkpoint_dir.mkdir(parents=True)
    first_checkpoint = first_checkpoint_dir / "epoch=1.ckpt"
    first_checkpoint.write_text("checkpoint")
    (first_runtime / "run_manifest.json").write_text(
        json.dumps(
            {
                "metrics": {"val_ahi_pearson": 0.9},
                "best_model_path": str(first_checkpoint),
                "epoch": 1,
            }
        )
    )
    if select_first_plan:
        hparam_selection.select_hparam_candidates(first_plan)

    second_recipe = _hparam_recipe(
        tmp_path,
        selection_metric=selection_metric,
        selection_mode=selection_mode,
    )
    second_plan = tmp_path / "plan-2"
    assert _run("plan", "--recipe", str(second_recipe), "--output-dir", str(second_plan)).returncode == 0
    second_run = _first_run(second_plan)
    second_runtime = Path(second_run["runtime_dir"])
    second_checkpoint_dir = Path(second_run["checkpoint_dir"])
    second_checkpoint_dir.mkdir(parents=True)
    second_checkpoint = second_checkpoint_dir / "epoch=2.ckpt"
    second_checkpoint.write_text("checkpoint")
    (second_runtime / "run_manifest.json").write_text(
        json.dumps(
            {
                "metrics": {selection_metric: 0.8},
                "best_model_path": str(second_checkpoint),
                "epoch": 2,
            }
        )
    )
    before = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}

    with pytest.raises(ValueError, match=expected_message):
        hparam_selection.select_hparam_candidates(second_plan)

    assert {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before


def test_hparam_select_preflights_ranking_before_read_or_runtime_scan(tmp_path: Path, monkeypatch):
    recipe = _hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    ranking = _ranking_path(plan_dir)
    ranking.parent.mkdir(parents=True, exist_ok=True)
    outside = tmp_path / "outside.csv"
    outside.write_text("sentinel\n")
    ranking.hardlink_to(outside)
    canonical_before = (tmp_path / "run_manifest.tsv").read_bytes()
    events_before = (tmp_path / "events.jsonl").read_bytes()
    ranking_reads = []
    runtime_reads = []
    original_read_rows = hparam_selection.read_rows

    def tracked_read_rows(path, **kwargs):
        if Path(path) == ranking:
            ranking_reads.append(Path(path))
            raise AssertionError("ranking read before topology preflight")
        return original_read_rows(path, **kwargs)

    monkeypatch.setattr(hparam_selection, "read_rows", tracked_read_rows)
    monkeypatch.setattr(
        hparam_selection.artifacts,
        "find_run_manifest",
        lambda _run: runtime_reads.append("runtime") or None,
    )

    with pytest.raises(ValueError, match="Managed output"):
        hparam_selection.select_hparam_candidates(plan_dir)

    assert ranking_reads == []
    assert runtime_reads == []
    assert (tmp_path / "run_manifest.tsv").read_bytes() == canonical_before
    assert (tmp_path / "events.jsonl").read_bytes() == events_before
    assert outside.read_text() == "sentinel\n"


def test_hparam_select_rejects_header_only_legacy_ranking(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    ranking = _ranking_path(plan_dir)
    ranking.parent.mkdir(parents=True, exist_ok=True)
    ranking.write_text("trial_id,rank\n")

    with pytest.raises(ValueError, match="Historical trial_id fields"):
        hparam_selection.select_hparam_candidates(plan_dir)

    assert ranking.read_text() == "trial_id,rank\n"


def test_hparam_checkpoint_scan_rejects_header_only_legacy_ranking(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    ranking = plan_dir / "checkpoint_ranking.csv"
    ranking.write_text("trial_id,epoch\n")

    with pytest.raises(ValueError, match="Historical trial_id fields"):
        hparam_selection.scan_hparam_checkpoints(plan_dir, "val_auroc", "max")

    assert ranking.read_text() == "trial_id,epoch\n"


def test_hparam_checkpoint_scan_rejects_symlink_output_before_runtime_scan(tmp_path: Path, monkeypatch):
    recipe = _hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    ranking = plan_dir / "checkpoint_ranking.csv"
    outside = tmp_path / "outside.csv"
    outside.write_text("step_id,run_id\n")
    ranking.symlink_to(outside)
    runtime_reads = []
    monkeypatch.setattr(
        hparam_selection.artifacts,
        "find_run_manifest",
        lambda _run: runtime_reads.append("runtime") or None,
    )

    with pytest.raises(ValueError, match="Managed output"):
        hparam_selection.scan_hparam_checkpoints(plan_dir, "val_auroc", "max")

    assert runtime_reads == []
    assert outside.read_text() == "step_id,run_id\n"


@pytest.mark.parametrize("existing_fault", ["unmanaged", "frozen_drift"])
def test_hparam_checkpoint_scan_validates_existing_ranking_before_runtime_scan(
    tmp_path: Path, monkeypatch, existing_fault: str
):
    recipe = _hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    run = _first_run(plan_dir)
    ranking = plan_dir / "checkpoint_ranking.csv"
    if existing_fault == "unmanaged":
        row = {
            "experiment_id": "foreign-experiment",
            "step_id": "foreign-step",
            "run_id": "run-999",
            "version": "foreign-version",
            "config": str(tmp_path / "foreign.yaml"),
            "checkpoint_path": str(tmp_path / "foreign.ckpt"),
        }
    else:
        row = {
            "experiment_id": run["experiment_id"],
            "step_id": run["step_id"],
            "run_id": run["run_id"],
            "version": "drifted-version",
            "config": run["config"],
            "checkpoint_path": str(tmp_path / "epoch=1.ckpt"),
        }
    write_rows(ranking, [row])
    before = ranking.read_bytes()
    runtime_reads = []
    monkeypatch.setattr(
        hparam_selection.artifacts,
        "find_run_manifest",
        lambda _run: runtime_reads.append("runtime") or None,
    )

    with pytest.raises(ValueError, match="outside the canonical manifest|Frozen run field differs"):
        hparam_selection.scan_hparam_checkpoints(plan_dir, "val_auroc", "max")

    assert runtime_reads == []
    assert ranking.read_bytes() == before
