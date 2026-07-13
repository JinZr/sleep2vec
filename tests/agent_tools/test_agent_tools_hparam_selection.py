from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys

from agent_tool_test_helpers import write_finetune_recipe, write_yaml
import pytest

from agent_tools import hparam_selection, run_artifacts
from agent_tools.experiment_workspace import merge_run_manifest
from agent_tools.manifests import read_rows, write_rows


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "agent_tools", *args], text=True, capture_output=True)


def _hparam_recipe(tmp_path: Path, *, execution: dict | None = None) -> Path:
    base = write_finetune_recipe(tmp_path)
    return write_yaml(
        tmp_path / "tune.yaml",
        {
            "name": "unit_hparam",
            "task": "hparam_tune",
            "variant": "sleep2vec",
            "base_recipe": str(base),
            "search": {
                "method": "grid",
                "max_runs": 1,
                "parameters": {"runtime.lr": [1e-6]},
            },
            "execution": execution if execution is not None else {"workdir": str(tmp_path)},
            "evaluation_policy": {
                "selection_metric": "val_ahi_pearson",
                "selection_mode": "max",
                "selection_split": "val",
                "external_test_locked": True,
                "test_after_fit": False,
                "final_eval_split": "test",
                "final_test_unlocked": False,
                "require_manual_unlock_for_final_test": True,
            },
            "decisions": {
                "task": {"value": "hparam_tune", "source": "explicit_recipe"},
                "label_name": {"value": "ahi", "source": "explicit_recipe"},
                "external_test_locked": {"value": True, "source": "explicit_recipe"},
                "train_val_test_policy": {
                    "value": "select on val",
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
    return json.loads((plan_dir / "plan.json").read_text())["runs"][0]


def _ranking_path(plan_dir: Path) -> Path:
    recipe = json.loads((plan_dir / "plan.json").read_text())["recipe"]
    return Path(recipe["experiment"]["root"]) / "reports" / "ranking.csv"


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


def test_hparam_select_uses_canonical_status_not_runtime_manifest_status(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    run = _first_run(plan_dir)
    run_dir = Path(run["runtime_dir"])
    run_dir.mkdir(parents=True)
    (run_dir / "run_manifest.json").write_text(json.dumps({"status": "completed", "metrics": {"val_ahi_pearson": 0.7}}))
    canonical = read_rows(tmp_path / "run_manifest.tsv")
    canonical[0]["status"] = "failed"
    write_rows(tmp_path / "run_manifest.tsv", canonical)

    result = _run("hparam-select", "--run-dir", str(plan_dir))

    assert result.returncode == 0, result.stderr
    assert _read_table(_ranking_path(plan_dir))[0]["status"] == "failed"


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


def test_hparam_checkpoint_scan_empty_output_remains_readable(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0

    first = hparam_selection.scan_hparam_checkpoints(plan_dir, "val_auroc", "max")
    second = hparam_selection.scan_hparam_checkpoints(plan_dir, "val_auroc", "max")

    assert first == second
    assert first.read_text() == "step_id,run_id\n"


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

    result = _run("hparam-select", "--run-dir", str(plan_dir))

    assert result.returncode == 0, result.stderr
    row = _read_table(_ranking_path(plan_dir))[0]
    assert row["score"] == ""
    assert row["run_manifest"] == ""
    assert row["checkpoint_path"] == ""


def test_fixed_checkpoint_does_not_escape_frozen_checkpoint_dir(tmp_path: Path):
    checkpoint_dir = tmp_path / "managed" / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    unmanaged = tmp_path / "unmanaged" / "epoch=07.ckpt"
    unmanaged.parent.mkdir()
    unmanaged.write_text("unmanaged")

    path = run_artifacts.fixed_checkpoint_path({"best_model_path": str(unmanaged), "epoch": 7}, checkpoint_dir)

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


def test_hparam_select_preserves_canonical_other_step_ranking(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    other_config = tmp_path / "other.yaml"
    merge_run_manifest(
        tmp_path,
        [
            {
                "experiment_id": "unit-experiment",
                "step_id": "other-step",
                "run_id": "run-999",
                "version": "other-version",
                "config": str(other_config),
                "status": "completed",
            }
        ],
    )
    ranking = _ranking_path(plan_dir)
    other_checkpoint = tmp_path / "other.ckpt"
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
                "rank": 1,
            }
        ],
    )

    hparam_selection.select_hparam_candidates(plan_dir)

    rows = read_rows(ranking)
    assert any(row["step_id"] == "other-step" and row["run_id"] == "run-999" for row in rows)


@pytest.mark.parametrize("target_kind", ["directory", "hardlink"])
def test_hparam_select_rejects_invalid_owner_target_before_ranking_write(tmp_path: Path, target_kind: str):
    recipe = _hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    ranking = _ranking_path(plan_dir)
    matrix = tmp_path / "run_matrix.csv"
    matrix.unlink()
    if target_kind == "directory":
        matrix.mkdir()
    else:
        matrix.hardlink_to(tmp_path / "run_manifest.tsv")
    before = {path.relative_to(tmp_path): path.read_bytes() if path.is_file() else None for path in tmp_path.rglob("*")}

    with pytest.raises(ValueError, match="Managed output"):
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
