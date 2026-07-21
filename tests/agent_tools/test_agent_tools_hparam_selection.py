from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys

from agent_tool_test_helpers import write_finetune_recipe, write_yaml
import pytest
import yaml

from agent_tools import hparam_selection, run_artifacts
from agent_tools.experiment_workspace import merge_run_manifest
from agent_tools.manifests import read_rows, write_rows
from agent_tools.models import REPO_ROOT

_RUNTIME_COMMIT = subprocess.run(
    ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, text=True, capture_output=True
).stdout.strip()


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "agent_tools", *args], text=True, capture_output=True)


def _hparam_recipe(
    tmp_path: Path,
    *,
    execution: dict | None = None,
    selection_metric: str = "val_ahi_pearson",
    selection_mode: str = "max",
) -> Path:
    base = write_finetune_recipe(tmp_path)
    base_payload = yaml.safe_load(base.read_text())
    config_path = Path(base_payload["inputs"]["config"])
    config_payload = yaml.safe_load(config_path.read_text())
    config_payload["finetune"]["task"]["monitor"] = selection_metric
    config_payload["finetune"]["task"]["monitor_mod"] = selection_mode
    write_yaml(config_path, config_payload)
    base_payload["evaluation_policy"]["selection_metric"] = selection_metric
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
                "max_runs": 1,
                "parameters": {"runtime.lr": [1e-6]},
            },
            "execution": execution_payload,
            "evaluation_policy": {
                "selection_metric": selection_metric,
                "selection_mode": selection_mode,
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


def test_hparam_select_preserves_canonical_other_step_ranking(tmp_path: Path):
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
    assert any(row["step_id"] == "other-step" and row["run_id"] == "run-999" for row in rows)


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
