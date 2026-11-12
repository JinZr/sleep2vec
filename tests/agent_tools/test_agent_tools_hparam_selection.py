from __future__ import annotations

import hashlib
import json
from pathlib import Path

from agent_tool_test_helpers import (
    call_while_run_lock_holder_commits,
    prepare_hparam_plan_fixture,
    read_table,
    run_cli,
    tree_files,
    write_finetune_recipe,
    write_local_hparam_recipe,
    write_yaml,
)
import pytest
import yaml

from agent_tools import experiment_tracking, experiments, hparam_selection, run_artifacts
from agent_tools.experiment_workspace import merge_run_manifest, read_run_manifest
from agent_tools.manifests import read_rows, write_rows


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
    first_recipe = write_local_hparam_recipe(tmp_path)
    first_plan = tmp_path / "plan-1"
    assert run_cli("plan", "--recipe", str(first_recipe), "--output-dir", str(first_plan)).returncode == 0
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

    second_recipe = write_local_hparam_recipe(tmp_path)
    second_payload = yaml.safe_load(second_recipe.read_text())
    second_payload["step"] = {
        "id": "second-tune",
        "phase": "train",
        "purpose": "Exercise a second hparam selection step.",
    }
    write_yaml(second_recipe, second_payload)
    second_plan = tmp_path / "plan-2"
    assert run_cli("plan", "--recipe", str(second_recipe), "--output-dir", str(second_plan)).returncode == 0
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


def test_hparam_select_uses_fixed_epoch_checkpoint_not_best_alias(tmp_path: Path):
    recipe = write_local_hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    prepare_hparam_plan_fixture(recipe, plan_dir)
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

    result = run_cli(
        "hparam-select",
        "--run-dir",
        str(plan_dir),
        "--metric",
        "val_ahi_pearson",
        "--mode",
        "max",
    )

    assert result.returncode == 0, result.stderr
    rows = read_table(_ranking_path(plan_dir))
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


@pytest.mark.parametrize(
    ("metric", "mode"),
    [("val_loss", None), (None, "min")],
    ids=["metric", "mode"],
)
def test_hparam_select_rejects_explicit_selection_contract_drift_without_writing(
    tmp_path: Path,
    metric: str | None,
    mode: str | None,
):
    recipe = write_local_hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    prepare_hparam_plan_fixture(recipe, plan_dir)
    before = tree_files(tmp_path)

    with pytest.raises(
        ValueError,
        match=r"^hparam-select must use the selection metric and mode frozen in the recipe\.$",
    ):
        hparam_selection.select_hparam_candidates(plan_dir, metric=metric, mode=mode)

    assert tree_files(tmp_path) == before


@pytest.mark.parametrize("reader", ["select_plan", "select_rows", "commit_rows"])
def test_hparam_selection_reads_workspace_state_only_under_run_lock(tmp_path: Path, monkeypatch, reader: str):
    recipe = write_local_hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    prepare_hparam_plan_fixture(recipe, plan_dir)
    run = _first_run(plan_dir)
    checkpoint = Path(run["checkpoint_dir"]) / "epoch=1.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text("checkpoint")
    Path(run["runtime_dir"], "run_manifest.json").write_text(
        json.dumps({"metrics": {"val_ahi_pearson": 0.9}, "best_model_path": str(checkpoint), "epoch": 1})
    )
    if reader.startswith("commit"):
        hparam_selection.select_hparam_candidates(plan_dir)
    # The harness races only the first manifest read, so later reads are targeted by skipping earlier ones.
    if reader.endswith("_rows"):
        plan = run_artifacts.read_hparam_plan(plan_dir)
        monkeypatch.setattr(hparam_selection.hparam_runtime, "read_hparam_plan_under_run_lock", lambda _path: plan)
    if reader == "commit_rows":
        selection = hparam_selection._build_hparam_selection(plan_dir)
        monkeypatch.setattr(hparam_selection, "merge_run_manifest", lambda *_args, **_kwargs: None)
    readers = {
        "select": lambda: hparam_selection.select_hparam_candidates(plan_dir),
        "commit": lambda: hparam_selection._commit_hparam_selection(selection),
    }

    result = call_while_run_lock_holder_commits(monkeypatch, tmp_path, readers[reader.rsplit("_", 1)[0]])

    assert result == _ranking_path(plan_dir)
    assert read_table(result)[0]["run_id"] == run["run_id"]


def test_hparam_select_rejects_completed_experiment_without_mutation(tmp_path: Path):
    recipe = write_local_hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    prepare_hparam_plan_fixture(recipe, plan_dir)
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
    before = tree_files(tmp_path)

    with pytest.raises(ValueError, match="completed"):
        hparam_selection.select_hparam_candidates(plan_dir)

    assert tree_files(tmp_path) == before


def test_hparam_select_clears_stale_result_evidence_from_unscored_candidates(tmp_path: Path):
    recipe = write_local_hparam_recipe(tmp_path, max_runs=2)
    plan_dir = tmp_path / "plan"
    prepare_hparam_plan_fixture(recipe, plan_dir)
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
    recipe = write_local_hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    assert run_cli("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    manifest_path = tmp_path / "experiment.yaml"
    manifest = yaml.safe_load(manifest_path.read_text())
    manifest["experiment"][field] = value
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False))
    before = tree_files(tmp_path)

    with pytest.raises(ValueError, match="Invalid active experiment owner"):
        hparam_selection.select_hparam_candidates(plan_dir)

    assert tree_files(tmp_path) == before


@pytest.mark.parametrize("ranking_present", [True, False])
def test_hparam_select_preserves_frozen_val_selection_when_runtime_evidence_drifts(
    tmp_path: Path, ranking_present: bool
):
    recipe = write_local_hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    assert run_cli("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
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
    before = tree_files(tmp_path)

    with pytest.raises(ValueError, match="Frozen canonical hparam selection"):
        hparam_selection.select_hparam_candidates(plan_dir)

    assert tree_files(tmp_path) == before


@pytest.mark.parametrize("field", ["run_manifest", "status"])
@pytest.mark.parametrize("mutation", ["tamper", "remove"])
def test_experiment_status_rejects_val_ranking_optional_provenance_drift(tmp_path: Path, field: str, mutation: str):
    recipe = write_local_hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    assert run_cli("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
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
    recipe = write_local_hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    prepare_hparam_plan_fixture(recipe, plan_dir)
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


def test_hparam_select_reads_the_user_materialized_effective_metric(tmp_path: Path):
    recipe = write_local_hparam_recipe(tmp_path, selection_metric="val_effective")
    payload = yaml.safe_load(recipe.read_text())
    payload["evaluation_policy"]["selection_metric"] = "val_stale"
    write_yaml(recipe, payload)
    decisions = write_yaml(
        tmp_path / "decisions.yaml",
        {"decisions": {"selection_metric": {"value": "val_effective", "source": "explicit_user"}}},
    )
    plan_dir = tmp_path / "plan"
    result = run_cli(
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

    row = read_table(ranking)[0]
    assert row["metric"] == "val_effective"
    assert row["score"] == "0.7"


def test_hparam_select_preserves_zero_padded_epoch_checkpoint(tmp_path: Path):
    recipe = write_local_hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    assert run_cli("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
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

    result = run_cli(
        "hparam-select",
        "--run-dir",
        str(plan_dir),
        "--metric",
        "val_ahi_pearson",
        "--mode",
        "max",
    )

    assert result.returncode == 0, result.stderr
    rows = read_table(_ranking_path(plan_dir))
    assert rows[0]["checkpoint_path"] == str(fixed)


@pytest.mark.parametrize("score", [None, "not-a-number", float("nan"), float("inf"), True])
def test_hparam_select_fails_without_any_valid_score_and_preserves_state(tmp_path: Path, score):
    recipe = write_local_hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    assert run_cli("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
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
    recipe = write_local_hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    assert run_cli("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
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

    result = run_cli("hparam-select", "--run-dir", str(plan_dir))

    assert result.returncode == 1
    assert "No valid val_ahi_pearson scores" in result.stderr
    assert not _ranking_path(plan_dir).exists()


def test_hparam_select_requires_terminal_canonical_runs_for_val_selection(tmp_path: Path):
    recipe = write_local_hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    assert run_cli("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
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


def test_hparam_select_does_not_scan_unmanaged_runtime_directories(tmp_path: Path):
    recipe = write_local_hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    assert run_cli("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
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

    result = run_cli("hparam-select", "--run-dir", str(plan_dir))

    assert result.returncode == 0, result.stderr
    row = read_table(_ranking_path(plan_dir))[0]
    assert row["score"] == "0.7"
    assert row["run_manifest"] == str(managed_manifest)
    assert row["checkpoint_path"] == str(managed_checkpoint)


def test_hparam_select_requires_checkpoint_evidence_for_finite_score(tmp_path: Path):
    recipe = write_local_hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    assert run_cli("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    run = _first_run(plan_dir)
    runtime_dir = Path(run["runtime_dir"])
    runtime_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(run["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "epoch=1.ckpt").write_text("unbound checkpoint")
    (runtime_dir / "run_manifest.json").write_text(json.dumps({"metrics": {"val_ahi_pearson": 0.7}}))

    with pytest.raises(ValueError, match="Valid val_ahi_pearson score lacks resolvable checkpoint evidence"):
        hparam_selection.select_hparam_candidates(plan_dir)

    assert not _ranking_path(plan_dir).exists()
    assert not any(
        json.loads(line)["event_type"] == "candidate_selected"
        for line in (tmp_path / "events.jsonl").read_text().splitlines()
    )


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

    assert run_artifacts.fixed_checkpoint_path_from_names({}, checkpoint_dir, names) == ""


def test_fixed_checkpoint_accepts_same_epoch_best_only_locally_and_remotely(tmp_path: Path):
    checkpoint_dir = tmp_path / "managed" / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    checkpoint = checkpoint_dir / "best-epoch=03.ckpt"
    checkpoint.write_text("checkpoint")
    manifest = {"best_model_path": str(checkpoint), "epoch": 3}

    assert run_artifacts.fixed_checkpoint_path_from_names(manifest, checkpoint_dir, [checkpoint.name]) == str(
        checkpoint
    )


@pytest.mark.parametrize("status", ["completed", "skipped_test", "running", "failed", None])
def test_fixed_checkpoint_binds_best_only_alias_from_successful_training_manifest(tmp_path: Path, status):
    """Training manifests carry no epoch; early stopping can leave best-epoch= as the only epoch checkpoint."""
    checkpoint_dir = tmp_path / "managed" / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    checkpoint = checkpoint_dir / "best-epoch=45.ckpt"
    checkpoint.write_text("checkpoint")
    (checkpoint_dir / "last.ckpt").write_text("last")
    names = [checkpoint.name, "last.ckpt"]
    manifest = {"status": status, "best_model_path": str(checkpoint), "best_model_score": 8.8}
    expected = str(checkpoint) if status in {"completed", "skipped_test"} else ""

    assert run_artifacts.fixed_checkpoint_path_from_names(manifest, checkpoint_dir, names) == expected
    assert run_artifacts.fixed_checkpoint_path_from_names(manifest, checkpoint_dir, ["last.ckpt"]) == ""

    fixed = checkpoint_dir / "epoch=45.ckpt"
    fixed.write_text("fixed")
    assert run_artifacts.fixed_checkpoint_path_from_names(manifest, checkpoint_dir, [*names, fixed.name]) == str(fixed)


def test_fixed_checkpoint_rejects_fractional_manifest_epoch_locally_and_remotely(tmp_path: Path):
    checkpoint_dir = tmp_path / "managed" / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    checkpoint = checkpoint_dir / "epoch=2.ckpt"
    checkpoint.write_text("checkpoint")
    manifest = {"epoch": 2.5}

    assert run_artifacts.fixed_checkpoint_path_from_names(manifest, checkpoint_dir, [checkpoint.name]) == ""


@pytest.mark.parametrize("value", [2, 2.0, "2", "2.0"])
def test_epoch_number_accepts_integer_values(value):
    assert run_artifacts.epoch_number(value) == 2


@pytest.mark.parametrize("value", [2.5, "2.5", float("nan"), float("inf"), True, "not-a-number"])
def test_epoch_number_rejects_non_integer_values(value):
    assert run_artifacts.epoch_number(value) is None


def test_hparam_select_rejects_historical_workspace_before_writing_ranking(tmp_path: Path):
    recipe = write_local_hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    assert run_cli("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    (tmp_path / "run_manifest.tsv").write_text("trial_id\tstatus\ntrial_000\tfinished\n")
    ranking = _ranking_path(plan_dir)

    with pytest.raises(ValueError, match="read-only"):
        hparam_selection.select_hparam_candidates(plan_dir)

    assert not ranking.exists()


@pytest.mark.parametrize("manifest_state", ["missing", "current_key_absent"])
def test_hparam_select_requires_registered_run_manifest_before_writing(tmp_path: Path, manifest_state: str):
    recipe = write_local_hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    assert run_cli("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    manifest = tmp_path / "run_manifest.tsv"
    if manifest_state == "missing":
        manifest.unlink()
    else:
        manifest.write_text("experiment_id\tstep_id\trun_id\tstatus\nunit-experiment\tother-step\trun-000\tplanned\n")
    ranking = _ranking_path(plan_dir)
    before = tree_files(tmp_path)

    with pytest.raises((FileNotFoundError, ValueError), match="missing"):
        hparam_selection.select_hparam_candidates(plan_dir)

    after = tree_files(tmp_path)
    assert after == before
    assert not ranking.exists()


def test_hparam_select_rejects_unmanaged_existing_ranking_before_preserving_other_steps(tmp_path: Path):
    recipe = write_local_hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    assert run_cli("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
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
    recipe = write_local_hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    assert run_cli("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
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


def test_hparam_select_rejects_empty_preserved_checkpoint_before_writing(tmp_path: Path):
    recipe = write_local_hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    assert run_cli("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
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
    recipe = write_local_hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    assert run_cli("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
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
    recipe = write_local_hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    assert run_cli("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
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
    before = tree_files(tmp_path)

    with pytest.raises(ValueError, match="checkpoint_path is outside the frozen checkpoint_dir"):
        hparam_selection.select_hparam_candidates(plan_dir)

    assert tree_files(tmp_path) == before


def test_hparam_select_preserves_and_reranks_previous_plans_for_same_step(tmp_path: Path):
    recipe = write_local_hparam_recipe(tmp_path)
    plans = []
    for index, score in enumerate((0.9, 0.8), start=1):
        plan_dir = tmp_path / f"plan-{index}"
        assert run_cli("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
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
    recipe = write_local_hparam_recipe(tmp_path)
    plans = []
    for index, score in enumerate((0.9, 0.8), start=1):
        plan_dir = tmp_path / f"plan-{index}"
        assert run_cli("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
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
    before = tree_files(tmp_path)

    with pytest.raises(ValueError, match=message):
        hparam_selection.select_hparam_candidates(second_plan)

    assert tree_files(tmp_path) == before


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
    recipe = write_local_hparam_recipe(tmp_path)
    first_plan = tmp_path / "plan-1"
    assert run_cli("plan", "--recipe", str(recipe), "--output-dir", str(first_plan)).returncode == 0
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
    assert run_cli("plan", "--recipe", str(finetune_recipe), "--output-dir", str(non_hparam_plan)).returncode == 0
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

    recipe = write_local_hparam_recipe(tmp_path)
    second_plan = tmp_path / "plan-2"
    assert run_cli("plan", "--recipe", str(recipe), "--output-dir", str(second_plan)).returncode == 0
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

    def tracked_read_hparam_plan(path, **kwargs):
        strict_reads.append(Path(path))
        return original_read_hparam_plan(path, **kwargs)

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
    recipe = write_local_hparam_recipe(tmp_path)
    plan_dir = tmp_path / "hparam-plan"
    assert run_cli("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
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
    assert run_cli("plan", "--recipe", str(finetune_recipe), "--output-dir", str(non_hparam_plan)).returncode == 0
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
    before = tree_files(tmp_path)

    with pytest.raises(ValueError, match="not owned by a registered hparam plan"):
        hparam_selection.select_hparam_candidates(plan_dir)

    assert tree_files(tmp_path) == before


def test_hparam_select_skips_registered_blocked_plan_after_successful_retry(tmp_path: Path):
    recipe = write_local_hparam_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    label_name = payload["decisions"]["label_name"]["value"]
    payload["decisions"]["label_name"]["value"] = "ASK_USER"
    write_yaml(recipe, payload)
    blocked_plan = tmp_path / "blocked-plan"

    blocked = run_cli("plan", "--recipe", str(recipe), "--output-dir", str(blocked_plan))

    assert blocked.returncode == 2
    assert (blocked_plan / "plan.blocked.md").exists()
    assert not (blocked_plan / "plan.json").exists()
    payload["decisions"]["label_name"]["value"] = label_name
    write_yaml(recipe, payload)
    current_plan = tmp_path / "current-plan"
    assert run_cli("plan", "--recipe", str(recipe), "--output-dir", str(current_plan)).returncode == 0
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
    first_recipe = write_local_hparam_recipe(tmp_path)
    first_plan = tmp_path / "plan-1"
    assert run_cli("plan", "--recipe", str(first_recipe), "--output-dir", str(first_plan)).returncode == 0
    first_plan_path = first_plan / "plan.json"
    first_payload = json.loads(first_plan_path.read_text())
    first_payload["recipe"]["task"] = "finetune"
    first_plan_path.write_text(json.dumps(first_payload))

    second_recipe = write_local_hparam_recipe(tmp_path, selection_mode="min")
    second_plan = tmp_path / "plan-2"
    assert run_cli("plan", "--recipe", str(second_recipe), "--output-dir", str(second_plan)).returncode == 0
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
    before = tree_files(tmp_path)

    with pytest.raises(ValueError, match="task|recipe.resolved.yaml"):
        hparam_selection.select_hparam_candidates(second_plan)

    assert tree_files(tmp_path) == before


def test_hparam_select_rejects_foreign_step_plan_before_writing(tmp_path: Path):
    current_recipe = write_local_hparam_recipe(tmp_path)
    current_plan = tmp_path / "current-plan"
    assert run_cli("plan", "--recipe", str(current_recipe), "--output-dir", str(current_plan)).returncode == 0

    foreign_payload = yaml.safe_load(current_recipe.read_text())
    foreign_payload["step"] = {
        "id": "foreign-hparam-step",
        "phase": "train",
        "purpose": "Exercise a different hparam step.",
    }
    foreign_recipe = write_yaml(tmp_path / "foreign-tune.yaml", foreign_payload)
    foreign_plan = tmp_path / "foreign-plan"
    assert run_cli("plan", "--recipe", str(foreign_recipe), "--output-dir", str(foreign_plan)).returncode == 0

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
    before = tree_files(tmp_path)

    with pytest.raises(ValueError, match="different step"):
        hparam_selection.select_hparam_candidates(current_plan)

    assert not ranking.exists()
    assert tree_files(tmp_path) == before


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
    first_recipe = write_local_hparam_recipe(tmp_path)
    first_plan = tmp_path / "plan-1"
    assert run_cli("plan", "--recipe", str(first_recipe), "--output-dir", str(first_plan)).returncode == 0
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

    second_recipe = write_local_hparam_recipe(
        tmp_path,
        selection_metric=selection_metric,
        selection_mode=selection_mode,
    )
    second_plan = tmp_path / "plan-2"
    assert run_cli("plan", "--recipe", str(second_recipe), "--output-dir", str(second_plan)).returncode == 0
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
    before = tree_files(tmp_path)

    with pytest.raises(ValueError, match=expected_message):
        hparam_selection.select_hparam_candidates(second_plan)

    assert tree_files(tmp_path) == before


def test_hparam_select_rejects_header_only_legacy_ranking(tmp_path: Path):
    recipe = write_local_hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    assert run_cli("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    ranking = _ranking_path(plan_dir)
    ranking.parent.mkdir(parents=True, exist_ok=True)
    ranking.write_text("trial_id,rank\n")

    with pytest.raises(ValueError, match="Historical trial_id fields"):
        hparam_selection.select_hparam_candidates(plan_dir)

    assert ranking.read_text() == "trial_id,rank\n"


@pytest.mark.parametrize("status", ["completed", "skipped_test", "running", "failed", None])
def test_fixed_best_checkpoint_requires_successful_manifest(tmp_path: Path, status):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    checkpoint = checkpoint_dir / "best.ckpt"
    checkpoint.write_text("selected weights")
    manifest = {"status": status, "best_model_path": str(checkpoint)}
    expected = str(checkpoint) if status in {"completed", "skipped_test"} else ""
    assert run_artifacts.fixed_checkpoint_path_from_names(manifest, checkpoint_dir, [checkpoint.name]) == expected
    assert run_artifacts.fixed_checkpoint_path_from_names(manifest, checkpoint_dir, []) == ""


def test_hparam_select_binds_terminal_best_checkpoint_and_monitor_score(tmp_path: Path):
    recipe = write_local_hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    assert run_cli("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    run = _first_run(plan_dir)
    checkpoint = Path(run["checkpoint_dir"]) / "best.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text("selected weights")
    manifest = {
        "status": "completed",
        "monitor": "val_ahi_pearson",
        "best_model_score": 0.72,
        "best_model_path": str(checkpoint),
        "metrics": {"test_ahi_pearson": 0.81},
    }
    (Path(run["runtime_dir"]) / "run_manifest.json").write_text(json.dumps(manifest))
    ranking = hparam_selection.select_hparam_candidates(plan_dir)
    row = read_table(ranking)[0]
    assert row["score"] == "0.72"
    assert row["checkpoint_path"] == str(checkpoint)
    assert row["checkpoint_sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()


def test_hparam_select_binds_early_stopped_best_alias_without_manifest_epoch(tmp_path: Path):
    recipe = write_local_hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    assert run_cli("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    run = _first_run(plan_dir)
    checkpoint = Path(run["checkpoint_dir"]) / "best-epoch=13.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text("selected weights")
    (checkpoint.parent / "best.ckpt").write_text("selected weights")
    (checkpoint.parent / "last.ckpt").write_text("last weights")
    manifest = {
        "status": "completed",
        "monitor": "val_ahi_pearson",
        "best_model_score": 0.72,
        "best_model_path": str(checkpoint),
        "metrics": {"test_ahi_pearson": 0.81},
    }
    (Path(run["runtime_dir"]) / "run_manifest.json").write_text(json.dumps(manifest))
    ranking = hparam_selection.select_hparam_candidates(plan_dir)
    row = read_table(ranking)[0]
    assert row["score"] == "0.72"
    assert row["checkpoint_path"] == str(checkpoint)
    assert row["checkpoint_sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
