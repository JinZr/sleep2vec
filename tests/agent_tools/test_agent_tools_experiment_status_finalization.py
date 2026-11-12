from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import threading

from agent_tool_test_helpers import (
    add_plan,
    init_status_workspace,
    read_manifest_rows,
    record_hparam_selection,
    sha256_of,
    workspace_files,
    write_public_hparam_recipe,
)
import pytest
import yaml

from agent_tools import experiment_io, experiment_tracking, experiments, plans
from agent_tools.experiment_workspace import FROZEN_RUN_FIELDS, managed_run_parameters, merge_run_manifest
from agent_tools.manifests import read_rows, write_rows

pytestmark = pytest.mark.usefixtures("stub_execution_target")


def test_experiment_finalize_rejects_unmaterialized_step(tmp_path):
    root = tmp_path / "experiment"
    init_status_workspace(root)
    add_plan(root, step_id="ordinary", status="completed")
    step_dir = root / "steps" / "evaluate"
    step_dir.mkdir(parents=True)
    (step_dir / "step.yaml").write_text(
        yaml.safe_dump(
            {
                "step": {"id": "evaluate", "phase": "evaluate", "purpose": "Run external evaluation."},
                "experiment_id": "status-unit",
                "recipe_path": "",
                "plans": [],
            },
            sort_keys=False,
        )
    )
    report = tmp_path / "final.md"
    report.write_text("# Final\n")
    before = workspace_files(root)

    with pytest.raises(ValueError, match="incomplete canonical steps: unmaterialized_step"):
        experiments.finalize_experiment(root, report)

    assert workspace_files(root) == before


@pytest.mark.parametrize("registration_mutation", ["empty", "missing"])
def test_experiment_finalize_rejects_canonical_runs_without_registered_plan(tmp_path, registration_mutation):
    root = tmp_path / "experiment"
    init_status_workspace(root)
    add_plan(root, step_id="tune", task="hparam_tune", status="completed")
    step_path = root / "steps" / "tune" / "step.yaml"
    if registration_mutation == "empty":
        step_manifest = yaml.safe_load(step_path.read_text())
        step_manifest["plans"] = []
        step_path.write_text(yaml.safe_dump(step_manifest, sort_keys=False))
    else:
        step_path.unlink()
    report = tmp_path / "final.md"
    report.write_text("# Final\n")
    before = workspace_files(root)

    with pytest.raises(
        ValueError, match="plans differ from canonical run keys|unregistered steps|Managed file is missing"
    ):
        experiments.finalize_experiment(root, report)

    assert workspace_files(root) == before
    assert not (root / "reports" / "final.md").exists()


@pytest.mark.parametrize("modern_evidence", ["managed_parameter", "selection_metadata"])
def test_experiment_finalize_does_not_downgrade_managed_hparam_evidence_to_legacy(tmp_path, modern_evidence):
    root = tmp_path / "experiment"
    recipe = write_public_hparam_recipe(root, {"runtime.lr": [1e-6]})
    plan_dir = root / "plans" / "tune"
    assert plans.build_plan(recipe_path=recipe, output_dir=plan_dir).exit_code == 0
    rows = read_manifest_rows(root)
    row = rows[0]
    row["status"] = "completed"
    legacy_run_identity_fields = {"experiment_id", "step_id", "run_id", "run_name", "version"}
    for field in FROZEN_RUN_FIELDS - legacy_run_identity_fields:
        row[field] = ""
    for field in managed_run_parameters(row):
        row[field] = ""
    if modern_evidence == "managed_parameter":
        row["runtime.lr"] = "1e-6"
    else:
        row.update(
            {
                "selection_task": "hparam_tune",
                "selection_mode": "min",
                "selection_split": "val",
                "selection_report": str(root / "reports" / "hparam_selection.md"),
                "selection_report_sha256": "a" * 64,
            }
        )
    write_rows(root / "run_manifest.tsv", rows)
    step_path = root / "steps" / "unit-hparam-tune" / "step.yaml"
    step_path.unlink()
    (step_path.parent / ".step.yaml.lock").unlink()
    step_path.parent.rmdir()
    report = tmp_path / "final.md"
    report.write_text("# Final\n")
    before = workspace_files(root)

    with pytest.raises(ValueError, match="unregistered steps"):
        experiments.finalize_experiment(root, report)

    assert workspace_files(root) == before
    assert not (root / "reports" / "final.md").exists()


def test_experiment_status_terminal_and_completed_contract(tmp_path):
    root = tmp_path / "experiment"
    init_status_workspace(root)
    add_plan(root, step_id="train", status="failed")

    ready = experiments.experiment_status(root)

    assert ready["summary"]["state"] == "ready_to_finalize"
    assert [blocker["code"] for blocker in ready["blockers"]] == ["final_report_required"]

    manifest = yaml.safe_load((root / "experiment.yaml").read_text())
    manifest["experiment"].update({"status": "completed", "completed_at": "2026-08-25T02:00:00Z"})
    (root / "experiment.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))
    completed = experiments.experiment_status(root)
    assert completed["summary"]["state"] == "completed"
    assert completed["blockers"] == []

    rows = read_manifest_rows(root)
    rows[0]["status"] = "running"
    write_rows(root / "run_manifest.tsv", rows)
    with pytest.raises(ValueError, match="Completed experiment metadata conflicts"):
        experiments.experiment_status(root)


def test_experiment_status_advances_ordinary_hparam_selection_and_report(tmp_path):
    root = tmp_path / "experiment"
    init_status_workspace(root)
    add_plan(root, step_id="tune", task="hparam_tune", status="completed")
    before = workspace_files(root)

    pending = experiments.experiment_status(root)

    assert workspace_files(root) == before
    assert pending["summary"]["state"] == "ready_to_select"
    assert [(blocker["code"], blocker["step_id"]) for blocker in pending["blockers"]] == [
        ("hparam_selection_required", "tune")
    ]

    record_hparam_selection(root)
    selected_before = workspace_files(root)
    selected = experiments.experiment_status(root)
    assert workspace_files(root) == selected_before
    assert selected["summary"]["state"] == "ready_to_report"
    assert [blocker["code"] for blocker in selected["blockers"]] == ["hparam_selection_report_required"]

    record_hparam_selection(root, write_report=True)
    ready_before = workspace_files(root)
    ready = experiments.experiment_status(root)
    assert workspace_files(root) == ready_before
    assert ready["summary"]["state"] == "ready_to_finalize"
    assert ready["blockers"] == []


def test_experiment_status_blocks_each_pending_hparam_selection(tmp_path):
    root = tmp_path / "experiment"
    init_status_workspace(root)
    for step_id in ("tune-a", "tune-b"):
        add_plan(root, step_id=step_id, task="hparam_tune", status="completed")
    before = workspace_files(root)

    snapshot = experiments.experiment_status(root)

    assert workspace_files(root) == before
    assert snapshot["summary"]["state"] == "ready_to_select"
    assert [(blocker["code"], blocker["step_id"]) for blocker in snapshot["blockers"]] == [
        ("hparam_selection_required", "tune-a"),
        ("hparam_selection_required", "tune-b"),
    ]


def test_experiment_status_requires_combined_report_for_mixed_ordinary_steps(tmp_path):
    root = tmp_path / "experiment"
    init_status_workspace(root)
    add_plan(root, step_id="tune", task="hparam_tune", status="completed")
    add_plan(root, step_id="prepare", status="completed")
    record_hparam_selection(root, write_report=True)

    snapshot = experiments.experiment_status(root)

    assert snapshot["summary"]["state"] == "ready_to_report"
    assert "combined_report_required" in {blocker["code"] for blocker in snapshot["blockers"]}


def test_experiment_status_all_failed_hparam_requires_failure_report_not_selection(tmp_path):
    root = tmp_path / "experiment"
    init_status_workspace(root)
    add_plan(root, step_id="tune", task="hparam_tune", status="failed")

    snapshot = experiments.experiment_status(root)

    assert snapshot["summary"]["state"] == "ready_to_report"
    assert "failure_report_required" in {blocker["code"] for blocker in snapshot["blockers"]}


def test_experiment_status_rejects_selection_metadata_on_non_hparam_run(tmp_path):
    root = tmp_path / "experiment"
    init_status_workspace(root)
    add_plan(root, step_id="train", status="completed")
    rows = read_manifest_rows(root)
    rows[0].update(
        {
            "selection_task": "hparam_tune",
            "selection_mode": "max",
            "selection_split": "val",
            "selection_report": str(root / "reports" / "hparam_selection.md"),
            "selection_report_sha256": "0" * 64,
        }
    )
    write_rows(root / "run_manifest.tsv", rows)
    report = tmp_path / "final.md"
    report.write_text("# Final report\n")
    before = workspace_files(root)

    with pytest.raises(ValueError, match="not owned by a registered hparam plan"):
        experiments.experiment_status(root)
    with pytest.raises(ValueError, match="not owned by a registered hparam plan"):
        experiments.finalize_experiment(root, report)

    assert workspace_files(root) == before


def test_hparam_selection_lifecycle_rejects_misowned_metadata_in_mixed_step(tmp_path):
    registered_steps = [
        {
            "manifest": {
                "step": {"id": "mixed", "phase": "train", "purpose": "mixed plan types"},
            },
            "plans": [
                {
                    "task": "hparam_tune",
                    "run_keys": [("mixed", "run-000")],
                    "path": str(tmp_path / "hparam"),
                    "selection": {"metric": "val_loss", "mode": "min", "split": "val"},
                },
                {
                    "task": "finetune",
                    "run_keys": [("mixed", "run-001")],
                    "path": str(tmp_path / "finetune"),
                },
            ],
        }
    ]
    rows = [
        {"step_id": "mixed", "run_id": "run-000", "status": "planned"},
        {
            "step_id": "mixed",
            "run_id": "run-001",
            "status": "completed",
            "selection_task": "hparam_tune",
        },
    ]

    with pytest.raises(ValueError, match="mixed / run-001"):
        experiment_tracking.hparam_selection_lifecycle(registered_steps, rows, root=tmp_path)


def test_experiment_status_keeps_completed_legacy_hparam_selection_readable(tmp_path):
    root = tmp_path / "experiment"
    init_status_workspace(root)
    add_plan(root, step_id="tune", task="hparam_tune", status="completed")
    rows = read_manifest_rows(root)
    rows[0].update({"metric": "val_loss", "score": "0.25", "rank": "1", "checkpoint_path": "/legacy.ckpt"})
    write_rows(root / "run_manifest.tsv", rows)

    active = experiments.experiment_status(root)
    assert active["summary"]["state"] == "ready_to_select"

    manifest = yaml.safe_load((root / "experiment.yaml").read_text())
    manifest["experiment"].update({"status": "completed", "completed_at": "2026-08-26T00:00:00Z"})
    (root / "experiment.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))
    completed = experiments.experiment_status(root)
    assert completed["summary"]["state"] == "completed"


@pytest.mark.parametrize("field", ["selection_task", "selection_report_sha256"])
def test_experiment_status_rejects_partially_materialized_selection_metadata(tmp_path, field):
    root = tmp_path / "experiment"
    init_status_workspace(root)
    add_plan(root, step_id="tune", task="hparam_tune", status="completed")
    rows = read_manifest_rows(root)
    rows[0][field] = "hparam_tune" if field == "selection_task" else "0" * 64
    write_rows(root / "run_manifest.tsv", rows)

    with pytest.raises(ValueError, match="partially materialized"):
        experiments.experiment_status(root)


def test_experiment_finalize_requires_selection_and_uses_verified_selection_report(tmp_path):
    root = tmp_path / "experiment"
    init_status_workspace(root)
    add_plan(root, step_id="tune", task="hparam_tune", status="completed")
    arbitrary = tmp_path / "arbitrary.md"
    arbitrary.write_text("# Arbitrary\n")

    with pytest.raises(ValueError, match="must be selected"):
        experiments.finalize_experiment(root, arbitrary)

    selection_report = record_hparam_selection(root, write_report=True)
    copied_report = root / "selection-copy.md"
    copied_report.write_bytes(selection_report.read_bytes())
    before = workspace_files(root)
    with pytest.raises(ValueError, match="must finalize from"):
        experiments.finalize_experiment(root, copied_report)
    assert workspace_files(root) == before

    target = experiments.finalize_experiment(root, selection_report)
    assert target.read_text() == selection_report.read_text()
    completed = yaml.safe_load((root / "experiment.yaml").read_text())["experiment"]
    assert completed["final_report"] == str(target)
    assert completed["final_report_sha256"] == sha256_of(target)
    assert completed["selection_report_sha256"] == sha256_of(selection_report)


def test_experiment_finalize_rehashes_selected_checkpoints_before_writing(tmp_path):
    root = tmp_path / "experiment"
    init_status_workspace(root)
    add_plan(root, step_id="tune", task="hparam_tune", status="completed")
    selection_report = record_hparam_selection(root, write_report=True)
    checkpoint = Path(read_manifest_rows(root)[0]["checkpoint_path"])
    checkpoint.write_bytes(b"tampered checkpoint\n")
    before = workspace_files(root)

    assert experiments.experiment_status(root)["summary"]["state"] == "ready_to_finalize"
    with pytest.raises(ValueError, match="Frozen checkpoint SHA-256 differs"):
        experiments.finalize_experiment(root, selection_report)

    assert workspace_files(root) == before
    assert not (root / "reports" / "final.md").exists()
    # The selected checkpoint evidence is revalidated, not only rehashed.
    checkpoint.unlink()
    before = workspace_files(root)
    with pytest.raises(ValueError, match="Checkpoint evidence is missing"):
        experiments.finalize_experiment(root, selection_report)
    assert workspace_files(root) == before


@pytest.mark.parametrize("mutation", ["delete", "tamper"])
def test_experiment_status_validates_bound_final_report(tmp_path, mutation):
    root = tmp_path / "experiment"
    init_status_workspace(root)
    add_plan(root, step_id="train", status="completed")
    report = tmp_path / "report.md"
    report.write_text("# Final report\n")
    target = experiments.finalize_experiment(root, report)
    if mutation == "delete":
        target.unlink()
    else:
        target.write_text("# Tampered final report\n")

    with pytest.raises(ValueError, match="final report|Managed file is missing"):
        experiments.experiment_status(root)


def test_experiment_status_rejects_incomplete_terminal_report_binding(tmp_path):
    root = tmp_path / "experiment"
    init_status_workspace(root)
    add_plan(root, step_id="train", status="completed")
    report = tmp_path / "report.md"
    report.write_text("# Final report\n")
    experiments.finalize_experiment(root, report)
    manifest_path = root / "experiment.yaml"
    manifest = yaml.safe_load(manifest_path.read_text())
    manifest["experiment"].pop("final_report_sha256")
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False))

    with pytest.raises(ValueError, match="incomplete or unexpected terminal fields"):
        experiments.experiment_status(root)


def test_experiment_status_requires_terminal_bindings_for_modern_hparam_selection(tmp_path):
    root = tmp_path / "experiment"
    init_status_workspace(root)
    add_plan(root, step_id="tune", task="hparam_tune", status="completed")
    selection_report = record_hparam_selection(root, write_report=True)
    final = experiments.finalize_experiment(root, selection_report)
    manifest_path = root / "experiment.yaml"
    manifest = yaml.safe_load(manifest_path.read_text())
    for field in ("final_report", "final_report_sha256", "selection_report_sha256"):
        manifest["experiment"].pop(field)
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False))
    final.write_text("# Tampered final report\n")

    with pytest.raises(ValueError, match="missing terminal report bindings"):
        experiments.experiment_status(root)


def test_experiment_status_rejects_modern_completion_downgraded_to_legacy_selection(tmp_path):
    root = tmp_path / "experiment"
    init_status_workspace(root)
    add_plan(root, step_id="tune", task="hparam_tune", status="completed")
    selection_report = record_hparam_selection(root, write_report=True)
    experiments.finalize_experiment(root, selection_report)

    rows = read_manifest_rows(root)
    for row in rows:
        for field in (
            "selection_task",
            "metric",
            "selection_mode",
            "selection_split",
            "score",
            "rank",
            "checkpoint_path",
            "checkpoint_sha256",
            "selection_report",
            "selection_report_sha256",
        ):
            row[field] = ""
    write_rows(root / "run_manifest.tsv", rows)
    manifest_path = root / "experiment.yaml"
    manifest = yaml.safe_load(manifest_path.read_text())
    manifest["experiment"].pop("selection_report_sha256")
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False))

    with pytest.raises(ValueError, match="incomplete hparam selection evidence"):
        experiments.experiment_status(root)


def test_experiment_status_detects_selection_commit_after_terminal_binding(tmp_path):
    root = tmp_path / "experiment"
    init_status_workspace(root)
    add_plan(root, step_id="tune", task="hparam_tune", status="completed")
    selection_report = record_hparam_selection(root, write_report=True)
    final = experiments.finalize_experiment(root, selection_report)
    final_bytes = final.read_bytes()

    record_hparam_selection(root, write_report=True, score="0.5")

    with pytest.raises(ValueError, match="selection report differs from its terminal binding"):
        experiments.experiment_status(root)
    assert final.read_bytes() == final_bytes


def test_completed_experiment_rejects_late_run_manifest_merge(tmp_path):
    root = tmp_path / "experiment"
    init_status_workspace(root)
    add_plan(root, step_id="train", status="completed")
    report = tmp_path / "report.md"
    report.write_text("# Final\n")
    experiments.finalize_experiment(root, report)
    before = workspace_files(root)
    row = read_manifest_rows(root)[0]

    with pytest.raises(ValueError, match="completed and cannot update canonical runs"):
        merge_run_manifest(root, [{**row, "health_status": "late observation"}])

    assert workspace_files(root) == before
    assert experiments.experiment_status(root)["summary"]["state"] == "completed"


def test_inflight_run_manifest_merge_rechecks_owner_after_waiting_for_lock(tmp_path, monkeypatch):
    root = tmp_path / "experiment"
    init_status_workspace(root)
    add_plan(root, step_id="train", status="completed")
    row = read_manifest_rows(root)[0]
    merge_waiting = threading.Event()
    allow_merge = threading.Event()
    errors = []
    real_lock = experiment_io.blocking_file_lock

    @contextmanager
    def delayed_lock(path):
        if threading.current_thread().name == "late-merge" and Path(path) == root / "run_manifest.tsv.lock":
            merge_waiting.set()
            assert allow_merge.wait(timeout=5)
        with real_lock(path):
            yield

    def late_merge():
        try:
            merge_run_manifest(root, [{**row, "health_status": "late observation"}])
        except Exception as exc:
            errors.append(exc)

    monkeypatch.setattr(experiment_io, "blocking_file_lock", delayed_lock)
    worker = threading.Thread(target=late_merge, name="late-merge")
    worker.start()
    assert merge_waiting.wait(timeout=5)

    report = tmp_path / "report.md"
    report.write_text("# Final\n")
    experiments.finalize_experiment(root, report)
    after_finalize = workspace_files(root)
    allow_merge.set()
    worker.join(timeout=5)

    assert not worker.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], ValueError)
    assert "completed and cannot update canonical runs" in str(errors[0])
    assert workspace_files(root) == after_finalize
    assert experiments.experiment_status(root)["summary"]["state"] == "completed"


def test_experiment_finalize_allows_combined_or_failure_reports(tmp_path):
    mixed = tmp_path / "mixed"
    init_status_workspace(mixed)
    add_plan(mixed, step_id="tune", task="hparam_tune", status="completed")
    add_plan(mixed, step_id="prepare", status="completed")
    selection_report = record_hparam_selection(mixed, write_report=True)
    with pytest.raises(ValueError, match="cannot replace"):
        experiments.finalize_experiment(mixed, selection_report)
    combined = tmp_path / "combined.md"
    combined.write_text("# Combined report\n\nSelection and preparation summary.\n")
    assert experiments.finalize_experiment(mixed, combined).read_text() == combined.read_text()

    failed = tmp_path / "failed"
    init_status_workspace(failed)
    add_plan(failed, step_id="tune", task="hparam_tune", status="failed")
    stale_selection = failed / "reports" / "hparam_selection.md"
    stale_selection.parent.mkdir()
    stale_selection.write_text("# Stale selection\n")
    with pytest.raises(ValueError, match="cannot replace the required hparam failure report"):
        experiments.finalize_experiment(failed, stale_selection)
    failure_report = tmp_path / "failure.md"
    failure_report.write_text("# Failure report\n\nNo candidate completed successfully.\n")
    assert experiments.finalize_experiment(failed, failure_report).read_text() == failure_report.read_text()


def test_experiment_finalize_rejects_selection_report_dotdot_alias_for_mixed_experiment(tmp_path):
    root = tmp_path / "mixed"
    init_status_workspace(root)
    add_plan(root, step_id="tune", task="hparam_tune", status="completed")
    add_plan(root, step_id="prepare", status="completed")
    selection_report = record_hparam_selection(root, write_report=True)
    aliased_report = selection_report.parent / ".." / "reports" / selection_report.name
    before = workspace_files(root)

    with pytest.raises(ValueError, match="cannot replace the required combined experiment report"):
        experiments.finalize_experiment(root, aliased_report)

    assert workspace_files(root) == before
    assert not (root / "reports" / "final.md").exists()


def test_experiment_finalize_rejects_selection_report_copy_as_failure_report(tmp_path):
    root = tmp_path / "failed"
    init_status_workspace(root)
    add_plan(root, step_id="tune", task="hparam_tune", status="failed")
    selection_report = root / "reports" / "hparam_selection.md"
    selection_report.parent.mkdir()
    selection_report.write_text("# Stale selection report\n")
    same_content = root / "failure.md"
    same_content.write_bytes(selection_report.read_bytes())
    before = workspace_files(root)

    with pytest.raises(ValueError, match="cannot replace the required hparam failure report"):
        experiments.finalize_experiment(root, same_content)

    assert workspace_files(root) == before
    assert not (root / "reports" / "final.md").exists()


def test_experiment_status_requires_combined_report_when_one_hparam_step_failed(tmp_path):
    root = tmp_path / "experiment"
    init_status_workspace(root)
    add_plan(root, step_id="selected", task="hparam_tune", status="completed")
    add_plan(root, step_id="failed", task="hparam_tune", status="failed")
    selection_report = record_hparam_selection(root, step_id="selected", write_report=True)

    snapshot = experiments.experiment_status(root)

    assert snapshot["summary"]["state"] == "ready_to_report"
    assert "combined_report_required" in {blocker["code"] for blocker in snapshot["blockers"]}
    with pytest.raises(ValueError, match="cannot replace"):
        experiments.finalize_experiment(root, selection_report)
    combined = tmp_path / "combined.md"
    combined.write_text("# Combined report\n\nOne hparam step failed.\n")
    assert experiments.finalize_experiment(root, combined).read_text() == combined.read_text()


def test_experiment_finalize_rejects_stale_selection_after_success_becomes_failed(tmp_path):
    root = tmp_path / "experiment"
    init_status_workspace(root)
    add_plan(root, step_id="tune", task="hparam_tune", status="completed")
    selection_report = record_hparam_selection(root, write_report=True)
    rows = read_manifest_rows(root)
    rows[0]["status"] = "failed"
    write_rows(root / "run_manifest.tsv", rows)
    before = workspace_files(root)

    with pytest.raises(ValueError, match="stale for all-failed step"):
        experiments.experiment_status(root)
    with pytest.raises(ValueError, match="stale for all-failed step"):
        experiments.finalize_experiment(root, selection_report)

    assert workspace_files(root) == before


@pytest.mark.parametrize("mutation", ["missing", "tampered", "duplicate"])
def test_experiment_status_rejects_missing_or_drifted_hparam_ranking(tmp_path, mutation):
    root = tmp_path / "experiment"
    init_status_workspace(root)
    add_plan(root, step_id="tune", task="hparam_tune", status="completed")
    selection_report = record_hparam_selection(root, write_report=True)
    ranking = root / "reports" / "ranking.csv"
    if mutation == "missing":
        ranking.unlink()
    elif mutation == "tampered":
        ranking.write_text(ranking.read_text().replace("0.25", "999", 1))
    else:
        lines = ranking.read_text().splitlines()
        ranking.write_text("\n".join([*lines, lines[1]]) + "\n")
    before = workspace_files(root)

    snapshot = experiments.experiment_status(root)

    assert snapshot["summary"]["state"] == "ready_to_report"
    assert [blocker["code"] for blocker in snapshot["blockers"]] == ["hparam_selection_report_required"]
    with pytest.raises(ValueError, match="selection report is missing or differs"):
        experiments.finalize_experiment(root, selection_report)
    assert workspace_files(root) == before


@pytest.mark.parametrize(
    ("field", "mutation"),
    [
        ("config", "tamper"),
        ("config", "remove"),
        ("unexpected", "add"),
    ],
)
def test_experiment_status_rejects_hparam_ranking_candidate_provenance_drift(tmp_path, field, mutation):
    root = tmp_path / "experiment"
    recipe = write_public_hparam_recipe(
        root,
        {"runtime.lr": [1e-6]},
        selection_metric="val_loss",
        selection_mode="min",
    )
    plan_dir = root / "plans" / "tune"
    assert plans.build_plan(recipe_path=recipe, output_dir=plan_dir).exit_code == 0
    rows = read_manifest_rows(root)
    for row in rows:
        row["status"] = "completed"
    write_rows(root / "run_manifest.tsv", rows)
    selection_report = record_hparam_selection(root, step_id="unit-hparam-tune", write_report=True)
    ranking = root / "reports" / "ranking.csv"
    ranking_rows = read_rows(ranking, require_managed_identity=True)
    if mutation == "remove":
        for row in ranking_rows:
            row.pop(field)
    else:
        ranking_rows[0][field] = "tampered"
    write_rows(ranking, ranking_rows)

    snapshot = experiments.experiment_status(root)

    assert snapshot["summary"]["state"] == "ready_to_report"
    assert [blocker["code"] for blocker in snapshot["blockers"]] == ["hparam_selection_report_required"]
    with pytest.raises(ValueError, match="selection report is missing or differs"):
        experiments.finalize_experiment(root, selection_report)


@pytest.mark.parametrize(
    ("selection_metric", "selection_mode", "scores"),
    [
        ("val_ahi_pearson", "max", ("0.1", "0.9")),
        ("val_loss", "min", ("0.9", "0.1")),
    ],
)
def test_experiment_status_rejects_hparam_ranks_opposed_to_selection_mode(
    tmp_path, selection_metric, selection_mode, scores
):
    root = tmp_path / "experiment"
    recipe = write_public_hparam_recipe(
        root,
        {"runtime.lr": [1e-6, 2e-6]},
        selection_metric=selection_metric,
        selection_mode=selection_mode,
        max_runs=2,
    )
    plan_dir = root / "plans" / "tune"
    assert plans.build_plan(recipe_path=recipe, output_dir=plan_dir).exit_code == 0
    rows = read_manifest_rows(root)
    for rank, (row, score) in enumerate(zip(rows, scores), start=1):
        row.update(
            {
                "status": "completed",
                "selection_task": "hparam_tune",
                "metric": selection_metric,
                "selection_mode": selection_mode,
                "selection_split": "val",
                "score": score,
                "rank": str(rank),
                "checkpoint_path": str(Path(row["checkpoint_dir"]) / f"epoch={rank}.ckpt"),
                "checkpoint_sha256": "a" * 64,
                "run_manifest": str(Path(row["runtime_dir"]) / "run_manifest.json"),
            }
        )
    write_rows(root / "run_manifest.tsv", rows)
    report = tmp_path / "report.md"
    report.write_text("# Report\n")
    before = workspace_files(root)

    with pytest.raises(ValueError, match="ranks disagree with selection mode"):
        experiments.experiment_status(root)
    with pytest.raises(ValueError, match="ranks disagree with selection mode"):
        experiments.finalize_experiment(root, report)
    assert workspace_files(root) == before


@pytest.mark.parametrize(("field", "mutation"), [("run_manifest", "remove"), ("source", "add")])
def test_experiment_status_rejects_coherent_val_selection_provenance_drift(tmp_path, field, mutation):
    root = tmp_path / "experiment"
    init_status_workspace(root)
    add_plan(root, step_id="tune", task="hparam_tune", status="completed")
    selection_report = record_hparam_selection(root, write_report=True)
    canonical = read_manifest_rows(root)
    ranking_path = root / "reports" / "ranking.csv"
    ranking = read_rows(ranking_path, require_managed_identity=True)
    if mutation == "remove":
        canonical[0].pop(field)
        ranking[0].pop(field)
    else:
        canonical[0][field] = "forged_source"
        ranking[0][field] = "forged_source"
    write_rows(root / "run_manifest.tsv", canonical)
    write_rows(ranking_path, ranking)
    before = workspace_files(root)

    with pytest.raises(ValueError, match="Canonical hparam selection evidence is invalid"):
        experiments.experiment_status(root)
    with pytest.raises(ValueError, match="Canonical hparam selection evidence is invalid"):
        experiments.finalize_experiment(root, selection_report)

    assert workspace_files(root) == before


def test_experiment_status_keeps_final_report_blocker_experiment_wide(tmp_path):
    root = tmp_path / "experiment"
    init_status_workspace(root)
    add_plan(root, step_id="prepare", status="completed")
    add_plan(root, step_id="evaluate", status="failed")

    snapshot = experiments.experiment_status(root)

    blocker = next(item for item in snapshot["blockers"] if item["code"] == "final_report_required")
    assert blocker["step_id"] is None
    assert blocker["run_ids"] == []
    assert all("final_report_required" not in run["blockers"] for run in snapshot["runs"])


def test_experiment_status_blocks_finalize_for_stopped_run_without_reason(tmp_path):
    root = tmp_path / "experiment"
    init_status_workspace(root)
    add_plan(root, step_id="train", status="stopped")
    before = workspace_files(root)

    blocked = experiments.experiment_status(root)

    assert workspace_files(root) == before
    assert blocked["summary"]["state"] == "blocked"
    assert blocked["blockers"][0]["code"] == "missing_stop_reason"
    assert blocked["blockers"][0]["run_ids"] == ["run-000"]
    assert blocked["runs"][0]["blockers"] == ["missing_stop_reason"]

    rows = read_manifest_rows(root)
    rows[0]["stop_reason"] = "manual stop after invalid labels"
    write_rows(root / "run_manifest.tsv", rows)
    ready = experiments.experiment_status(root)
    assert ready["summary"]["state"] == "ready_to_finalize"
    assert [blocker["code"] for blocker in ready["blockers"]] == ["final_report_required"]


def test_experiment_status_rejects_completed_stopped_run_without_reason(tmp_path):
    root = tmp_path / "experiment"
    init_status_workspace(root)
    add_plan(root, step_id="train", status="stopped")
    manifest = yaml.safe_load((root / "experiment.yaml").read_text())
    manifest["experiment"].update({"status": "completed", "completed_at": "2026-08-25T02:00:00Z"})
    (root / "experiment.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))

    with pytest.raises(ValueError, match="stopped runs missing stop_reason"):
        experiments.experiment_status(root)

    rows = read_manifest_rows(root)
    rows[0]["stop_reason"] = "manual stop after invalid labels"
    write_rows(root / "run_manifest.tsv", rows)
    assert experiments.experiment_status(root)["summary"]["state"] == "completed"
