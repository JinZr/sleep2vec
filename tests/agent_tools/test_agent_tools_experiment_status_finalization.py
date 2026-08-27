from __future__ import annotations

from contextlib import contextmanager
import hashlib
from pathlib import Path
import threading

import pytest
from test_agent_tools_experiment_status import (
    _add_plan,
    _init_workspace,
    _read_manifest_rows,
    _record_hparam_selection,
    _sha256,
    _workspace_files,
    _write_public_hparam_recipe,
)
from test_agent_tools_experiment_status import _stub_execution_target  # noqa: F401
import yaml

from agent_tools import experiment_io, experiment_pipeline, experiment_tracking, experiments, plans
from agent_tools.experiment_workspace import FROZEN_RUN_FIELDS, managed_run_parameters, merge_run_manifest
from agent_tools.manifests import read_rows, write_rows


def test_experiment_finalize_rejects_unmaterialized_step(tmp_path):
    root = tmp_path / "experiment"
    _init_workspace(root)
    _add_plan(root, step_id="ordinary", status="completed")
    step_dir = root / "steps" / "evaluate"
    step_dir.mkdir(parents=True)
    (step_dir / "step.yaml").write_text(
        yaml.safe_dump(
            {
                "step": {"id": "evaluate", "phase": "evaluate", "purpose": "Run external evaluation."},
                "experiment_id": "status-unit",
                "plan_controller": "unassigned",
                "recipe_path": "",
                "plans": [],
            },
            sort_keys=False,
        )
    )
    report = tmp_path / "final.md"
    report.write_text("# Final\n")
    before = _workspace_files(root)

    with pytest.raises(ValueError, match="incomplete canonical steps: unmaterialized_step"):
        experiments.finalize_experiment(root, report)

    assert _workspace_files(root) == before


@pytest.mark.parametrize("registration_mutation", ["empty", "missing"])
def test_experiment_finalize_rejects_canonical_runs_without_registered_plan(tmp_path, registration_mutation):
    root = tmp_path / "experiment"
    _init_workspace(root)
    _add_plan(root, step_id="tune", task="hparam_tune", status="completed")
    step_path = root / "steps" / "tune" / "step.yaml"
    if registration_mutation == "empty":
        step_manifest = yaml.safe_load(step_path.read_text())
        step_manifest["plans"] = []
        step_path.write_text(yaml.safe_dump(step_manifest, sort_keys=False))
    else:
        step_path.unlink()
    report = tmp_path / "final.md"
    report.write_text("# Final\n")
    before = _workspace_files(root)

    with pytest.raises(
        ValueError, match="plans differ from canonical run keys|unregistered steps|Managed file is missing"
    ):
        experiments.finalize_experiment(root, report)

    assert _workspace_files(root) == before
    assert not (root / "reports" / "final.md").exists()


@pytest.mark.parametrize("modern_evidence", ["managed_parameter", "selection_metadata"])
def test_experiment_finalize_does_not_downgrade_managed_hparam_evidence_to_legacy(tmp_path, modern_evidence):
    root = tmp_path / "experiment"
    recipe = _write_public_hparam_recipe(root, {"runtime.lr": [1e-6]})
    plan_dir = root / "plans" / "tune"
    assert plans.build_plan(recipe_path=recipe, output_dir=plan_dir).exit_code == 0
    rows = _read_manifest_rows(root)
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
    (step_path.parent / ".step.yaml.cas.lock").unlink()
    step_path.parent.rmdir()
    report = tmp_path / "final.md"
    report.write_text("# Final\n")
    before = _workspace_files(root)

    with pytest.raises(ValueError, match="unregistered steps"):
        experiments.finalize_experiment(root, report)

    assert _workspace_files(root) == before
    assert not (root / "reports" / "final.md").exists()


@pytest.mark.parametrize("controller", ["adaptive", "pipeline"])
def test_experiment_finalize_preserves_controller_verified_finalization(tmp_path, controller):
    root = tmp_path / "experiment"
    _init_workspace(root)
    _add_plan(
        root,
        step_id="tune" if controller == "adaptive" else "evaluate",
        task="hparam_tune" if controller == "adaptive" else "finetune",
        status="completed",
        adaptive=controller == "adaptive",
        pipeline=controller == "pipeline",
    )
    report = tmp_path / "controller-report.md"
    report.write_text("# Controller-verified report\n")

    target = experiments.finalize_experiment(root, report)

    assert target.read_text() == report.read_text()


def test_pipeline_facade_callback_can_finalize_controller_verified_report(tmp_path, monkeypatch):
    root = tmp_path / "experiment"
    _init_workspace(root)
    _add_plan(root, step_id="evaluate", status="completed", pipeline=True)
    report = tmp_path / "pipeline-report.md"
    report.write_text("# Pipeline report\n")
    spec = tmp_path / "pipeline.yaml"
    spec.write_text("pipeline: unit\n")

    def controller(_root, _spec, **kwargs):
        target = kwargs["finalize_callback"](root, report)
        return {"status": "completed", "final_report": str(target)}

    monkeypatch.setattr(experiment_pipeline, "run_experiment_pipeline", controller)

    result = experiments.run_experiment_pipeline(root, spec, execute=True)

    assert result == {"status": "completed", "final_report": str(root / "reports" / "final.md")}
    assert (root / "reports" / "final.md").read_text() == report.read_text()


def test_experiment_status_terminal_and_completed_contract(tmp_path):
    root = tmp_path / "experiment"
    _init_workspace(root)
    _add_plan(root, step_id="train", status="failed")

    ready = experiments.experiment_status(root)

    assert ready["summary"]["state"] == "ready_to_finalize"
    finalize = ready["decision"]["other_legal_actions"][0]
    assert finalize["id"] == "experiment-finalize"
    assert finalize["required_inputs"] == ["report_path"]

    manifest = yaml.safe_load((root / "experiment.yaml").read_text())
    manifest["experiment"].update({"status": "completed", "completed_at": "2026-08-25T02:00:00Z"})
    (root / "experiment.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))
    completed = experiments.experiment_status(root)
    assert completed["summary"]["state"] == "completed"
    assert completed["decision"]["recommended_next"] is None

    rows = _read_manifest_rows(root)
    rows[0]["status"] = "running"
    write_rows(root / "run_manifest.tsv", rows)
    with pytest.raises(ValueError, match="Completed experiment metadata conflicts"):
        experiments.experiment_status(root)


def test_experiment_status_advances_ordinary_hparam_selection_and_report(tmp_path):
    root = tmp_path / "experiment"
    _init_workspace(root)
    plan_dir, _row = _add_plan(root, step_id="tune", task="hparam_tune", status="completed")
    before = _workspace_files(root)

    pending = experiments.experiment_status(root)

    assert _workspace_files(root) == before
    assert pending["summary"]["state"] == "ready_to_select"
    assert pending["decision"]["recommended_next"]["argv"] == [
        "python",
        "-m",
        "agent_tools",
        "hparam-select",
        "--run-dir",
        str(plan_dir),
    ]
    assert pending["decision"]["blocked_actions"] == ["finalize"]

    report_path = _record_hparam_selection(root)
    selected_before = _workspace_files(root)
    selected = experiments.experiment_status(root)
    assert _workspace_files(root) == selected_before
    assert selected["summary"]["state"] == "ready_to_report"
    assert selected["decision"]["recommended_next"]["id"] == "hparam-select"

    _record_hparam_selection(root, write_report=True)
    ready_before = _workspace_files(root)
    ready = experiments.experiment_status(root)
    assert _workspace_files(root) == ready_before
    assert ready["summary"]["state"] == "ready_to_finalize"
    assert ready["decision"]["recommended_next"]["argv"][-2:] == ["--report", str(report_path)]


def test_experiment_status_requires_combined_report_for_mixed_ordinary_steps(tmp_path):
    root = tmp_path / "experiment"
    _init_workspace(root)
    _add_plan(root, step_id="tune", task="hparam_tune", status="completed")
    _add_plan(root, step_id="prepare", status="completed")
    _record_hparam_selection(root, write_report=True)

    snapshot = experiments.experiment_status(root)

    assert snapshot["summary"]["state"] == "ready_to_report"
    assert snapshot["decision"]["manual_choice_required"] is True
    assert snapshot["decision"]["other_legal_actions"][0]["required_inputs"] == ["report_path"]
    assert "combined_report_required" in {blocker["code"] for blocker in snapshot["blockers"]}


def test_pipeline_step_prevents_hparam_selection_report_from_becoming_experiment_final(tmp_path):
    root = tmp_path / "experiment"
    _init_workspace(root)
    _add_plan(root, step_id="tune", task="hparam_tune", status="completed")
    _add_plan(root, step_id="evaluate", status="completed", pipeline=True)
    selection_report = _record_hparam_selection(root, write_report=True)
    combined = tmp_path / "pipeline-combined.md"
    combined.write_text("# Pipeline combined report\n")

    with pytest.raises(ValueError, match="cannot replace the required combined experiment report"):
        experiments.finalize_experiment(root, selection_report)

    target = experiments.finalize_experiment(root, combined)

    assert target.read_text() == combined.read_text()


def test_experiment_status_all_failed_hparam_requires_failure_report_not_selection(tmp_path):
    root = tmp_path / "experiment"
    _init_workspace(root)
    _add_plan(root, step_id="tune", task="hparam_tune", status="failed")

    snapshot = experiments.experiment_status(root)

    assert snapshot["summary"]["state"] == "ready_to_report"
    assert snapshot["decision"]["recommended_next"] is None
    assert snapshot["decision"]["other_legal_actions"][0]["required_inputs"] == ["report_path"]
    assert "failure_report_required" in {blocker["code"] for blocker in snapshot["blockers"]}


def test_experiment_status_rejects_selection_metadata_on_non_hparam_run(tmp_path):
    root = tmp_path / "experiment"
    _init_workspace(root)
    _add_plan(root, step_id="train", status="completed")
    rows = _read_manifest_rows(root)
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
    before = _workspace_files(root)

    with pytest.raises(ValueError, match="not owned by a registered hparam plan"):
        experiments.experiment_status(root)
    with pytest.raises(ValueError, match="not owned by a registered hparam plan"):
        experiments.finalize_experiment(root, report)

    assert _workspace_files(root) == before


def test_hparam_selection_lifecycle_rejects_misowned_metadata_in_mixed_step(tmp_path):
    registered_steps = [
        {
            "manifest": {
                "step": {"id": "mixed", "phase": "train", "purpose": "mixed plan types"},
                "plan_controller": "ordinary",
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


def test_experiment_status_rejects_all_failed_hparam_with_stale_checkpoint_rank(tmp_path):
    root = tmp_path / "experiment"
    _init_workspace(root)
    _add_plan(root, step_id="tune", task="hparam_tune", status="failed")
    rows = _read_manifest_rows(root)
    rows[0]["checkpoint_rank"] = "1"
    write_rows(root / "run_manifest.tsv", rows)
    report = tmp_path / "failure.md"
    report.write_text("# Failure report\n")
    before = _workspace_files(root)

    with pytest.raises(ValueError, match="stale for all-failed step"):
        experiments.experiment_status(root)
    with pytest.raises(ValueError, match="stale for all-failed step"):
        experiments.finalize_experiment(root, report)

    assert _workspace_files(root) == before


def test_experiment_status_keeps_completed_legacy_hparam_selection_readable(tmp_path):
    root = tmp_path / "experiment"
    _init_workspace(root)
    _add_plan(root, step_id="tune", task="hparam_tune", status="completed")
    rows = _read_manifest_rows(root)
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
    _init_workspace(root)
    _add_plan(root, step_id="tune", task="hparam_tune", status="completed")
    rows = _read_manifest_rows(root)
    rows[0][field] = "hparam_tune" if field == "selection_task" else "0" * 64
    write_rows(root / "run_manifest.tsv", rows)

    with pytest.raises(ValueError, match="partially materialized"):
        experiments.experiment_status(root)


def test_experiment_finalize_requires_selection_and_uses_verified_selection_report(tmp_path):
    root = tmp_path / "experiment"
    _init_workspace(root)
    _add_plan(root, step_id="tune", task="hparam_tune", status="completed")
    arbitrary = tmp_path / "arbitrary.md"
    arbitrary.write_text("# Arbitrary\n")

    with pytest.raises(ValueError, match="must be selected"):
        experiments.finalize_experiment(root, arbitrary)

    selection_report = _record_hparam_selection(root, write_report=True)
    with pytest.raises(ValueError, match="must finalize from"):
        experiments.finalize_experiment(root, arbitrary)

    target = experiments.finalize_experiment(root, selection_report)
    assert target.read_text() == selection_report.read_text()
    completed = yaml.safe_load((root / "experiment.yaml").read_text())["experiment"]
    assert completed["final_report"] == str(target)
    assert completed["final_report_sha256"] == _sha256(target)
    assert completed["selection_report_sha256"] == _sha256(selection_report)


def test_experiment_finalize_rehashes_selected_checkpoints_before_writing(tmp_path):
    root = tmp_path / "experiment"
    _init_workspace(root)
    _add_plan(root, step_id="tune", task="hparam_tune", status="completed")
    selection_report = _record_hparam_selection(root, write_report=True)
    checkpoint = Path(_read_manifest_rows(root)[0]["checkpoint_path"])
    checkpoint.write_bytes(b"tampered checkpoint\n")
    before = _workspace_files(root)

    assert experiments.experiment_status(root)["summary"]["state"] == "ready_to_finalize"
    with pytest.raises(ValueError, match="Frozen checkpoint SHA-256 differs"):
        experiments.finalize_experiment(root, selection_report)

    assert _workspace_files(root) == before
    assert not (root / "reports" / "final.md").exists()


@pytest.mark.parametrize(
    ("target", "host", "remote", "expected_host"),
    [("local", "", "controller", "controller"), ("ssh", "worker", "controller", "worker")],
)
def test_hparam_checkpoint_rehash_uses_execution_evidence_host(
    monkeypatch, target: str, host: str, remote: str, expected_host: str
):
    row = {
        "step_id": "tune",
        "run_id": "run-000",
        "target": target,
        "host": host,
        "checkpoint_path": "/data/epoch=1.ckpt",
        "checkpoint_sha256": "a" * 64,
    }
    audit_row = {
        **row,
        "checkpoint_path": "/data/epoch=2.ckpt",
        "checkpoint_sha256": "b" * 64,
    }
    validated = []
    hashed = []
    monkeypatch.setattr(
        experiments.tracking,
        "validate_checkpoint_evidence_rows",
        lambda rows, ranked, *, remote: validated.append((rows, ranked, remote)),
    )

    def checkpoint_sha256(evidence_row, checkpoint_path):
        hashed.append((evidence_row, checkpoint_path))
        return "a" * 64 if checkpoint_path == row["checkpoint_path"] else "b" * 64

    monkeypatch.setattr(experiments.evidence, "checkpoint_file_sha256", checkpoint_sha256)

    experiments._validate_hparam_checkpoints(
        [row],
        [{"ranked": [row], "checkpoint_audit_rows": [row, audit_row]}],
        remote=remote,
    )

    assert validated == [([row], [row, audit_row], remote)]
    assert [(evidence_row["target"], evidence_row["host"], path) for evidence_row, path in hashed] == [
        ("ssh", expected_host, row["checkpoint_path"]),
        ("ssh", expected_host, audit_row["checkpoint_path"]),
    ]


def test_experiment_finalize_rejects_selection_report_copy_for_pure_hparam_experiment(tmp_path):
    root = tmp_path / "experiment"
    _init_workspace(root)
    _add_plan(root, step_id="tune", task="hparam_tune", status="completed")
    selection_report = _record_hparam_selection(root, write_report=True)
    copied_report = root / "selection-copy.md"
    copied_report.write_bytes(selection_report.read_bytes())
    before = _workspace_files(root)

    with pytest.raises(ValueError, match="must finalize from"):
        experiments.finalize_experiment(root, copied_report)

    assert _workspace_files(root) == before
    assert not (root / "reports" / "final.md").exists()


@pytest.mark.parametrize("mutation", ["delete", "tamper"])
def test_experiment_status_validates_bound_final_report(tmp_path, mutation):
    root = tmp_path / "experiment"
    _init_workspace(root)
    _add_plan(root, step_id="train", status="completed")
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
    _init_workspace(root)
    _add_plan(root, step_id="train", status="completed")
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
    _init_workspace(root)
    _add_plan(root, step_id="tune", task="hparam_tune", status="completed")
    selection_report = _record_hparam_selection(root, write_report=True)
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
    _init_workspace(root)
    _add_plan(root, step_id="tune", task="hparam_tune", status="completed")
    selection_report = _record_hparam_selection(root, write_report=True)
    experiments.finalize_experiment(root, selection_report)

    rows = _read_manifest_rows(root)
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
    _init_workspace(root)
    _add_plan(root, step_id="tune", task="hparam_tune", status="completed")
    selection_report = _record_hparam_selection(root, write_report=True)
    final = experiments.finalize_experiment(root, selection_report)
    final_bytes = final.read_bytes()

    _record_hparam_selection(root, write_report=True, score="0.5")

    with pytest.raises(ValueError, match="selection report differs from its terminal binding"):
        experiments.experiment_status(root)
    assert final.read_bytes() == final_bytes


def test_experiment_finalize_rejects_selection_report_changed_after_snapshot(tmp_path, monkeypatch):
    root = tmp_path / "experiment"
    _init_workspace(root)
    _add_plan(root, step_id="tune", task="hparam_tune", status="completed")
    selection_report = _record_hparam_selection(root, write_report=True)
    manifest = root / "experiment.yaml"
    manifest_before = manifest.read_bytes()
    original_reader = experiments._hparam_selection_report

    def read_then_tamper(*args, **kwargs):
        payload = original_reader(*args, **kwargs)
        selection_report.write_text("# Tampered after verification\n")
        return payload

    monkeypatch.setattr(experiments, "_hparam_selection_report", read_then_tamper)

    with pytest.raises(ValueError, match="selection report changed during finalization"):
        experiments.finalize_experiment(root, selection_report)

    assert manifest.read_bytes() == manifest_before
    assert not (root / "events.jsonl").exists()
    assert not (root / "reports" / "final.md").exists()
    assert selection_report.read_text() == "# Tampered after verification\n"


def test_experiment_finalize_rechecks_selection_report_before_terminal_commit(tmp_path, monkeypatch):
    root = tmp_path / "experiment"
    _init_workspace(root)
    _add_plan(root, step_id="tune", task="hparam_tune", status="completed")
    selection_report = _record_hparam_selection(root, write_report=True)
    manifest = root / "experiment.yaml"
    manifest_before = manifest.read_bytes()
    original_replace = experiment_io.conditional_atomic_replace_text_at

    def publish_then_tamper(path, text, expected_sha256, *, remote=None, **kwargs):
        committed = original_replace(path, text, expected_sha256, remote=remote, **kwargs)
        if Path(path) == root / "reports" / "final.md" and committed:
            selection_report.write_text("# Tampered before terminal commit\n")
        return committed

    monkeypatch.setattr(experiment_io, "conditional_atomic_replace_text_at", publish_then_tamper)

    with pytest.raises(ValueError, match="selection report changed during finalization"):
        experiments.finalize_experiment(root, selection_report)

    assert manifest.read_bytes() == manifest_before
    assert (root / "reports" / "final.md").exists()
    assert yaml.safe_load(manifest.read_text())["experiment"].get("status") is None


def test_experiment_finalize_rechecks_ranking_before_terminal_commit(tmp_path, monkeypatch):
    root = tmp_path / "experiment"
    _init_workspace(root)
    _add_plan(root, step_id="tune", task="hparam_tune", status="completed")
    selection_report = _record_hparam_selection(root, write_report=True)
    ranking = root / "reports" / "ranking.csv"
    manifest = root / "experiment.yaml"
    manifest_before = manifest.read_bytes()
    original_replace = experiment_io.conditional_atomic_replace_text_at

    def publish_then_tamper(path, text, expected_sha256, *, remote=None, **kwargs):
        committed = original_replace(path, text, expected_sha256, remote=remote, **kwargs)
        if Path(path) == root / "reports" / "final.md" and committed:
            ranking.write_text(ranking.read_text().replace("0.25", "999", 1))
        return committed

    monkeypatch.setattr(experiment_io, "conditional_atomic_replace_text_at", publish_then_tamper)

    with pytest.raises(ValueError, match="ranking changed during finalization"):
        experiments.finalize_experiment(root, selection_report)

    assert manifest.read_bytes() == manifest_before
    assert (root / "reports" / "final.md").exists()
    assert yaml.safe_load(manifest.read_text())["experiment"].get("status") is None


def test_experiment_finalize_rechecks_checkpoint_before_terminal_commit(tmp_path, monkeypatch):
    root = tmp_path / "experiment"
    _init_workspace(root)
    _add_plan(root, step_id="tune", task="hparam_tune", status="completed")
    selection_report = _record_hparam_selection(root, write_report=True)
    checkpoint = Path(_read_manifest_rows(root)[0]["checkpoint_path"])
    manifest = root / "experiment.yaml"
    manifest_before = manifest.read_bytes()
    original_replace = experiment_io.conditional_atomic_replace_text_at

    def publish_then_tamper(path, text, expected_sha256, *, remote=None, **kwargs):
        committed = original_replace(path, text, expected_sha256, remote=remote, **kwargs)
        if Path(path) == root / "reports" / "final.md" and committed:
            checkpoint.write_bytes(b"tampered before terminal commit\n")
        return committed

    monkeypatch.setattr(experiment_io, "conditional_atomic_replace_text_at", publish_then_tamper)

    with pytest.raises(ValueError, match="Frozen checkpoint SHA-256 differs"):
        experiments.finalize_experiment(root, selection_report)

    assert manifest.read_bytes() == manifest_before
    assert (root / "reports" / "final.md").exists()
    assert yaml.safe_load(manifest.read_text())["experiment"].get("status") is None


def test_experiment_finalize_rechecks_run_manifest_before_terminal_commit(tmp_path, monkeypatch):
    root = tmp_path / "experiment"
    _init_workspace(root)
    _add_plan(root, step_id="tune", task="hparam_tune", status="completed")
    selection_report = _record_hparam_selection(root, write_report=True)
    manifest = root / "experiment.yaml"
    manifest_before = manifest.read_bytes()
    original_replace = experiment_io.conditional_atomic_replace_text_at

    def publish_then_tamper(path, text, expected_sha256, *, remote=None, **kwargs):
        committed = original_replace(path, text, expected_sha256, remote=remote, **kwargs)
        if Path(path) == root / "reports" / "final.md" and committed:
            rows = _read_manifest_rows(root)
            rows[0]["score"] = "0.5"
            write_rows(root / "run_manifest.tsv", rows)
        return committed

    monkeypatch.setattr(experiment_io, "conditional_atomic_replace_text_at", publish_then_tamper)

    with pytest.raises(RuntimeError, match="run manifest changed during finalization"):
        experiments.finalize_experiment(root, selection_report)

    assert manifest.read_bytes() == manifest_before
    assert (root / "reports" / "final.md").exists()
    assert yaml.safe_load(manifest.read_text())["experiment"].get("status") is None


def test_experiment_finalize_guards_terminal_commit_with_run_manifest_lock(tmp_path, monkeypatch):
    root = tmp_path / "experiment"
    _init_workspace(root)
    _add_plan(root, step_id="tune", task="hparam_tune", status="completed")
    selection_report = _record_hparam_selection(root, write_report=True)
    manifest = root / "experiment.yaml"
    manifest_before = manifest.read_bytes()
    original_replace = experiment_io.conditional_atomic_replace_text_at

    def tamper_at_terminal_commit(path, text, expected_sha256, *, remote=None, **kwargs):
        if Path(path) == manifest:
            rows = _read_manifest_rows(root)
            rows[0]["score"] = "0.5"
            write_rows(root / "run_manifest.tsv", rows)
        return original_replace(path, text, expected_sha256, remote=remote, **kwargs)

    monkeypatch.setattr(experiment_io, "conditional_atomic_replace_text_at", tamper_at_terminal_commit)

    with pytest.raises(RuntimeError, match="run manifest changed during finalization"):
        experiments.finalize_experiment(root, selection_report)

    assert manifest.read_bytes() == manifest_before
    assert (root / "reports" / "final.md").exists()
    assert yaml.safe_load(manifest.read_text())["experiment"].get("status") is None


def test_stale_finalizer_cannot_bind_an_overwritten_final_report(tmp_path, monkeypatch):
    root = tmp_path / "experiment"
    _init_workspace(root)
    _add_plan(root, step_id="train", status="completed")
    report_a = tmp_path / "report-a.md"
    report_b = tmp_path / "report-b.md"
    report_a.write_text("# Final A\n")
    report_b.write_text("# Final B\n")
    final_report = root / "reports" / "final.md"
    manifest = root / "experiment.yaml"
    real_replace = experiment_io.conditional_atomic_replace_text_at
    interleaved = False

    def overwrite_before_terminal_commit(path, text, expected_sha256, *, remote=None, **kwargs):
        nonlocal interleaved
        if Path(path) == manifest and not interleaved:
            interleaved = True
            assert real_replace(
                final_report,
                report_b.read_text(),
                hashlib.sha256(final_report.read_bytes()).hexdigest(),
                managed_root=kwargs["managed_root"],
                dependency_path=kwargs["dependency_path"],
                expected_dependency_sha256=kwargs["expected_dependency_sha256"],
                guard_path=manifest,
                expected_guard_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
            )
        return real_replace(path, text, expected_sha256, remote=remote, **kwargs)

    monkeypatch.setattr(experiment_io, "conditional_atomic_replace_text_at", overwrite_before_terminal_commit)

    with pytest.raises(RuntimeError, match="changed during finalization"):
        experiments.finalize_experiment(root, report_a)

    assert yaml.safe_load(manifest.read_text())["experiment"].get("status") is None
    assert final_report.read_text() == report_b.read_text()

    monkeypatch.setattr(experiment_io, "conditional_atomic_replace_text_at", real_replace)
    experiments.finalize_experiment(root, report_b)
    completed = yaml.safe_load(manifest.read_text())["experiment"]
    assert completed["final_report_sha256"] == hashlib.sha256(final_report.read_bytes()).hexdigest()


def test_completed_experiment_rejects_late_run_manifest_merge(tmp_path):
    root = tmp_path / "experiment"
    _init_workspace(root)
    _add_plan(root, step_id="train", status="completed")
    report = tmp_path / "report.md"
    report.write_text("# Final\n")
    experiments.finalize_experiment(root, report)
    before = _workspace_files(root)
    row = _read_manifest_rows(root)[0]

    with pytest.raises(ValueError, match="completed and cannot update canonical runs"):
        merge_run_manifest(root, [{**row, "health_status": "late observation"}])

    assert _workspace_files(root) == before
    assert experiments.experiment_status(root)["summary"]["state"] == "completed"


def test_inflight_run_manifest_merge_rechecks_owner_after_waiting_for_lock(tmp_path, monkeypatch):
    root = tmp_path / "experiment"
    _init_workspace(root)
    _add_plan(root, step_id="train", status="completed")
    row = _read_manifest_rows(root)[0]
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
    after_finalize = _workspace_files(root)
    allow_merge.set()
    worker.join(timeout=5)

    assert not worker.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], ValueError)
    assert "completed and cannot update canonical runs" in str(errors[0])
    assert _workspace_files(root) == after_finalize
    assert experiments.experiment_status(root)["summary"]["state"] == "completed"


def test_experiment_finalize_rechecks_rows_bound_to_run_manifest_snapshot(tmp_path, monkeypatch):
    root = tmp_path / "experiment"
    _init_workspace(root)
    _add_plan(root, step_id="tune", task="hparam_tune", status="completed")
    selection_report = _record_hparam_selection(root, write_report=True)
    manifest = root / "experiment.yaml"
    manifest_before = manifest.read_bytes()
    original_rows = experiments._managed_rows
    calls = 0

    def read_then_add_active_run(candidate, *, remote):
        nonlocal calls
        calls += 1
        rows = original_rows(candidate, remote=remote)
        if calls == 1:
            active = {**rows[0], "run_id": "run-999", "status": "running"}
            write_rows(root / "run_manifest.tsv", [*rows, active])
        return rows

    monkeypatch.setattr(experiments, "_managed_rows", read_then_add_active_run)

    with pytest.raises(ValueError, match="unresolved runs.*run-999"):
        experiments.finalize_experiment(root, selection_report)

    assert manifest.read_bytes() == manifest_before
    assert not (root / "reports" / "final.md").exists()


def test_experiment_finalize_allows_combined_or_failure_reports(tmp_path):
    mixed = tmp_path / "mixed"
    _init_workspace(mixed)
    _add_plan(mixed, step_id="tune", task="hparam_tune", status="completed")
    _add_plan(mixed, step_id="prepare", status="completed")
    selection_report = _record_hparam_selection(mixed, write_report=True)
    with pytest.raises(ValueError, match="cannot replace"):
        experiments.finalize_experiment(mixed, selection_report)
    combined = tmp_path / "combined.md"
    combined.write_text("# Combined report\n\nSelection and preparation summary.\n")
    assert experiments.finalize_experiment(mixed, combined).read_text() == combined.read_text()

    failed = tmp_path / "failed"
    _init_workspace(failed)
    _add_plan(failed, step_id="tune", task="hparam_tune", status="failed")
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
    _init_workspace(root)
    _add_plan(root, step_id="tune", task="hparam_tune", status="completed")
    _add_plan(root, step_id="prepare", status="completed")
    selection_report = _record_hparam_selection(root, write_report=True)
    aliased_report = selection_report.parent / ".." / "reports" / selection_report.name
    before = _workspace_files(root)

    with pytest.raises(ValueError, match="cannot replace the required combined experiment report"):
        experiments.finalize_experiment(root, aliased_report)

    assert _workspace_files(root) == before
    assert not (root / "reports" / "final.md").exists()


def test_experiment_finalize_rejects_selection_report_copy_as_failure_report(tmp_path):
    root = tmp_path / "failed"
    _init_workspace(root)
    _add_plan(root, step_id="tune", task="hparam_tune", status="failed")
    selection_report = root / "reports" / "hparam_selection.md"
    selection_report.parent.mkdir()
    selection_report.write_text("# Stale selection report\n")
    same_content = root / "failure.md"
    same_content.write_bytes(selection_report.read_bytes())
    before = _workspace_files(root)

    with pytest.raises(ValueError, match="cannot replace the required hparam failure report"):
        experiments.finalize_experiment(root, same_content)

    assert _workspace_files(root) == before
    assert not (root / "reports" / "final.md").exists()


def test_experiment_status_requires_combined_report_when_one_hparam_step_failed(tmp_path):
    root = tmp_path / "experiment"
    _init_workspace(root)
    _add_plan(root, step_id="selected", task="hparam_tune", status="completed")
    _add_plan(root, step_id="failed", task="hparam_tune", status="failed")
    selection_report = _record_hparam_selection(root, step_id="selected", write_report=True)

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
    _init_workspace(root)
    _add_plan(root, step_id="tune", task="hparam_tune", status="completed")
    selection_report = _record_hparam_selection(root, write_report=True)
    rows = _read_manifest_rows(root)
    rows[0]["status"] = "failed"
    write_rows(root / "run_manifest.tsv", rows)
    before = _workspace_files(root)

    with pytest.raises(ValueError, match="stale for all-failed step"):
        experiments.experiment_status(root)
    with pytest.raises(ValueError, match="stale for all-failed step"):
        experiments.finalize_experiment(root, selection_report)

    assert _workspace_files(root) == before


@pytest.mark.parametrize("mutation", ["missing", "tampered", "duplicate"])
def test_experiment_status_rejects_missing_or_drifted_hparam_ranking(tmp_path, mutation):
    root = tmp_path / "experiment"
    _init_workspace(root)
    _add_plan(root, step_id="tune", task="hparam_tune", status="completed")
    selection_report = _record_hparam_selection(root, write_report=True)
    ranking = root / "reports" / "ranking.csv"
    if mutation == "missing":
        ranking.unlink()
    elif mutation == "tampered":
        ranking.write_text(ranking.read_text().replace("0.25", "999", 1))
    else:
        lines = ranking.read_text().splitlines()
        ranking.write_text("\n".join([*lines, lines[1]]) + "\n")
    before = _workspace_files(root)

    snapshot = experiments.experiment_status(root)

    assert snapshot["summary"]["state"] == "ready_to_report"
    assert snapshot["decision"]["recommended_next"]["id"] == "hparam-select"
    with pytest.raises(ValueError, match="selection report is missing or differs"):
        experiments.finalize_experiment(root, selection_report)
    assert _workspace_files(root) == before


@pytest.mark.parametrize(
    ("field", "mutation"),
    [
        ("run_name", "tamper"),
        ("parameter_summary", "tamper"),
        ("version", "tamper"),
        ("config", "tamper"),
        ("runtime.lr", "tamper"),
        ("run_manifest", "tamper"),
        ("status", "tamper"),
        ("run_manifest", "remove"),
        ("status", "remove"),
        ("checkpoint_rank", "empty"),
        ("config", "remove"),
        ("runtime.lr", "remove"),
        ("unexpected", "add"),
    ],
)
def test_experiment_status_rejects_hparam_ranking_candidate_provenance_drift(tmp_path, field, mutation):
    root = tmp_path / "experiment"
    recipe = _write_public_hparam_recipe(
        root,
        {"runtime.lr": [1e-6]},
        selection_metric="val_loss",
        selection_mode="min",
    )
    plan_dir = root / "plans" / "tune"
    assert plans.build_plan(recipe_path=recipe, output_dir=plan_dir).exit_code == 0
    rows = _read_manifest_rows(root)
    for row in rows:
        row["status"] = "completed"
    write_rows(root / "run_manifest.tsv", rows)
    selection_report = _record_hparam_selection(root, step_id="unit-hparam-tune", write_report=True)
    ranking = root / "reports" / "ranking.csv"
    ranking_rows = read_rows(ranking, require_managed_identity=True)
    if mutation == "remove":
        for row in ranking_rows:
            row.pop(field)
    elif mutation == "empty":
        ranking_rows[0][field] = ""
    else:
        ranking_rows[0][field] = "tampered"
    write_rows(ranking, ranking_rows)

    snapshot = experiments.experiment_status(root)

    assert snapshot["summary"]["state"] == "ready_to_report"
    assert snapshot["decision"]["recommended_next"]["id"] == "hparam-select"
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
    recipe = _write_public_hparam_recipe(
        root,
        {"runtime.lr": [1e-6, 2e-6]},
        selection_metric=selection_metric,
        selection_mode=selection_mode,
        max_runs=2,
    )
    plan_dir = root / "plans" / "tune"
    assert plans.build_plan(recipe_path=recipe, output_dir=plan_dir).exit_code == 0
    rows = _read_manifest_rows(root)
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
    before = _workspace_files(root)

    with pytest.raises(ValueError, match="ranks disagree with selection mode"):
        experiments.experiment_status(root)
    with pytest.raises(ValueError, match="ranks disagree with selection mode"):
        experiments.finalize_experiment(root, report)
    assert _workspace_files(root) == before


@pytest.mark.parametrize(("field", "mutation"), [("run_manifest", "remove"), ("source", "add")])
def test_experiment_status_rejects_coherent_val_selection_provenance_drift(tmp_path, field, mutation):
    root = tmp_path / "experiment"
    _init_workspace(root)
    _add_plan(root, step_id="tune", task="hparam_tune", status="completed")
    selection_report = _record_hparam_selection(root, write_report=True)
    canonical = _read_manifest_rows(root)
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
    before = _workspace_files(root)

    with pytest.raises(ValueError, match="Canonical hparam selection evidence is invalid"):
        experiments.experiment_status(root)
    with pytest.raises(ValueError, match="Canonical hparam selection evidence is invalid"):
        experiments.finalize_experiment(root, selection_report)

    assert _workspace_files(root) == before


def test_experiment_status_keeps_final_report_blocker_experiment_wide(tmp_path):
    root = tmp_path / "experiment"
    _init_workspace(root)
    _add_plan(root, step_id="prepare", status="completed")
    _add_plan(root, step_id="evaluate", status="failed")

    snapshot = experiments.experiment_status(root)

    blocker = next(item for item in snapshot["blockers"] if item["code"] == "final_report_required")
    assert blocker["step_id"] is None
    assert blocker["run_ids"] == []
    assert all("final_report_required" not in run["blockers"] for run in snapshot["runs"])


def test_experiment_status_blocks_finalize_for_stopped_run_without_reason(tmp_path):
    root = tmp_path / "experiment"
    _init_workspace(root)
    _add_plan(root, step_id="train", status="stopped")
    before = _workspace_files(root)

    blocked = experiments.experiment_status(root)

    assert _workspace_files(root) == before
    assert blocked["summary"]["state"] == "blocked"
    assert blocked["decision"]["recommended_next"] is None
    assert blocked["decision"]["other_legal_actions"] == []
    assert blocked["decision"]["blocked_actions"] == ["finalize"]
    assert blocked["blockers"][0]["code"] == "missing_stop_reason"
    assert blocked["blockers"][0]["run_ids"] == ["run-000"]
    assert blocked["runs"][0]["blockers"] == ["missing_stop_reason"]

    rows = _read_manifest_rows(root)
    rows[0]["stop_reason"] = "manual stop after invalid labels"
    write_rows(root / "run_manifest.tsv", rows)
    ready = experiments.experiment_status(root)
    assert ready["summary"]["state"] == "ready_to_finalize"
    assert ready["decision"]["other_legal_actions"][0]["id"] == "experiment-finalize"


def test_experiment_status_rejects_completed_stopped_run_without_reason(tmp_path):
    root = tmp_path / "experiment"
    _init_workspace(root)
    _add_plan(root, step_id="train", status="stopped")
    manifest = yaml.safe_load((root / "experiment.yaml").read_text())
    manifest["experiment"].update({"status": "completed", "completed_at": "2026-08-25T02:00:00Z"})
    (root / "experiment.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))

    with pytest.raises(ValueError, match="stopped runs missing stop_reason"):
        experiments.experiment_status(root)

    rows = _read_manifest_rows(root)
    rows[0]["stop_reason"] = "manual stop after invalid labels"
    write_rows(root / "run_manifest.tsv", rows)
    assert experiments.experiment_status(root)["summary"]["state"] == "completed"
