from __future__ import annotations

import csv
import os
from pathlib import Path
import subprocess
import sys

import pytest

from agent_tools import experiment_io, experiments


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "agent_tools", *args], text=True, capture_output=True)


def _read_table(path: Path) -> list[dict[str, str]]:
    delimiter = "\t" if path.suffix == ".tsv" else ","
    with path.open(newline="") as file_obj:
        return list(csv.DictReader(file_obj, delimiter=delimiter))


def _experiment_spec(tmp_path: Path) -> Path:
    path = tmp_path / "experiment_spec.yaml"
    path.write_text(
        "id: unit\n"
        "title: Unit experiment\n"
        "objective: Exercise experiment workspace contracts.\n"
        "baseline:\n"
        "  type: none\n"
        "  rationale: Unit fixture.\n"
    )
    return path


def _workspace_files(root: Path) -> dict[Path, bytes]:
    return {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def test_experiment_init_creates_manifest(tmp_path: Path):
    spec = _experiment_spec(tmp_path.parent)
    result = _run("experiment-init", "--run-dir", str(tmp_path), "--spec", str(spec))

    assert result.returncode == 0, result.stderr
    rows = _read_table(tmp_path / "experiment_manifest.tsv")
    assert rows[0]["experiment_id"] == "unit"
    assert rows[0]["remote_host"] == ""
    assert (tmp_path / "reports").exists()
    assert (tmp_path / "run_manifest.tsv").read_text() == "step_id\trun_id\n"


def test_experiment_init_rejects_non_string_id_before_writing(tmp_path: Path):
    root = tmp_path / "workspace"
    spec = tmp_path / "numeric_id.yaml"
    spec.write_text(
        "id: 123\n"
        "title: Unit experiment\n"
        "objective: Exercise experiment workspace contracts.\n"
        "baseline: {type: none}\n"
    )

    with pytest.raises(ValueError, match="experiment.id must be a string"):
        experiments.init_experiment(root, spec)

    assert not root.exists()


def test_experiment_init_and_mutation_share_canonical_relative_root(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = (tmp_path / "workspace").resolve()

    manifest = experiments.init_experiment("workspace", _experiment_spec(tmp_path))
    monitored = experiments.monitor_experiment("workspace")

    assert manifest == root / "experiment_manifest.tsv"
    assert f"root: {root}" in (root / "experiment.yaml").read_text()
    assert monitored["run_dir"] == str(root)


def test_experiment_init_rejects_metadata_drift_for_existing_id(tmp_path: Path):
    spec = _experiment_spec(tmp_path.parent)
    assert _run("experiment-init", "--run-dir", str(tmp_path), "--spec", str(spec)).returncode == 0
    changed = tmp_path.parent / "changed_experiment_spec.yaml"
    changed.write_text(
        "id: unit\n"
        "title: Changed title\n"
        "objective: Changed objective.\n"
        "baseline:\n"
        "  type: none\n"
        "  rationale: Changed baseline.\n"
    )

    result = _run("experiment-init", "--run-dir", str(tmp_path), "--spec", str(changed))

    assert result.returncode == 1
    assert "differs from the existing experiment manifest" in result.stderr
    assert "# Unit experiment" in (tmp_path / "README.md").read_text()


def test_experiment_init_failure_leaves_workspace_unchanged(tmp_path: Path):
    (tmp_path / "experiment.yaml").write_text(
        "experiment:\n"
        "  id: existing\n"
        "  title: Existing\n"
        "  objective: Existing objective.\n"
        "  root: placeholder\n"
        "  baseline: {type: none}\n"
    )
    before = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}

    with pytest.raises(ValueError, match="different experiment"):
        experiments.init_experiment(tmp_path, _experiment_spec(tmp_path.parent))

    after = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    assert after == before
    assert not (tmp_path / "reports").exists()
    assert not (tmp_path / "wandb").exists()


def test_experiment_init_rejects_existing_root_drift_without_writing(tmp_path: Path):
    spec = _experiment_spec(tmp_path.parent)
    (tmp_path / "experiment.yaml").write_text(
        "experiment:\n"
        "  id: unit\n"
        "  title: Unit experiment\n"
        "  objective: Exercise experiment workspace contracts.\n"
        "  root: /different/root\n"
        "  baseline:\n"
        "    type: none\n"
        "    rationale: Unit fixture.\n"
    )
    before = _workspace_files(tmp_path)

    with pytest.raises(ValueError, match="experiment.root differs"):
        experiments.init_experiment(tmp_path, spec)

    assert _workspace_files(tmp_path) == before
    assert not (tmp_path / "reports").exists()


def test_experiment_init_rejects_duplicate_spec_keys_before_writing(tmp_path: Path):
    root = tmp_path / "workspace"
    spec = tmp_path / "duplicate_experiment.yaml"
    spec.write_text(
        "id: foreign\n"
        "id: unit\n"
        "title: Unit experiment\n"
        "objective: Exercise experiment workspace contracts.\n"
        "baseline: {type: none}\n"
    )

    with pytest.raises(ValueError, match="duplicate key: id"):
        experiments.init_experiment(root, spec)

    assert not root.exists()


@pytest.mark.parametrize("operation", ["init", "monitor"])
def test_experiment_mutation_rejects_duplicate_workspace_ownership_without_writing(tmp_path: Path, operation: str):
    spec = _experiment_spec(tmp_path.parent)
    experiments.init_experiment(tmp_path, spec)
    manifest = tmp_path / "experiment.yaml"
    manifest.write_text(manifest.read_text().replace("  id: unit\n", "  id: foreign\n  id: unit\n"))
    before = _workspace_files(tmp_path)

    with pytest.raises(ValueError, match="duplicate key"):
        if operation == "init":
            experiments.init_experiment(tmp_path, spec)
        else:
            experiments.monitor_experiment(tmp_path)

    assert _workspace_files(tmp_path) == before


@pytest.mark.parametrize(
    ("filename", "header", "operation"),
    [
        ("metrics_manifest.tsv", "trial_id\n", "index"),
        ("checkpoint_manifest.tsv", "run_id\n", "rank"),
    ],
)
def test_experiment_mutation_rejects_header_only_invalid_managed_table_before_writing(
    tmp_path: Path, filename: str, header: str, operation: str
):
    experiments.init_experiment(tmp_path, _experiment_spec(tmp_path.parent))
    (tmp_path / filename).write_text(header)
    before = _workspace_files(tmp_path)

    with pytest.raises(ValueError):
        if operation == "monitor":
            experiments.monitor_experiment(tmp_path)
        elif operation == "index":
            experiments.index_checkpoints(tmp_path)
        else:
            experiments.rank_experiment_candidates(tmp_path, metric="val_auroc", mode="max")

    assert _workspace_files(tmp_path) == before


def test_experiment_init_validates_existing_tables_before_writing(tmp_path: Path):
    spec = _experiment_spec(tmp_path.parent)
    experiments.init_experiment(tmp_path, spec)
    manifest = tmp_path / "experiment_manifest.tsv"
    lines = manifest.read_text().splitlines()
    manifest.write_text("\n".join([lines[0], lines[1], lines[1]]) + "\n")
    before = _workspace_files(tmp_path)

    with pytest.raises(ValueError, match="exactly one row"):
        experiments.init_experiment(tmp_path, spec)

    assert _workspace_files(tmp_path) == before


def test_experiment_monitor_delegates_manifest_validation_before_mutation(tmp_path: Path):
    spec = _experiment_spec(tmp_path.parent)
    experiments.init_experiment(tmp_path, spec)
    manifest = tmp_path / "experiment_manifest.tsv"
    header, row = manifest.read_text().splitlines()
    manifest.write_text(f"{header}\n{row}\textra\n")
    before = _workspace_files(tmp_path)

    with pytest.raises(ValueError):
        experiments.monitor_experiment(tmp_path)

    assert _workspace_files(tmp_path) == before


def test_experiment_reinit_rejects_readme_alias_before_writing(tmp_path: Path):
    spec = _experiment_spec(tmp_path.parent)
    experiments.init_experiment(tmp_path, spec)
    run_manifest = tmp_path / "run_manifest.tsv"
    run_manifest.write_text("experiment_id\tstep_id\trun_id\tstatus\nunit\ttrain\trun-000\trunning\n")
    readme = tmp_path / "README.md"
    readme.unlink()
    os.link(run_manifest, readme)
    before = _workspace_files(tmp_path)

    with pytest.raises(ValueError, match="independent regular files"):
        experiments.init_experiment(tmp_path, spec)

    assert _workspace_files(tmp_path) == before


def test_experiment_finalize_rejects_report_alias_before_writing(tmp_path: Path):
    spec = _experiment_spec(tmp_path.parent)
    experiments.init_experiment(tmp_path, spec)
    run_manifest = tmp_path / "run_manifest.tsv"
    run_manifest.write_text("experiment_id\tstep_id\trun_id\tstatus\nunit\ttrain\trun-000\tcompleted\n")
    final = tmp_path / "reports" / "final.md"
    os.link(run_manifest, final)
    report = tmp_path.parent / "final.md"
    report.write_text("# Final\n\nValidation-selected result.\n")
    before = _workspace_files(tmp_path)

    with pytest.raises(ValueError, match="independent regular files"):
        experiments.finalize_experiment(tmp_path, report)

    assert _workspace_files(tmp_path) == before


def test_experiment_registers_step_and_finalizes_completed_workspace(tmp_path: Path):
    spec = _experiment_spec(tmp_path.parent)
    assert _run("experiment-init", "--run-dir", str(tmp_path), "--spec", str(spec)).returncode == 0
    step = tmp_path.parent / "step.yaml"
    step.write_text(
        "id: analyze-results\n"
        "phase: analyze\n"
        "purpose: Summarize selected validation results.\n"
        "inputs: [reports/ranking.csv]\n"
        "outputs: [reports/final.md]\n"
    )
    registered = _run("experiment-register-step", "--run-dir", str(tmp_path), "--spec", str(step))
    assert registered.returncode == 0, registered.stderr
    assert (tmp_path / "steps" / "analyze-results" / "step.yaml").exists()
    (tmp_path / "run_manifest.tsv").write_text(
        "experiment_id\tstep_id\trun_id\tstatus\nunit\ttrain\trun-000\tfinished\n"
    )
    report = tmp_path.parent / "final.md"
    report.write_text("# Final\n\nValidation-selected result.\n")

    finalized = _run("experiment-finalize", "--run-dir", str(tmp_path), "--report", str(report))

    assert finalized.returncode == 0, finalized.stderr
    assert (tmp_path / "reports" / "final.md").read_text() == report.read_text()
    assert "status: completed" in (tmp_path / "experiment.yaml").read_text()


def test_interrupted_finalization_preserves_complete_experiment_manifest(tmp_path: Path, monkeypatch):
    spec = _experiment_spec(tmp_path.parent)
    experiments.init_experiment(tmp_path, spec)
    (tmp_path / "run_manifest.tsv").write_text(
        "experiment_id\tstep_id\trun_id\tstatus\nunit\ttrain\trun-000\tcompleted\n"
    )
    report = tmp_path.parent / "final.md"
    report.write_text("# Final\n\nValidation-selected result.\n")
    manifest = tmp_path / "experiment.yaml"
    before = manifest.read_bytes()
    real_replace = experiment_io.os.replace

    def interrupt_manifest_replace(source, target):
        if Path(target) == manifest:
            raise OSError("interrupted")
        return real_replace(source, target)

    monkeypatch.setattr(experiment_io.os, "replace", interrupt_manifest_replace)

    with pytest.raises(OSError, match="interrupted"):
        experiments.finalize_experiment(tmp_path, report)

    assert manifest.read_bytes() == before
    assert (tmp_path / "reports" / "final.md").read_text() == report.read_text()


@pytest.mark.parametrize(
    "existing",
    [
        "",
        "null\n",
        "{}\n",
        (
            "step:\n"
            "  id: analyze-results\n"
            "  phase: analyze\n"
            "  phase: train\n"
            "  purpose: Summarize selected validation results.\n"
            "experiment_id: unit\n"
            "recipe_path: ''\n"
            "plans: []\n"
        ),
    ],
)
def test_experiment_register_step_rejects_corrupt_existing_manifest_without_writing(tmp_path: Path, existing: str):
    experiments.init_experiment(tmp_path, _experiment_spec(tmp_path.parent))
    step = tmp_path.parent / "step.yaml"
    step.write_text(
        "id: analyze-results\n"
        "phase: analyze\n"
        "purpose: Summarize selected validation results.\n"
        "inputs: [reports/ranking.csv]\n"
        "outputs: [reports/final.md]\n"
    )
    target = tmp_path / "steps" / "analyze-results" / "step.yaml"
    target.parent.mkdir(parents=True)
    target.write_text(existing)
    before = _workspace_files(tmp_path)

    with pytest.raises(ValueError, match="step manifest"):
        experiments.register_experiment_step(tmp_path, step)

    assert _workspace_files(tmp_path) == before


def test_experiment_register_step_rejects_duplicate_spec_keys_before_writing(tmp_path: Path):
    root = tmp_path / "workspace"
    experiments.init_experiment(root, _experiment_spec(tmp_path))
    spec = tmp_path / "duplicate_step.yaml"
    spec.write_text(
        "id: analyze-results\n"
        "phase: train\n"
        "phase: analyze\n"
        "purpose: Summarize selected validation results.\n"
        "inputs: [reports/ranking.csv]\n"
        "outputs: [reports/final.md]\n"
    )
    before = _workspace_files(root)

    with pytest.raises(ValueError, match="duplicate key: phase"):
        experiments.register_experiment_step(root, spec)

    assert _workspace_files(root) == before


def test_experiment_finalize_rejects_missing_pid_status(tmp_path: Path):
    spec = _experiment_spec(tmp_path.parent)
    assert _run("experiment-init", "--run-dir", str(tmp_path), "--spec", str(spec)).returncode == 0
    (tmp_path / "run_manifest.tsv").write_text(
        "experiment_id\tstep_id\trun_id\tstatus\nunit\ttrain\trun-000\tmissing_pid\n"
    )
    report = tmp_path.parent / "missing_pid_final.md"
    report.write_text("# Final\n")

    result = _run("experiment-finalize", "--run-dir", str(tmp_path), "--report", str(report))

    assert result.returncode == 1
    assert "unresolved runs" in result.stderr
    assert "status: completed" not in (tmp_path / "experiment.yaml").read_text()


def test_experiment_finalize_rejects_workspace_without_managed_runs(tmp_path: Path):
    spec = _experiment_spec(tmp_path.parent)
    assert _run("experiment-init", "--run-dir", str(tmp_path), "--spec", str(spec)).returncode == 0
    report = tmp_path.parent / "empty_final.md"
    report.write_text("# Final\n")

    result = _run("experiment-finalize", "--run-dir", str(tmp_path), "--report", str(report))

    assert result.returncode == 1
    assert "no managed runs" in result.stderr
    assert "status: completed" not in (tmp_path / "experiment.yaml").read_text()


def test_experiment_finalize_validates_manifest_before_writing_report(tmp_path: Path):
    (tmp_path / "run_manifest.tsv").write_text("step_id\trun_id\tstatus\ntrain\trun-000\tfinished\n")
    report = tmp_path.parent / "final_without_manifest.md"
    report.write_text("# Final\n")

    result = _run("experiment-finalize", "--run-dir", str(tmp_path), "--report", str(report))

    assert result.returncode == 1
    assert "experiment.yaml is missing" in result.stderr
    assert not (tmp_path / "reports" / "final.md").exists()


def test_experiment_mutations_require_initialized_workspace_before_side_effects(tmp_path: Path, monkeypatch):
    wandb_calls = []
    monkeypatch.setattr(experiments.tracking, "wandb_runs", lambda *_args: wandb_calls.append(True) or [])
    actions = (
        lambda root: experiments.register_experiment_step(root, root / "missing-step.yaml"),
        lambda root: experiments.finalize_experiment(root, root / "missing-report.md"),
        lambda root: experiments.sync_wandb_runs(root, entity="entity", project="project"),
        lambda root: experiments.index_checkpoints(root),
        lambda root: experiments.monitor_experiment(root),
        lambda root: experiments.rank_experiment_candidates(root, metric="val_auroc", mode="max"),
    )

    for index, action in enumerate(actions):
        root = tmp_path / str(index)
        root.mkdir()
        before = _workspace_files(root)
        with pytest.raises(ValueError, match="experiment.yaml is missing"):
            action(root)
        assert _workspace_files(root) == before

    assert wandb_calls == []


def test_experiment_mutation_rejects_empty_legacy_table_without_writing(tmp_path: Path):
    experiments.init_experiment(tmp_path, _experiment_spec(tmp_path.parent))
    (tmp_path / "trial_status.tsv").touch()
    before = _workspace_files(tmp_path)

    with pytest.raises(ValueError, match="read-only"):
        experiments.monitor_experiment(tmp_path)

    assert _workspace_files(tmp_path) == before


def test_experiment_mutation_rejects_manifest_root_drift_without_writing(tmp_path: Path):
    experiments.init_experiment(tmp_path, _experiment_spec(tmp_path.parent))
    manifest = tmp_path / "experiment_manifest.tsv"
    manifest.write_text(manifest.read_text().replace(str(tmp_path), "/different/root"))
    before = _workspace_files(tmp_path)

    with pytest.raises(ValueError, match="root differs"):
        experiments.monitor_experiment(tmp_path)

    assert _workspace_files(tmp_path) == before


@pytest.mark.parametrize("alias", ["symlink", "hardlink"])
def test_experiment_mutation_rejects_experiment_manifest_alias_before_writing(tmp_path: Path, monkeypatch, alias: str):
    experiments.init_experiment(tmp_path, _experiment_spec(tmp_path.parent))
    manifest = tmp_path / "experiment.yaml"
    outside = tmp_path.parent / f"{alias}_experiment.yaml"
    outside.write_text(manifest.read_text())
    manifest.unlink()
    if alias == "symlink":
        manifest.symlink_to(outside)
    else:
        os.link(outside, manifest)
    observation_calls = []
    monkeypatch.setattr(
        experiments.tracking,
        "experiment_run_rows",
        lambda *_args, **_kwargs: observation_calls.append(True) or [],
    )
    before = _workspace_files(tmp_path)

    with pytest.raises(ValueError, match="independent regular files"):
        experiments.monitor_experiment(tmp_path)

    assert observation_calls == []
    assert _workspace_files(tmp_path) == before


def test_experiment_remote_mutation_preflights_manifest_before_reading_workspace_identity(monkeypatch):
    root = Path("/wujidata/remote_run")
    reads = []

    def _reject_alias(root_arg, paths, *, remote=None):
        assert root_arg == root
        assert paths == [root / "experiment.yaml"]
        assert remote == "baichuan3"
        raise ValueError("Managed output paths must be independent regular files")

    monkeypatch.setattr(experiments.exp_io, "validate_managed_output_paths", _reject_alias)
    monkeypatch.setattr(
        experiments.exp_io,
        "read_text_at",
        lambda *args, **kwargs: reads.append((args, kwargs)) or "",
    )

    with pytest.raises(ValueError, match="independent regular files"):
        experiments.monitor_experiment(root, remote="baichuan3")

    assert reads == []


def test_experiment_monitor_preflights_canonical_outputs_before_observation(tmp_path: Path, monkeypatch):
    experiments.init_experiment(tmp_path, _experiment_spec(tmp_path.parent))
    experiment_io.write_rows_at(
        tmp_path / "run_manifest.tsv",
        [{"experiment_id": "unit", "step_id": "train-model", "run_id": "run-000", "status": "running"}],
    )
    (tmp_path / "run_matrix.csv").mkdir()
    before = _workspace_files(tmp_path)
    observation_calls = []
    monkeypatch.setattr(
        experiments.tracking,
        "monitor_run_row",
        lambda *_args, **_kwargs: observation_calls.append(True),
    )

    with pytest.raises(ValueError, match="independent regular files"):
        experiments.monitor_experiment(tmp_path)

    assert observation_calls == []
    assert _workspace_files(tmp_path) == before


@pytest.mark.parametrize(
    ("operation", "table"),
    [
        ("index", "metrics_manifest.tsv"),
        ("rank", "checkpoint_manifest.tsv"),
    ],
)
def test_experiment_rejects_aliased_evidence_before_scan_or_rank(
    tmp_path: Path, monkeypatch, operation: str, table: str
):
    experiments.init_experiment(tmp_path, _experiment_spec(tmp_path.parent))
    experiment_io.write_rows_at(
        tmp_path / "run_manifest.tsv",
        [{"experiment_id": "unit", "step_id": "train-model", "run_id": "run-000", "status": "running"}],
    )
    outside = tmp_path / "outside.tsv"
    if table == "metrics_manifest.tsv":
        experiment_io.write_rows_at(
            outside,
            [{"step_id": "train-model", "run_id": "run-000", "metric": "val_auroc", "value": "0.9"}],
        )
    else:
        experiment_io.write_rows_at(
            outside,
            [{"step_id": "train-model", "run_id": "run-000", "checkpoint_path": "/tmp/epoch=1.ckpt"}],
        )
    (tmp_path / table).symlink_to(outside)
    calls = []
    monkeypatch.setattr(
        experiments.tracking,
        "checkpoint_rows",
        lambda *_args, **_kwargs: calls.append("checkpoint") or [],
    )
    monkeypatch.setattr(
        experiments.tracking,
        "experiment_run_rows",
        lambda *_args, **_kwargs: calls.append("rank") or [],
    )
    before = _workspace_files(tmp_path)

    with pytest.raises(ValueError, match="independent regular files"):
        if operation == "index":
            experiments.index_checkpoints(tmp_path)
        else:
            experiments.rank_experiment_candidates(tmp_path, metric="val_auroc", mode="max")

    assert calls == []
    assert _workspace_files(tmp_path) == before


def test_experiment_init_remote_writes_remote_not_local(tmp_path: Path, monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        if "mkdir -p" in command[-1] or "seen_inodes" in command[-1]:
            return subprocess.CompletedProcess(command, 0, "", "")
        return subprocess.CompletedProcess(command, experiment_io.REMOTE_MISSING_RETURN_CODE, "", "")

    monkeypatch.setattr("agent_tools.experiment_io.subprocess.run", fake_run)

    experiments.init_experiment("/wujidata/remote_run", _experiment_spec(tmp_path), remote="baichuan3")

    assert all(command[:2] == ["ssh", "baichuan3"] for command, _kwargs in calls)
    assert any("mkdir -p" in command[-1] for command, _kwargs in calls)
    write_targets = [command[-1] for command, _kwargs in calls if "cat >" in command[-1]]
    assert any("/wujidata/remote_run/experiment.yaml" in target for target in write_targets)
    assert any("/wujidata/remote_run/README.md" in target for target in write_targets)
    assert any("/wujidata/remote_run/events.jsonl" in target for target in write_targets)
    assert any("/wujidata/remote_run/experiment_manifest.tsv" in target for target in write_targets)
    assert not (tmp_path / "reports").exists()


def test_experiment_remote_read_failure_is_not_treated_as_missing(tmp_path: Path, monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 255, "", "connection failed")

    monkeypatch.setattr("agent_tools.experiment_io.subprocess.run", fake_run)

    with pytest.raises(RuntimeError, match="SSH read failed"):
        experiments.init_experiment("/wujidata/remote_run", _experiment_spec(tmp_path), remote="baichuan3")

    assert len(calls) == 1
    assert not any("mkdir -p" in command[-1] or "cat >" in command[-1] for command, _kwargs in calls)


def test_experiment_remote_finalize_rejects_relative_report_before_ssh(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "agent_tools.experiment_io.subprocess.run",
        lambda command, **kwargs: calls.append((command, kwargs))
        or subprocess.CompletedProcess(command, experiment_io.REMOTE_MISSING_RETURN_CODE, "", ""),
    )

    with pytest.raises(ValueError, match="Remote final report path must be absolute"):
        experiments.finalize_experiment(
            "/wujidata/remote_run",
            "reports/final.md",
            remote="baichuan3",
        )

    assert calls == []


def test_remote_directory_probe_fails_closed(monkeypatch):
    monkeypatch.setattr(
        "agent_tools.experiment_io.subprocess.run",
        lambda command, **_kwargs: subprocess.CompletedProcess(command, 1, "", "permission denied"),
    )

    with pytest.raises(RuntimeError, match="SSH directory probe failed"):
        experiment_io.remote_dir_nonempty(Path("/wujidata/remote_run"), "baichuan3")
