from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

import pytest

from agent_tools import experiment_io, experiment_workspace, experiments


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


def _research_entry(
    tmp_path: Path,
    entry_id: str,
    *,
    kind: str = "observation",
    body: str = "Validation loss decreased after the scheduler change.",
    authority: str | None = None,
    scope: dict | None = None,
    supersedes: list[str] | None = None,
) -> Path:
    entry = {
        "id": entry_id,
        "recorded_at": "2026-07-25T02:03:04Z",
        "kind": kind,
        "title": f"Research entry {entry_id}",
        "actor": "agent:test",
        "source": "codex-task:test",
        "evidence": [{"label": "validation report", "locator": "reports/validation.md"}],
        "body": body,
    }
    if authority is not None:
        entry["authority"] = authority
    if scope is not None:
        entry["scope"] = scope
    if supersedes is not None:
        entry["supersedes"] = supersedes
    path = tmp_path / f"{entry_id}.yaml"
    path.write_text(json.dumps(entry))
    return path


def test_experiment_init_creates_manifest(tmp_path: Path):
    spec = _experiment_spec(tmp_path.parent)
    result = _run("experiment-init", "--run-dir", str(tmp_path), "--spec", str(spec))

    assert result.returncode == 0, result.stderr
    rows = _read_table(tmp_path / "experiment_manifest.tsv")
    assert rows[0]["experiment_id"] == "unit"
    assert rows[0]["remote_host"] == ""
    assert (tmp_path / "reports").exists()
    assert (tmp_path / "run_manifest.tsv").read_text() == "step_id\trun_id\n"
    assert (tmp_path / "RESEARCH_LOG.md").read_text().startswith("# Research Log\n")
    assert "`run_manifest.tsv` remains the sole authority" in (tmp_path / "RESEARCH_LOG.md").read_text()
    assert "`RESEARCH_LOG.md`" in (tmp_path / "README.md").read_text()


def test_experiment_note_appends_idempotently_and_preserves_evidence_locator(tmp_path: Path):
    root = tmp_path / "workspace"
    experiments.init_experiment(root, _experiment_spec(tmp_path))
    entry = _research_entry(tmp_path, "obs-001")

    first = experiments.append_experiment_note(root, entry)
    after_first = (root / "RESEARCH_LOG.md").read_bytes()
    second = experiments.append_experiment_note(root, entry)

    assert first == {
        "path": str(root / "RESEARCH_LOG.md"),
        "entry_id": "obs-001",
        "appended": True,
    }
    assert second["appended"] is False
    assert (root / "RESEARCH_LOG.md").read_bytes() == after_first
    text = after_first.decode()
    assert text.count('id="obs-001"') == 1
    assert "- Label: validation report" in text
    assert "  Locator: reports/validation.md" in text


def test_experiment_note_cli_reports_append_and_idempotent_retry(tmp_path: Path):
    root = tmp_path / "workspace"
    experiments.init_experiment(root, _experiment_spec(tmp_path))
    entry = _research_entry(tmp_path, "obs-cli")

    first = _run("experiment-note", "--run-dir", str(root), "--entry", str(entry))
    second = _run("experiment-note", "--run-dir", str(root), "--entry", str(entry))

    assert first.returncode == 0, first.stderr
    assert f"Research log {root / 'RESEARCH_LOG.md'}: obs-cli appended" in first.stdout
    assert second.returncode == 0, second.stderr
    assert f"Research log {root / 'RESEARCH_LOG.md'}: obs-cli already present" in second.stdout


@pytest.mark.parametrize("entry_kind", ["missing", "inline", "directory", "stdin"])
@pytest.mark.parametrize("remote", [None, "unreachable-host"])
def test_experiment_note_cli_rejects_non_file_entry_without_mutation(
    tmp_path: Path,
    entry_kind: str,
    remote: str | None,
):
    root = tmp_path / "workspace"
    experiments.init_experiment(root, _experiment_spec(tmp_path))
    before = _workspace_files(root)
    entry = {
        "missing": str(tmp_path / "missing-entry.yaml"),
        "inline": "id: obs-inline",
        "directory": str(tmp_path),
        "stdin": "-",
    }[entry_kind]
    args = ["experiment-note", "--run-dir", str(root), "--entry", entry]
    if remote is not None:
        args.extend(["--remote", remote])

    result = _run(*args)

    assert result.returncode == 2
    assert "--entry must be an existing local YAML file path" in result.stderr
    assert "Traceback" not in result.stdout
    assert "Traceback" not in result.stderr
    assert _workspace_files(root) == before


def test_experiment_note_rejects_same_id_with_different_content_without_writing(tmp_path: Path):
    root = tmp_path / "workspace"
    experiments.init_experiment(root, _experiment_spec(tmp_path))
    experiments.append_experiment_note(root, _research_entry(tmp_path, "obs-001"))
    before = (root / "RESEARCH_LOG.md").read_bytes()
    changed = _research_entry(tmp_path, "obs-001", body="A different interpretation.")

    with pytest.raises(ValueError, match="already exists with different content"):
        experiments.append_experiment_note(root, changed)

    assert (root / "RESEARCH_LOG.md").read_bytes() == before


def test_experiment_note_rejects_same_id_with_ambiguously_rendered_evidence(tmp_path: Path):
    root = tmp_path / "workspace"
    experiments.init_experiment(root, _experiment_spec(tmp_path))
    entry_path = _research_entry(tmp_path, "obs-evidence")
    entry = json.loads(entry_path.read_text())
    entry["evidence"] = [{"label": "a: b", "locator": "c"}]
    entry_path.write_text(json.dumps(entry))
    experiments.append_experiment_note(root, entry_path)
    before = (root / "RESEARCH_LOG.md").read_bytes()
    entry["evidence"] = [{"label": "a", "locator": "b: c"}]
    entry_path.write_text(json.dumps(entry))

    with pytest.raises(ValueError, match="already exists with different content"):
        experiments.append_experiment_note(root, entry_path)

    assert (root / "RESEARCH_LOG.md").read_bytes() == before


def test_experiment_note_idempotency_uses_normalized_content(tmp_path: Path):
    root = tmp_path / "workspace"
    experiments.init_experiment(root, _experiment_spec(tmp_path))
    entry_path = _research_entry(tmp_path, "obs-normalized")
    experiments.append_experiment_note(root, entry_path)
    before = (root / "RESEARCH_LOG.md").read_bytes()
    entry = json.loads(entry_path.read_text())
    entry["recorded_at"] = "2026-07-25T02:03:04+00:00"
    entry["body"] = f"\n{entry['body']}\n"
    entry["supersedes"] = []
    entry_path.write_text(json.dumps(entry))

    result = experiments.append_experiment_note(root, entry_path)

    assert result["appended"] is False
    assert (root / "RESEARCH_LOG.md").read_bytes() == before


def test_experiment_note_requires_decision_authority_and_existing_superseded_entries(tmp_path: Path):
    root = tmp_path / "workspace"
    experiments.init_experiment(root, _experiment_spec(tmp_path))
    before = (root / "RESEARCH_LOG.md").read_bytes()

    with pytest.raises(ValueError, match="require authority"):
        experiments.append_experiment_note(root, _research_entry(tmp_path, "decision-001", kind="decision"))
    with pytest.raises(ValueError, match="unknown entry ids"):
        experiments.append_experiment_note(
            root,
            _research_entry(
                tmp_path,
                "decision-002",
                kind="decision",
                authority="human",
                supersedes=["missing-entry"],
            ),
        )

    assert (root / "RESEARCH_LOG.md").read_bytes() == before


def test_experiment_note_appends_correction_without_rewriting_superseded_entry(tmp_path: Path):
    root = tmp_path / "workspace"
    experiments.init_experiment(root, _experiment_spec(tmp_path))
    experiments.append_experiment_note(root, _research_entry(tmp_path, "interpretation-001"))
    original = (root / "RESEARCH_LOG.md").read_text()

    result = experiments.append_experiment_note(
        root,
        _research_entry(
            tmp_path,
            "interpretation-002",
            kind="interpretation",
            body="The earlier interpretation did not control for cohort size.",
            supersedes=["interpretation-001"],
        ),
    )

    text = (root / "RESEARCH_LOG.md").read_text()
    assert result["appended"] is True
    assert text.startswith(original)
    assert "- Supersedes: `interpretation-001`" in text


def test_experiment_note_validates_step_and_run_scope(tmp_path: Path):
    root = tmp_path / "workspace"
    experiments.init_experiment(root, _experiment_spec(tmp_path))
    step = tmp_path / "step.yaml"
    step.write_text(
        "id: train-model\n"
        "phase: train\n"
        "purpose: Train a candidate.\n"
        "inputs: [data]\n"
        "outputs: [checkpoint]\n"
    )
    experiments.register_experiment_step(root, step)
    (root / "run_manifest.tsv").write_text(
        "experiment_id\tstep_id\trun_id\tstatus\nunit\ttrain-model\trun-001\tcompleted\n"
    )

    result = experiments.append_experiment_note(
        root,
        _research_entry(
            tmp_path,
            "obs-scoped",
            scope={"step_id": "train-model", "run_ids": ["run-001"]},
        ),
    )

    assert result["appended"] is True
    text = (root / "RESEARCH_LOG.md").read_text()
    assert "- Step: `train-model`" in text
    assert "- Runs:\n  - `run-001`" in text

    with pytest.raises(ValueError, match="unknown managed runs"):
        experiments.append_experiment_note(
            root,
            _research_entry(
                tmp_path,
                "obs-unknown-run",
                scope={"step_id": "train-model", "run_ids": ["run-999"]},
            ),
        )

    before = (root / "RESEARCH_LOG.md").read_bytes()
    with pytest.raises(ValueError, match="scope.run_ids must be a non-empty list"):
        experiments.append_experiment_note(
            root,
            _research_entry(
                tmp_path,
                "obs-null-run-scope",
                scope={"step_id": "train-model", "run_ids": None},
            ),
        )
    assert (root / "RESEARCH_LOG.md").read_bytes() == before


def test_experiment_note_rejects_same_id_with_ambiguously_rendered_run_ids(tmp_path: Path):
    root = tmp_path / "workspace"
    experiments.init_experiment(root, _experiment_spec(tmp_path))
    step = tmp_path / "step.yaml"
    step.write_text(
        "id: train-model\n"
        "phase: train\n"
        "purpose: Train a candidate.\n"
        "inputs: [data]\n"
        "outputs: [checkpoint]\n"
    )
    experiments.register_experiment_step(root, step)
    (root / "run_manifest.tsv").write_text(
        "experiment_id\tstep_id\trun_id\tstatus\n"
        "unit\ttrain-model\ta`, `b\tcompleted\n"
        "unit\ttrain-model\ta\tcompleted\n"
        "unit\ttrain-model\tb\tcompleted\n"
    )
    entry_path = _research_entry(
        tmp_path,
        "obs-run-scope",
        scope={"step_id": "train-model", "run_ids": ["a`, `b"]},
    )
    experiments.append_experiment_note(root, entry_path)
    before = (root / "RESEARCH_LOG.md").read_bytes()
    entry = json.loads(entry_path.read_text())
    entry["scope"]["run_ids"] = ["a", "b"]
    entry_path.write_text(json.dumps(entry))

    with pytest.raises(ValueError, match="already exists with different content"):
        experiments.append_experiment_note(root, entry_path)

    assert (root / "RESEARCH_LOG.md").read_bytes() == before


def test_experiment_note_allows_retrospective_conclusion_after_finalization(tmp_path: Path):
    root = tmp_path / "workspace"
    experiments.init_experiment(root, _experiment_spec(tmp_path))
    (root / "run_manifest.tsv").write_text("experiment_id\tstep_id\trun_id\tstatus\nunit\ttrain\trun-000\tcompleted\n")
    report = tmp_path / "final.md"
    report.write_text("# Final\n\nValidation-selected result.\n")
    experiments.finalize_experiment(root, report)

    result = experiments.append_experiment_note(
        root,
        _research_entry(tmp_path, "conclusion-001", kind="conclusion"),
    )

    assert result["appended"] is True
    assert "status: completed" in (root / "experiment.yaml").read_text()
    assert "conclusion-001" in (root / "RESEARCH_LOG.md").read_text()


def test_experiment_note_cannot_change_canonical_run_state(tmp_path: Path):
    root = tmp_path / "workspace"
    experiments.init_experiment(root, _experiment_spec(tmp_path))
    run_manifest = root / "run_manifest.tsv"
    run_manifest.write_text("experiment_id\tstep_id\trun_id\tstatus\nunit\ttrain\trun-000\trunning\n")
    before = run_manifest.read_bytes()

    experiments.append_experiment_note(
        root,
        _research_entry(tmp_path, "conclusion-001", kind="conclusion", body="The experiment is complete."),
    )

    assert run_manifest.read_bytes() == before
    with pytest.raises(ValueError, match="unresolved runs"):
        experiments.finalize_experiment(root, tmp_path / "missing-report.md")


def test_experiment_finalization_does_not_read_research_log_as_state(tmp_path: Path):
    root = tmp_path / "workspace"
    experiments.init_experiment(root, _experiment_spec(tmp_path))
    (root / "run_manifest.tsv").write_text("experiment_id\tstep_id\trun_id\tstatus\nunit\ttrain\trun-000\tcompleted\n")
    (root / "RESEARCH_LOG.md").write_text("corrupt narrative that claims the run is active\n")
    report = tmp_path / "final.md"
    report.write_text("# Final\n\nCanonical state is terminal.\n")

    target = experiments.finalize_experiment(root, report)

    assert target.read_text() == report.read_text()
    assert "status: completed" in (root / "experiment.yaml").read_text()


def test_experiment_note_rejects_embedded_entry_marker_before_writing(tmp_path: Path):
    root = tmp_path / "workspace"
    experiments.init_experiment(root, _experiment_spec(tmp_path))
    before = (root / "RESEARCH_LOG.md").read_bytes()
    injected_marker = '<!-- agent-tools-research-entry id="injected" ' f'sha256="{"0" * 64}" -->'

    with pytest.raises(ValueError, match="digest differs"):
        experiments.append_experiment_note(
            root,
            _research_entry(tmp_path, "obs-marker", body=injected_marker),
        )

    assert (root / "RESEARCH_LOG.md").read_bytes() == before


@pytest.mark.parametrize("target_name", ["RESEARCH_LOG.md", ".RESEARCH_LOG.md.cas.lock"])
def test_experiment_note_rejects_aliased_log_or_lock_before_writing(tmp_path: Path, target_name: str):
    root = tmp_path / "workspace"
    experiments.init_experiment(root, _experiment_spec(tmp_path))
    target = root / target_name
    if target.exists():
        target.unlink()
    os.link(root / "run_manifest.tsv", target)
    before = _workspace_files(root)

    with pytest.raises(ValueError, match="Managed file is missing or aliased"):
        experiments.append_experiment_note(root, _research_entry(tmp_path, "obs-alias"))

    assert _workspace_files(root) == before


def test_concurrent_experiment_notes_preserve_both_entries(tmp_path: Path):
    root = tmp_path / "workspace"
    experiments.init_experiment(root, _experiment_spec(tmp_path))
    entries = [_research_entry(tmp_path, entry_id) for entry_id in ("obs-a", "obs-b")]
    errors = []

    def append(entry: Path) -> None:
        try:
            experiments.append_experiment_note(root, entry)
        except Exception as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    threads = [threading.Thread(target=append, args=(entry,)) for entry in entries]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert not errors
    assert not any(thread.is_alive() for thread in threads)
    text = (root / "RESEARCH_LOG.md").read_text()
    assert text.count('id="obs-a"') == 1
    assert text.count('id="obs-b"') == 1


def test_historical_workspace_creates_research_log_only_on_explicit_note(tmp_path: Path):
    root = tmp_path / "workspace"
    spec = _experiment_spec(tmp_path)
    experiments.init_experiment(root, spec)
    (root / "RESEARCH_LOG.md").unlink()

    experiments.init_experiment(root, spec)
    experiments.monitor_experiment(root)

    assert not (root / "RESEARCH_LOG.md").exists()

    result = experiments.append_experiment_note(root, _research_entry(tmp_path, "retrospective-001"))

    assert result["appended"] is True
    assert (root / "RESEARCH_LOG.md").read_text().startswith("# Research Log\n")


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ({"unexpected": True}, "Unexpected research log entry fields"),
        ({"kind": "guess"}, "kind must be one of"),
        ({"evidence": []}, "evidence must be a non-empty list"),
        ({"scope": None}, "scope must be a non-empty mapping"),
        ({"scope": {"step_id": None}}, "scope.step_id must be a non-empty string"),
        ({"scope": {"run_ids": None}}, "requires scope.step_id"),
        ({"scope": {"run_ids": ["run-001"]}}, "requires scope.step_id"),
    ],
)
def test_experiment_note_rejects_invalid_closed_schema_without_writing(
    tmp_path: Path,
    mutation: dict,
    error: str,
):
    root = tmp_path / "workspace"
    experiments.init_experiment(root, _experiment_spec(tmp_path))
    entry_path = _research_entry(tmp_path, "invalid-001")
    entry = json.loads(entry_path.read_text())
    entry.update(mutation)
    entry_path.write_text(json.dumps(entry))
    before = (root / "RESEARCH_LOG.md").read_bytes()

    with pytest.raises(ValueError, match=error):
        experiments.append_experiment_note(root, entry_path)

    assert (root / "RESEARCH_LOG.md").read_bytes() == before


def test_experiment_note_rejects_aliased_scoped_step_before_reading_it(tmp_path: Path):
    root = tmp_path / "workspace"
    experiments.init_experiment(root, _experiment_spec(tmp_path))
    step = tmp_path / "step.yaml"
    step.write_text(
        "id: train-model\n"
        "phase: train\n"
        "purpose: Train a candidate.\n"
        "inputs: [data]\n"
        "outputs: [checkpoint]\n"
    )
    experiments.register_experiment_step(root, step)
    step_manifest = root / "steps" / "train-model" / "step.yaml"
    step_manifest.unlink()
    os.link(root / "run_manifest.tsv", step_manifest)
    before = (root / "RESEARCH_LOG.md").read_bytes()

    with pytest.raises(ValueError, match="Managed file is missing or aliased"):
        experiments.append_experiment_note(
            root,
            _research_entry(tmp_path, "obs-step-alias", scope={"step_id": "train-model"}),
        )

    assert (root / "RESEARCH_LOG.md").read_bytes() == before


def test_remote_research_log_retries_conflict_and_preserves_competing_entry(tmp_path: Path, monkeypatch):
    root = Path("/remote/workspace")
    entry = json.loads(_research_entry(tmp_path, "obs-new").read_text())
    competing = dict(entry)
    competing.update({"id": "obs-competing", "title": "Competing observation"})
    competing_block = experiment_workspace._research_log_block(competing, "unit")
    competing_digest = experiment_workspace.hashlib.sha256(competing_block.encode()).hexdigest()
    state = {
        "text": experiment_workspace.RESEARCH_LOG_PREAMBLE,
        "attempts": 0,
        "paths": [],
    }

    def validate(_root, paths, *, remote=None):
        state["paths"] = [Path(path) for path in paths]
        assert remote == "unit-host"

    def commit(_path, replacement, _expected_sha256, *, remote=None, **_kwargs):
        state["attempts"] += 1
        if state["attempts"] == 1:
            marker = "<!-- agent-tools-research-entry " f'id="obs-competing" sha256="{competing_digest}" -->\n'
            state["text"] += marker + competing_block
            return False
        state["text"] = replacement
        return True

    monkeypatch.setattr(experiment_workspace.exp_io, "validate_managed_output_paths", validate)
    monkeypatch.setattr(experiment_workspace.exp_io, "path_exists_at", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        experiment_workspace.exp_io,
        "read_text_at",
        lambda *_args, **_kwargs: state["text"],
    )
    monkeypatch.setattr(experiment_workspace.exp_io, "conditional_atomic_replace_text_at", commit)

    _path, _entry_id, appended = experiment_workspace.append_research_log(
        root,
        entry,
        experiment_id="unit",
        managed_rows=[],
        remote="unit-host",
    )

    assert appended is True
    assert state["attempts"] == 2
    assert root / "RESEARCH_LOG.md.lock" in state["paths"]
    assert root / ".RESEARCH_LOG.md.cas.lock" in state["paths"]
    assert state["text"].count('id="obs-competing"') == 1
    assert state["text"].count('id="obs-new"') == 1


def test_uncertain_remote_research_log_commit_is_idempotent_on_retry(tmp_path: Path, monkeypatch):
    root = Path("/remote/workspace")
    entry = json.loads(_research_entry(tmp_path, "obs-timeout").read_text())
    state = {"text": experiment_workspace.RESEARCH_LOG_PREAMBLE, "raise_timeout": True}

    def commit(_path, replacement, _expected_sha256, *, remote=None, **_kwargs):
        state["text"] = replacement
        if state["raise_timeout"]:
            state["raise_timeout"] = False
            raise RuntimeError("SSH response timed out after commit")
        return True

    monkeypatch.setattr(experiment_workspace.exp_io, "validate_managed_output_paths", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(experiment_workspace.exp_io, "path_exists_at", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        experiment_workspace.exp_io,
        "read_text_at",
        lambda *_args, **_kwargs: state["text"],
    )
    monkeypatch.setattr(experiment_workspace.exp_io, "conditional_atomic_replace_text_at", commit)

    with pytest.raises(RuntimeError, match="timed out"):
        experiment_workspace.append_research_log(
            root,
            entry,
            experiment_id="unit",
            managed_rows=[],
            remote="unit-host",
        )

    _path, _entry_id, appended = experiment_workspace.append_research_log(
        root,
        entry,
        experiment_id="unit",
        managed_rows=[],
        remote="unit-host",
    )

    assert appended is False
    assert state["text"].count('id="obs-timeout"') == 1


def test_remote_research_log_fails_after_three_conflicts(tmp_path: Path, monkeypatch):
    root = Path("/remote/workspace")
    entry = json.loads(_research_entry(tmp_path, "obs-conflict").read_text())
    attempts = []

    monkeypatch.setattr(experiment_workspace.exp_io, "validate_managed_output_paths", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(experiment_workspace.exp_io, "path_exists_at", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        experiment_workspace.exp_io,
        "read_text_at",
        lambda *_args, **_kwargs: experiment_workspace.RESEARCH_LOG_PREAMBLE,
    )
    monkeypatch.setattr(
        experiment_workspace.exp_io,
        "conditional_atomic_replace_text_at",
        lambda *_args, **_kwargs: attempts.append("conflict") is None and False,
    )

    with pytest.raises(RuntimeError, match="three append attempts"):
        experiment_workspace.append_research_log(
            root,
            entry,
            experiment_id="unit",
            managed_rows=[],
            remote="unit-host",
        )

    assert attempts == ["conflict", "conflict", "conflict"]


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

    with pytest.raises(ValueError, match="Managed file is missing or aliased"):
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

    with pytest.raises(ValueError, match="Managed file is missing or aliased"):
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
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    prepared = [event for event in events if event["event_type"] == "experiment_finalization_prepared"]
    assert len(prepared) == 1
    assert prepared[0]["report"] == str(tmp_path / "reports" / "final.md")
    assert "experiment_finalized" not in {event["event_type"] for event in events}


def test_experiment_finalize_records_preparation_before_manifest_conflict(tmp_path: Path, monkeypatch):
    spec = _experiment_spec(tmp_path.parent)
    experiments.init_experiment(tmp_path, spec)
    (tmp_path / "run_manifest.tsv").write_text(
        "experiment_id\tstep_id\trun_id\tstatus\nunit\ttrain\trun-000\tcompleted\n"
    )
    report = tmp_path.parent / "final.md"
    report.write_text("# Final\n\nValidation-selected result.\n")
    manifest = tmp_path / "experiment.yaml"
    before = manifest.read_bytes()
    real_commit = experiment_io.conditional_atomic_replace_text_at

    def conflict_manifest(path, text, expected_sha256, *, remote=None, **kwargs):
        if Path(path) == manifest:
            events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
            assert events[-1]["event_type"] == "experiment_finalization_prepared"
            return False
        return real_commit(path, text, expected_sha256, remote=remote, **kwargs)

    monkeypatch.setattr(experiment_io, "conditional_atomic_replace_text_at", conflict_manifest)

    with pytest.raises(RuntimeError, match="manifest changed during finalization"):
        experiments.finalize_experiment(tmp_path, report)

    assert manifest.read_bytes() == before
    assert (tmp_path / "reports" / "final.md").read_text() == report.read_text()
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert [event["event_type"] for event in events].count("experiment_finalization_prepared") == 1
    assert "experiment_finalized" not in {event["event_type"] for event in events}


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

    def interrupt_manifest_replace(source, target, **kwargs):
        if target == manifest.name:
            raise OSError("interrupted")
        return real_replace(source, target, **kwargs)

    monkeypatch.setattr(experiment_io.os, "replace", interrupt_manifest_replace)

    with pytest.raises(OSError, match="interrupted"):
        experiments.finalize_experiment(tmp_path, report)

    assert manifest.read_bytes() == before
    assert (tmp_path / "reports" / "final.md").read_text() == report.read_text()
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert [event["event_type"] for event in events].count("experiment_finalization_prepared") == 1
    assert "experiment_finalized" not in {event["event_type"] for event in events}


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


def test_experiment_finalize_requires_stop_reason_before_writing(tmp_path: Path):
    experiments.init_experiment(tmp_path, _experiment_spec(tmp_path.parent))
    run_manifest = tmp_path / "run_manifest.tsv"
    run_manifest.write_text("experiment_id\tstep_id\trun_id\tstatus\tstop_reason\n" "unit\ttrain\trun-000\tstopped\t\n")
    report = tmp_path.parent / "stopped_final.md"
    report.write_text("# Final\n")
    before = _workspace_files(tmp_path)

    with pytest.raises(ValueError, match="missing required stop_reason"):
        experiments.finalize_experiment(tmp_path, report)

    assert _workspace_files(tmp_path) == before
    rows = _read_table(run_manifest)
    rows[0]["stop_reason"] = "manual stop after invalid labels"
    experiment_io.write_rows_at(run_manifest, rows)
    target = experiments.finalize_experiment(tmp_path, report)
    assert target.read_text() == report.read_text()
    assert "status: completed" in (tmp_path / "experiment.yaml").read_text()


def test_experiment_remote_finalize_checks_stop_reason_before_report_read_or_writes(monkeypatch):
    root = Path("/remote/experiment")
    calls = []

    def managed_rows(candidate, *, remote):
        calls.append((candidate, remote))
        return [
            {
                "experiment_id": "unit",
                "step_id": "train",
                "run_id": "run-000",
                "status": "stopped",
                "stop_reason": "",
            }
        ]

    def unexpected(*_args, **_kwargs):
        raise AssertionError("remote finalize read the report or attempted a write")

    monkeypatch.setattr(experiments, "_managed_rows", managed_rows)
    monkeypatch.setattr(experiment_io, "read_text_at", unexpected)
    monkeypatch.setattr(experiment_io, "conditional_atomic_replace_text_at", unexpected)
    monkeypatch.setattr(experiment_io, "append_event_at", unexpected)

    with pytest.raises(ValueError, match="missing required stop_reason"):
        experiments.finalize_experiment(root, "/remote/final.md", remote="baichuan3")

    assert calls == [(root, "baichuan3")]


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

    with pytest.raises(ValueError, match="Managed file is missing or aliased"):
        experiments.monitor_experiment(tmp_path)

    assert observation_calls == []
    assert _workspace_files(tmp_path) == before


def test_experiment_remote_mutation_preflights_manifest_before_reading_workspace_identity(monkeypatch):
    root = Path("/wujidata/remote_run")
    reads = []

    def _reject_alias(root_arg, paths, *, remote=None, exact_directory_entries=False):
        assert root_arg == root
        assert paths == [root / "experiment.yaml"]
        assert remote == "baichuan3"
        assert exact_directory_entries is False
        raise ValueError("Managed file is missing or aliased")

    monkeypatch.setattr(experiments.exp_io, "path_exists_at", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(experiments.exp_io, "read_managed_files_at", _reject_alias)
    monkeypatch.setattr(
        experiments.exp_io,
        "read_text_at",
        lambda *args, **kwargs: reads.append((args, kwargs)) or "",
    )

    with pytest.raises(ValueError, match="Managed file is missing or aliased"):
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
        if "mkdir -p" in command[-1] or "seen_inodes" in command[-1] or "append_mode" in command[-1]:
            return subprocess.CompletedProcess(command, 0, "", "")
        return subprocess.CompletedProcess(command, experiment_io.REMOTE_MISSING_RETURN_CODE, "", "")

    monkeypatch.setattr("agent_tools.experiment_io.subprocess.run", fake_run)

    experiments.init_experiment("/wujidata/remote_run", _experiment_spec(tmp_path), remote="baichuan3")

    assert all(command[:2] == ["ssh", "baichuan3"] for command, _kwargs in calls)
    assert any("mkdir -p" in command[-1] for command, _kwargs in calls)
    write_targets = [command[-1] for command, _kwargs in calls if "cat >" in command[-1]]
    assert any("/wujidata/remote_run/experiment.yaml" in target for target in write_targets)
    assert any("/wujidata/remote_run/RESEARCH_LOG.md" in target for target in write_targets)
    assert any("/wujidata/remote_run/README.md" in target for target in write_targets)
    assert any("/wujidata/remote_run/experiment_manifest.tsv" in target for target in write_targets)
    assert any(
        "append_mode" in command[-1] and "/wujidata/remote_run/events.jsonl" in command[-1]
        for command, _kwargs in calls
    )
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
