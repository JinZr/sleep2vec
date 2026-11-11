import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from agent_tools import experiment_io, experiment_workspace, manifests


def test_event_append_publishes_one_json_row_per_event(tmp_path: Path):
    experiment_workspace.append_event(tmp_path, "experiment_initialized", {"experiment_id": "unit"})
    experiment_workspace.append_event(tmp_path, "step_registered", {"step_id": "train"})

    events = experiment_workspace.read_experiment_events(tmp_path)
    assert [event.pop("time") is not None for event in events] == [True, True]
    assert events == [
        {"event_type": "experiment_initialized", "experiment_id": "unit"},
        {"event_type": "step_registered", "step_id": "train"},
    ]
    assert not list(tmp_path.glob(".events.jsonl.*.tmp"))


def test_atomic_replace_never_exposes_a_partial_file(tmp_path: Path, monkeypatch):
    target = tmp_path / "run_manifest.tsv"
    target.write_text("step_id\trun_id\n")
    real_fsync = os.fsync

    def crash_before_publish(descriptor):
        real_fsync(descriptor)
        raise OSError("simulated crash before the rename")

    monkeypatch.setattr(os, "fsync", crash_before_publish)

    with pytest.raises(OSError, match="simulated crash"):
        experiment_io.atomic_replace(target, "step_id\trun_id\nplan\trun-000\n")

    assert target.read_text() == "step_id\trun_id\n"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["run_manifest.tsv"]


def test_locked_replace_publishes_complete_text_and_rejects_a_stale_digest(tmp_path: Path):
    target = tmp_path / "steps" / "train" / "step.yaml"

    experiment_io.locked_replace(target, "first\n")
    first_sha256 = hashlib.sha256(b"first\n").hexdigest()
    experiment_io.locked_replace(target, "second\n", expected_sha256=first_sha256)

    assert target.read_text() == "second\n"
    with pytest.raises(RuntimeError, match="changed before its replacement"):
        experiment_io.locked_replace(target, "third\n", expected_sha256=first_sha256)
    assert target.read_text() == "second\n"
    with pytest.raises(RuntimeError, match="changed before its replacement"):
        experiment_io.locked_replace(tmp_path / "missing.yaml", "text\n", expected_sha256=first_sha256)
    assert not (tmp_path / "missing.yaml").exists()


def test_locked_append_extends_the_file_under_its_sidecar_lock(tmp_path: Path):
    target = tmp_path / "events.jsonl"

    experiment_io.locked_append(target, "a\n")
    experiment_io.locked_append(target, "b\n")

    assert target.read_text() == "a\nb\n"
    assert experiment_io.lock_path_for(target) == tmp_path / ".events.jsonl.lock"


def test_managed_output_preflight_rejects_outside_paths_and_a_symlinked_root(tmp_path: Path):
    root = tmp_path / "workspace"
    root.mkdir()
    experiment_io.validate_managed_output_paths(root, [root / "reports" / "final.md"])

    with pytest.raises(ValueError, match="outside its workspace"):
        experiment_io.validate_managed_output_paths(root, [tmp_path / "final.md"])
    alias = tmp_path / "alias"
    alias.symlink_to(root, target_is_directory=True)
    with pytest.raises(ValueError, match="must not be a symlink"):
        experiment_io.validate_managed_output_paths(alias, [alias / "final.md"])


def test_managed_control_reads_reject_dot_dot_before_resolution(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()

    with pytest.raises(ValueError, match="must not contain '..'"):
        experiment_io.read_managed_files_at(root, [root / "missing" / ".." / "file"])


def test_local_managed_control_bundle_requires_exact_directory_entries(tmp_path):
    root = tmp_path / "workspace"
    bundle = root / "bundle"
    bundle.mkdir(parents=True)
    paths = [bundle / "questions.json", bundle / "questions.md", bundle / "plan.blocked.md"]
    for path in paths:
        path.write_text(f"{path.name}\n")

    experiment_io.read_managed_files_at(root, paths, exact_directory_entries=True)
    (bundle / "run.sh").write_text("partial launch\n")

    with pytest.raises(ValueError, match="directory entries differ"):
        experiment_io.read_managed_files_at(root, paths, exact_directory_entries=True)


def _run_managed_output_reader(root, paths, *, prefix=""):
    source = """import json, sys
from agent_tools import experiment_io
root, paths = json.loads(sys.argv[1])
print(json.dumps(experiment_io.read_managed_output_texts_at(root, paths)))
"""
    return subprocess.run(
        [sys.executable, "-c", prefix + source, json.dumps([str(root), [str(path) for path in paths]])],
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        capture_output=True,
        timeout=5,
    )


def test_managed_output_reader_preserves_requested_keys_empty_text_and_missing(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "terminal.json").write_bytes("状态\r\nready\r\n".encode())
    empty = root / "empty.json"
    empty.write_text("")
    paths = [str(root) + "/./terminal.json", str(empty), str(root / "missing.json"), str(root / "absent" / "file")]

    result = _run_managed_output_reader(root, paths)

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == dict(zip(paths, ["状态\r\nready\r\n", "", None, None]))
    assert not (root / "absent").exists()


@pytest.mark.parametrize("missing", ["root", "parent", "leaf"])
def test_managed_output_reader_only_missing_opens_return_none(tmp_path, missing):
    root = tmp_path / "workspace"
    parent = root / "outputs"
    if missing != "root":
        root.mkdir()
    if missing == "leaf":
        parent.mkdir()
    paths = [parent / "terminal.json", parent / "allocation.json"]

    result = _run_managed_output_reader(root, paths)

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {str(path): None for path in paths}
    assert not paths[0].exists()


@pytest.mark.parametrize("location", ["root", "target"])
@pytest.mark.parametrize("component", ["absent", "alias"])
def test_managed_output_reader_rejects_raw_parent_components(tmp_path, location, component):
    root = tmp_path / "workspace"
    root.mkdir()
    if component == "alias":
        (root / component).symlink_to(tmp_path, target_is_directory=True)
    if location == "root":
        root = root / component / ".."
        target = root / "terminal.json"
    else:
        target = root / component / ".." / "terminal.json"

    result = _run_managed_output_reader(root, [target])

    assert result.returncode != 0
    assert "must not contain '..'" in result.stderr
    assert result.stdout == ""


@pytest.mark.parametrize("alias", [False, True])
def test_managed_output_reader_rejects_duplicate_requested_targets(tmp_path, alias):
    target = str(tmp_path / "terminal.json")
    duplicate = str(tmp_path) + "/./terminal.json" if alias else target

    result = _run_managed_output_reader(tmp_path, [target, duplicate])

    assert result.returncode != 0
    assert "must be unique" in result.stderr
    assert result.stdout == ""


def test_managed_output_reader_rejects_invalid_utf8_without_partial_output(tmp_path):
    good = tmp_path / "allocation.json"
    bad = tmp_path / "terminal.json"
    good.write_text("{}")
    bad.write_bytes(b"\xff")

    result = _run_managed_output_reader(tmp_path, [good, bad])

    assert result.returncode != 0
    assert result.stdout == ""
    assert "utf-8" in result.stderr


def test_managed_output_reader_does_not_hide_permission_errors(tmp_path):
    if os.geteuid() == 0:
        pytest.skip("Root bypasses the file permission used by this test")
    target = tmp_path / "terminal.json"
    target.write_text("{}")
    target.chmod(0)
    try:
        result = _run_managed_output_reader(tmp_path, [target])
    finally:
        target.chmod(0o600)

    assert result.returncode != 0
    assert result.stdout == ""
    assert "PermissionError" in result.stderr


@pytest.mark.parametrize("header", ["trial_id\n", "run_id\n", "step_id\tstep_id\trun_id\n"])
def test_managed_table_reader_rejects_removed_or_malformed_header_only_tables(tmp_path: Path, header: str):
    path = tmp_path / "run_status.tsv"
    path.write_text(header)

    with pytest.raises(ValueError):
        experiment_io.read_rows_at(path, require_managed_identity=True)


def test_managed_table_reader_accepts_current_header_only_table(tmp_path: Path):
    path = tmp_path / "run_status.tsv"
    path.write_text("step_id\trun_id\n")

    assert experiment_io.read_rows_at(path, require_managed_identity=True) == []


@pytest.mark.parametrize(
    "reader",
    [
        pytest.param(lambda path: manifests.read_rows(path, require_managed_identity=True), id="read_rows"),
        pytest.param(lambda path: experiment_io.read_rows_at(path, require_managed_identity=True), id="read_rows_at"),
    ],
)
@pytest.mark.parametrize(
    ("header", "message"),
    [
        pytest.param(
            "trial_id\tstep_id\trun_id\n",
            "Historical managed table fields are read-only; Historical trial_id fields are unsupported: {path}",
            id="trial_id",
        ),
        pytest.param(
            "step_id\trun_id\tparam.lr\n",
            "Historical parameter fields are read-only: {path}",
            id="param_prefix",
        ),
        pytest.param(
            "experiment_id\n",
            "Managed table header must define step_id and run_id; missing step_id, run_id: {path}",
            id="missing_identity",
        ),
    ],
)
def test_managed_header_contract_messages_are_identical_across_readers(
    tmp_path: Path, reader, header: str, message: str
):
    path = tmp_path / "run_status.tsv"
    path.write_text(header)

    with pytest.raises(ValueError) as excinfo:
        reader(path)

    assert str(excinfo.value) == message.format(path=path)


@pytest.mark.parametrize("document", ["[]", '"plan"', "1", "null"])
def test_json_reader_rejects_a_document_that_is_not_an_object(tmp_path: Path, document: str):
    path = tmp_path / "plan.json"
    path.write_text(document + "\n")

    with pytest.raises(ValueError) as excinfo:
        manifests.read_json(path)

    assert str(excinfo.value) == f"JSON must be a mapping: {path}"


@pytest.mark.parametrize(
    "contents",
    [
        "experiment_id\texperiment_id\texperiment_root\nunit\tunit\t/root\n",
        "experiment_id\texperiment_root\nunit\t/root\textra\n",
    ],
)
def test_strict_table_reader_rejects_duplicate_header_and_non_rectangular_rows(tmp_path: Path, contents: str):
    path = tmp_path / "experiment_manifest.tsv"
    path.write_text(contents)

    with pytest.raises(ValueError):
        experiment_io.read_rows_at(path, strict=True)


def test_strict_table_reader_does_not_require_managed_run_identity(tmp_path: Path):
    path = tmp_path / "experiment_manifest.tsv"
    path.write_text("experiment_id\texperiment_root\nunit\t/root\n")

    assert experiment_io.read_rows_at(path, strict=True) == [{"experiment_id": "unit", "experiment_root": "/root"}]
