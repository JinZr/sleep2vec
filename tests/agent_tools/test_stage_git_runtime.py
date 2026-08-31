from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
from threading import Barrier, Event
import time
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "utils" / "stage_git_runtime.py"
REAL_GIT = shutil.which("git")


def _git(root, *args):
    return subprocess.check_output([REAL_GIT, "-C", str(root), *args], text=True).strip()


@pytest.fixture
def staging():
    spec = importlib.util.spec_from_file_location("stage_git_runtime", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def source_repo(tmp_path_factory):
    root = tmp_path_factory.mktemp("stage-source") / "source repo"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.name", "Runtime Fixture")
    _git(root, "config", "user.email", "runtime@example.invalid")
    _git(root, "config", "commit.gpgsign", "false")
    _git(root, "remote", "add", "origin", "https://example.invalid/source.git")
    (root / "tracked.txt").write_text("requested commit\n")
    _git(root, "add", "tracked.txt")
    _git(root, "commit", "-qm", "Add requested content")
    commit = _git(root, "rev-parse", "HEAD")
    (root / "tracked.txt").write_text("newer commit\n")
    _git(root, "commit", "-qam", "Change later content")
    (root / "tracked.txt").write_text("uncommitted edit\n")
    (root / "untracked.txt").write_text("private local work\n")
    return SimpleNamespace(root=root, commit=commit, head=_git(root, "rev-parse", "HEAD"))


@pytest.fixture
def fake_transport(tmp_path, monkeypatch):
    directory = tmp_path / "fake transport"
    directory.mkdir()
    calls = tmp_path / "transport-calls.jsonl"
    program = f"#!{sys.executable}\n" + """import json
import os
from pathlib import Path
import shutil
import shlex
import subprocess
import sys

kind = Path(sys.argv[0]).name
with open(os.environ["STAGE_TEST_TRANSPORT_CALLS"], "a") as stream:
    stream.write(json.dumps({"kind": kind, "argv": sys.argv[1:]}) + "\\n")
if kind == "scp":
    if os.environ.get("STAGE_TEST_SCP_FAILURE"):
        sys.exit(23)
    args = [value for value in sys.argv[1:] if not value.startswith("-")]
    target = Path(shlex.split(args[-1].split(":", 1)[1])[0])
    for source in args[:-1]:
        copied = target / Path(source).name if target.is_dir() else target
        shutil.copyfile(source, copied)
        if os.environ.get("STAGE_TEST_CORRUPT_BUNDLE") and copied.name == "runtime.bundle":
            copied.write_bytes(b"damaged transfer")
    sys.exit(0)
mode = os.environ.get("STAGE_TEST_SSH_MODE", "normal")
failure_call = os.environ.get("STAGE_TEST_SSH_FAILURE_CALL")
if failure_call:
    calls = [json.loads(line) for line in Path(os.environ["STAGE_TEST_TRANSPORT_CALLS"]).read_text().splitlines()]
    if sum(call["kind"] == "ssh" for call in calls) != int(failure_call):
        mode = "normal"
if mode == "eof":
    print("authentication EOF", file=sys.stderr)
    sys.exit(0)
result = subprocess.run(["bash", "-c", sys.argv[-1]], input=sys.stdin.buffer.read(), capture_output=True)
output = result.stdout
if mode == "empty":
    output = b""
elif mode == "truncated":
    output = output[:len(output) // 2]
elif mode in {"wrong_binding", "wrong_head", "missing_steps"} and output:
    data = json.loads(output)
    if mode == "wrong_binding":
        data["attempt_id"] = "another-attempt"
    elif mode == "wrong_head":
        data["head"] = "a" * 40
    else:
        data.pop("steps", None)
    output = json.dumps(data).encode()
sys.stdout.buffer.write(output)
sys.stderr.buffer.write(result.stderr)
sys.exit(255 if mode == "ssh255" else result.returncode)
"""
    for name in ("ssh", "scp"):
        executable = directory / name
        executable.write_text(program)
        executable.chmod(0o755)
    monkeypatch.setenv("PATH", str(directory) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("STAGE_TEST_TRANSPORT_CALLS", str(calls))
    return calls


def _transport_calls(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


@pytest.fixture
def observed_git(tmp_path, monkeypatch):
    directory = tmp_path / "observed git"
    directory.mkdir()
    calls = tmp_path / "git-calls.jsonl"
    program = f"#!{sys.executable}\n" + """import json
import os
from pathlib import Path
import subprocess
import sys
import time

argv = sys.argv[1:]
record = {"argv": argv, "cwd": os.getcwd(), "pid": os.getpid(), "sid": os.getsid(0)}
record["git_environment"] = {key: os.environ.get(key) for key in (
    "GIT_CONFIG_GLOBAL", "GIT_CONFIG_NOSYSTEM", "GIT_ATTR_NOSYSTEM")}
record["stdio"] = [{"mode": os.fstat(fd).st_mode, "device": os.fstat(fd).st_rdev,
                    "inode": os.fstat(fd).st_ino} for fd in (0, 1, 2)]
with open(os.environ["STAGE_TEST_GIT_CALLS"], "a") as stream:
    stream.write(json.dumps(record) + "\\n")
failure = os.environ.get("STAGE_TEST_GIT_FAILURE")
destination = os.environ["STAGE_TEST_DESTINATION"]
producer = os.environ.get("STAGE_TEST_GIT_SCOPE") == "producer"
affected = "producer.git" in (" ".join(argv) + os.getcwd()) if producer else (
    destination in " ".join(argv) or os.getcwd() == destination)
if failure and all(word in argv for word in failure.split()) and affected:
    print("injected git phase failure: " + failure, file=sys.stderr)
    sys.exit(int(os.environ.get("STAGE_TEST_GIT_RETURN_CODE", "19")))
if argv[0] == "checkout" and os.environ.get("STAGE_TEST_ATTACH_HEAD"):
    subprocess.run([os.environ["STAGE_TEST_REAL_GIT"], *argv], check=True)
    os.execv(os.environ["STAGE_TEST_REAL_GIT"], [os.environ["STAGE_TEST_REAL_GIT"],
        "checkout", "-b", "unexpected-attached-head"])
if argv[0] == "clone" and os.environ.get("STAGE_TEST_GIT_GATE"):
    gate = Path(os.environ["STAGE_TEST_GIT_GATE"])
    gate.with_suffix(".waiting").write_text(str(os.getpid()))
    deadline = time.monotonic() + 10
    while not gate.exists():
        if time.monotonic() >= deadline:
            sys.exit(24)
        time.sleep(0.01)
print("durable git stderr", file=sys.stderr)
os.execv(os.environ["STAGE_TEST_REAL_GIT"], [os.environ["STAGE_TEST_REAL_GIT"], *argv])
"""
    executable = directory / "git"
    executable.write_text(program)
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", str(directory) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("STAGE_TEST_GIT_CALLS", str(calls))
    monkeypatch.setenv("STAGE_TEST_REAL_GIT", REAL_GIT)
    monkeypatch.setenv("STAGE_TEST_DESTINATION", str(tmp_path / "runtime destination"))
    return calls


def _finish(staging, evidence):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        result = staging.check_runtime(evidence)
        if result["kind"] != "pending":
            return result
        time.sleep(0.02)
    pytest.fail("The tiny Git staging operation did not finish within ten seconds")


def _stage(staging, source, tmp_path, *, remote=False):
    evidence = tmp_path / "evidence"
    destination = tmp_path / "runtime destination"
    kwargs = {}
    if remote:
        kwargs = {
            "host": "fake-host",
            "remote_python": sys.executable,
            "remote_attempt_dir": str(tmp_path / "remote attempt"),
        }
    result = staging.stage_runtime(source.root, source.commit, destination, evidence, **kwargs)
    return evidence, destination, result


@pytest.mark.parametrize("remote", [False, True])
def test_stage_exact_commit_excludes_dirty_source(staging, source_repo, tmp_path, fake_transport, remote):
    evidence, destination, started = _stage(staging, source_repo, tmp_path, remote=remote)
    assert started["kind"] == "started"
    result = _finish(staging, evidence)

    assert result["kind"] == "completed"
    assert result["returncode"] == 0
    assert result["head"] == source_repo.commit
    assert result["clean"] is True
    assert _git(destination, "rev-parse", "HEAD") == source_repo.commit
    assert _git(destination, "status", "--porcelain") == ""
    assert (destination / "tracked.txt").read_text() == "requested commit\n"
    assert not (destination / "untracked.txt").exists()
    assert _git(source_repo.root, "rev-parse", "HEAD") == source_repo.head
    assert (source_repo.root / "tracked.txt").read_text() == "uncommitted edit\n"
    assert (source_repo.root / "untracked.txt").read_text() == "private local work\n"
    assert [(step["argv"][1], step["returncode"]) for step in result["steps"]] == [
        ("clone", 0),
        ("checkout", 0),
        ("symbolic-ref", 1),
        ("rev-parse", 0),
        ("status", 0),
    ]


@pytest.mark.parametrize("remote", [False, True])
@pytest.mark.parametrize("ambient", ["autocrlf", "attributes"])
def test_ambient_git_configuration_does_not_transform_runtime_bytes(
    staging, source_repo, tmp_path, fake_transport, observed_git, monkeypatch, ambient, remote
):
    ambient_home = tmp_path / "ambient home"
    ambient_home.mkdir()
    if ambient == "autocrlf":
        (ambient_home / ".gitconfig").write_text("[core]\n\tautocrlf = true\n")
    else:
        attributes = ambient_home / ".config" / "git" / "attributes"
        attributes.parent.mkdir(parents=True)
        attributes.write_text("* text eol=crlf\n")
    monkeypatch.setenv("HOME", str(ambient_home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(ambient_home / ".config"))

    evidence, destination, _started = _stage(staging, source_repo, tmp_path, remote=remote)

    assert _finish(staging, evidence)["kind"] == "completed"
    assert (destination / "tracked.txt").read_bytes() == b"requested commit\n"
    assert (source_repo.root / "tracked.txt").read_bytes() == b"uncommitted edit\n"
    records = _transport_calls(observed_git)
    assert records
    assert all(
        record["git_environment"]
        == {"GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1", "GIT_ATTR_NOSYSTEM": "1"}
        for record in records
    )


def test_producer_failure_does_not_transfer(staging, source_repo, tmp_path, fake_transport, monkeypatch):
    def fail_producer(*_args):
        raise subprocess.CalledProcessError(3, ["git", "bundle", "create"])

    monkeypatch.setattr(staging, "_produce_bundle", fail_producer)
    with pytest.raises(subprocess.CalledProcessError):
        _stage(staging, source_repo, tmp_path, remote=True)
    assert _transport_calls(fake_transport) == []
    assert not (tmp_path / "runtime destination" / ".git").exists()


@pytest.mark.parametrize("operation", ["init", "fetch", "rev-parse", "bundle create", "bundle verify"])
def test_real_producer_git_failure_stops_before_transfer(
    staging, source_repo, tmp_path, fake_transport, observed_git, monkeypatch, operation
):
    monkeypatch.setenv("STAGE_TEST_GIT_SCOPE", "producer")
    monkeypatch.setenv("STAGE_TEST_GIT_FAILURE", operation)
    with pytest.raises(subprocess.CalledProcessError):
        _stage(staging, source_repo, tmp_path, remote=True)
    assert _transport_calls(fake_transport) == []
    results = json.loads((tmp_path / "evidence" / "producer-results.json").read_text())
    assert results[-1]["returncode"] == 19
    assert all(word in results[-1]["argv"] for word in operation.split())
    assert not (tmp_path / "runtime destination").exists()
    assert not (tmp_path / "remote attempt").exists()


def test_valid_hash_does_not_make_invalid_bundle_usable(staging, source_repo, tmp_path, monkeypatch):
    produce = staging._produce_bundle

    def invalid_bundle(*args):
        path = produce(*args)
        path.write_bytes(b"not a Git bundle")
        return path

    monkeypatch.setattr(staging, "_produce_bundle", invalid_bundle)
    evidence, destination, _started = _stage(staging, source_repo, tmp_path)
    result = _finish(staging, evidence)
    assert result["kind"] == "failed"
    assert result["returncode"] != 0
    assert len(result["steps"]) == 1
    assert result["steps"][0]["argv"][1] == "clone"
    assert not (destination / ".git").exists()


@pytest.mark.parametrize("mode", ["eof", "empty", "truncated", "wrong_binding", "ssh255"])
def test_ssh_uncertainty_never_becomes_success_or_retries(
    staging, source_repo, tmp_path, fake_transport, monkeypatch, mode
):
    monkeypatch.setenv("STAGE_TEST_SSH_MODE", mode)
    with pytest.raises((RuntimeError, ValueError, subprocess.CalledProcessError)):
        _stage(staging, source_repo, tmp_path, remote=True)
    calls = _transport_calls(fake_transport)
    assert len([call for call in calls if call["kind"] == "ssh"]) == 1
    assert not any(call["kind"] == "scp" for call in calls)
    assert not (tmp_path / "runtime destination" / ".git").exists()


def test_transfer_failure_does_not_start_worker(staging, source_repo, tmp_path, fake_transport, monkeypatch):
    monkeypatch.setenv("STAGE_TEST_SCP_FAILURE", "1")
    with pytest.raises((RuntimeError, subprocess.CalledProcessError)):
        _stage(staging, source_repo, tmp_path, remote=True)
    calls = _transport_calls(fake_transport)
    assert len([call for call in calls if call["kind"] == "scp"]) == 1
    assert len([call for call in calls if call["kind"] == "ssh"]) == 1
    assert not (tmp_path / "runtime destination" / ".git").exists()


def test_corrupted_transfer_never_prepares_runtime(staging, source_repo, tmp_path, fake_transport, monkeypatch):
    monkeypatch.setenv("STAGE_TEST_CORRUPT_BUNDLE", "1")
    with pytest.raises(RuntimeError, match="Transport failed"):
        _stage(staging, source_repo, tmp_path, remote=True)
    assert not (tmp_path / "runtime destination" / ".git").exists()
    assert not (tmp_path / "remote attempt" / "launch-attempt.json").exists()
    calls = _transport_calls(fake_transport)
    assert len([call for call in calls if call["kind"] == "scp"]) == 1
    assert len([call for call in calls if call["kind"] == "ssh"]) == 2


@pytest.mark.parametrize("operation", ["clone", "checkout", "rev-parse", "status"])
def test_failed_git_phase_never_has_positive_terminal_receipt(
    staging, source_repo, tmp_path, fake_transport, observed_git, monkeypatch, operation
):
    monkeypatch.setenv("STAGE_TEST_GIT_FAILURE", operation)
    evidence, _destination, _started = _stage(staging, source_repo, tmp_path)
    result = _finish(staging, evidence)
    assert result["kind"] == "failed"
    assert result["returncode"] != 0
    assert any(step["returncode"] != 0 for step in result["steps"])
    records = _transport_calls(observed_git)
    failed = [index for index, record in enumerate(records) if operation in record["argv"]]
    assert failed
    assert operation in records[-1]["argv"]


@pytest.mark.parametrize("mode", ["empty", "truncated", "wrong_binding", "ssh255"])
def test_lost_start_reply_does_not_repeat_successful_launch(
    staging, source_repo, tmp_path, fake_transport, monkeypatch, mode
):
    monkeypatch.setenv("STAGE_TEST_SSH_MODE", mode)
    monkeypatch.setenv("STAGE_TEST_SSH_FAILURE_CALL", "2")
    with pytest.raises((RuntimeError, ValueError, subprocess.CalledProcessError)):
        _stage(staging, source_repo, tmp_path, remote=True)
    calls = _transport_calls(fake_transport)
    assert len([call for call in calls if call["kind"] == "ssh"]) == 2
    assert len([call for call in calls if call["kind"] == "scp"]) == 1

    monkeypatch.delenv("STAGE_TEST_SSH_MODE")
    completed = _finish(staging, tmp_path / "evidence")
    assert completed["kind"] == "completed"
    assert completed["head"] == source_repo.commit
    assert (tmp_path / "remote attempt" / "launch-attempt.json").is_file()


def test_lost_start_reply_preserves_pending_worker_handle(
    staging, source_repo, tmp_path, fake_transport, observed_git, monkeypatch
):
    gate = tmp_path / "release-git"
    attempt = tmp_path / "remote attempt"
    evidence = tmp_path / "evidence"
    monkeypatch.setenv("STAGE_TEST_GIT_GATE", str(gate))
    monkeypatch.setenv("STAGE_TEST_SSH_MODE", "empty")
    monkeypatch.setenv("STAGE_TEST_SSH_FAILURE_CALL", "2")
    try:
        with pytest.raises(RuntimeError, match="no complete operation reply"):
            _stage(staging, source_repo, tmp_path, remote=True)
        deadline = time.monotonic() + 5
        while not gate.with_suffix(".waiting").exists():
            assert time.monotonic() < deadline, "Worker did not reach the controlled Git barrier"
            time.sleep(0.01)
        worker = json.loads((attempt / "worker.json").read_text())
        monkeypatch.delenv("STAGE_TEST_SSH_MODE")
        pending = staging.check_runtime(evidence)
        assert pending["kind"] == "pending"
        assert pending["recorded_pid"] == worker["pid"]
        assert os.getpgid(worker["pid"]) == worker["pid"]
        assert not (attempt / "receipt.json").exists()
        calls = _transport_calls(fake_transport)
        assert len([call for call in calls if call["kind"] == "ssh" and "_launch" in call["argv"][-1]]) == 1
        assert len([call for call in calls if call["kind"] == "scp"]) == 1
    finally:
        gate.touch()
    assert _finish(staging, evidence)["kind"] == "completed"


@pytest.mark.parametrize(("field", "value"), [("attempt_id", "wrong-attempt"), ("pid", 0), ("pid", True)])
def test_pending_worker_handle_rejects_wrong_identity_or_pid(staging, source_repo, tmp_path, field, value):
    evidence, _destination, _started = _stage(staging, source_repo, tmp_path)
    assert _finish(staging, evidence)["kind"] == "completed"
    attempt = evidence / "target"
    (attempt / "receipt.json").unlink()
    worker_path = attempt / "worker.json"
    worker = json.loads(worker_path.read_text())
    worker[field] = value
    worker_path.write_text(json.dumps(worker))
    with pytest.raises(RuntimeError, match="Transport failed"):
        staging.check_runtime(evidence)


def test_pending_worker_handle_is_atomically_visible(staging, source_repo, tmp_path, fake_transport, monkeypatch):
    monkeypatch.setenv("STAGE_TEST_SSH_MODE", "eof")
    monkeypatch.setenv("STAGE_TEST_SSH_FAILURE_CALL", "2")
    with pytest.raises(RuntimeError, match="no complete operation reply"):
        _stage(staging, source_repo, tmp_path, remote=True)
    monkeypatch.delenv("STAGE_TEST_SSH_MODE")
    attempt = tmp_path / "remote attempt"
    evidence = tmp_path / "evidence"
    partial = Event()
    release = Event()
    dump = staging.json.dump

    def pause_handle_write(value, stream, **kwargs):
        if isinstance(value, dict) and "pid" in value:
            serialized = json.dumps(value, **kwargs)
            stream.write(serialized[:1])
            stream.flush()
            partial.set()
            assert release.wait(timeout=5), "Worker handle publication was not released"
            stream.write(serialized[1:])
        else:
            dump(value, stream, **kwargs)

    monkeypatch.setattr(staging.json, "dump", pause_handle_write)
    with ThreadPoolExecutor(max_workers=1) as executor:
        worker = executor.submit(staging._worker, attempt)
        try:
            assert partial.wait(timeout=5), "Worker did not begin writing its handle"
            assert not (attempt / "worker.json").exists()
            pending = staging.check_runtime(evidence)
            assert pending["kind"] == "pending"
            assert "recorded_pid" not in pending
        finally:
            release.set()
        assert worker.result(timeout=10)["kind"] == "completed"
    handle = json.loads((attempt / "worker.json").read_text())
    assert handle["pid"] == os.getpid()
    assert _finish(staging, evidence)["kind"] == "completed"


@pytest.mark.parametrize("remote", [False, True])
def test_changed_worker_cannot_forward_receipt_for_dirty_runtime(
    staging, source_repo, tmp_path, fake_transport, remote
):
    evidence, destination, _started = _stage(staging, source_repo, tmp_path, remote=remote)
    assert _finish(staging, evidence)["kind"] == "completed"
    attempt = tmp_path / "remote attempt" if remote else evidence / "target"
    marker = tmp_path / "changed-worker-executed"
    (attempt / "stage_git_runtime.py").write_text(
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).touch()\n"
        f"print(Path({str(attempt / 'receipt.json')!r}).read_text())\n"
    )
    (destination / "tracked.txt").write_text("dirty runtime\n")
    before = _transport_calls(fake_transport)

    with pytest.raises(RuntimeError, match="Transport failed"):
        staging.check_runtime(evidence)

    assert not marker.exists()
    assert (destination / "tracked.txt").read_text() == "dirty runtime\n"
    extra = _transport_calls(fake_transport)[len(before) :]
    assert len(extra) == int(remote)
    assert all(call["kind"] == "ssh" and "_check" in call["argv"][-1] for call in extra)


def test_staging_evidence_and_parent_entries_are_fsynced(staging, source_repo, tmp_path, fake_transport, monkeypatch):
    observer = tmp_path / "fsync observer"
    observer.mkdir()
    trace = tmp_path / "fsync-calls.jsonl"
    (observer / "sitecustomize.py").write_text(
        "import json, os\n"
        "original_fsync = os.fsync\n"
        "def observed_fsync(descriptor):\n"
        "    original_fsync(descriptor)\n"
        "    info = os.fstat(descriptor)\n"
        "    with open(os.environ['STAGE_TEST_FSYNC_LOG'], 'a') as stream:\n"
        "        stream.write(json.dumps([info.st_dev, info.st_ino]) + '\\n')\n"
        "os.fsync = observed_fsync\n"
    )
    monkeypatch.setenv("PYTHONPATH", str(observer))
    monkeypatch.setenv("STAGE_TEST_FSYNC_LOG", str(trace))
    original_fsync = staging.os.fsync

    def observed_fsync(descriptor):
        original_fsync(descriptor)
        info = os.fstat(descriptor)
        with trace.open("a") as stream:
            stream.write(json.dumps([info.st_dev, info.st_ino]) + "\n")

    monkeypatch.setattr(staging.os, "fsync", observed_fsync)
    evidence = tmp_path / "evidence parent" / "evidence"
    attempt = tmp_path / "attempt parent" / "attempt"
    destination = tmp_path / "destination parent" / "runtime"
    for path in (evidence, attempt, destination):
        path.parent.mkdir()
    staging.stage_runtime(
        source_repo.root,
        source_repo.commit,
        destination,
        evidence,
        host="fake-host",
        remote_python=sys.executable,
        remote_attempt_dir=attempt,
    )
    before_completion = [json.loads(line) for line in trace.read_text().splitlines()]
    for directory in (evidence, attempt, destination, evidence.parent, attempt.parent, destination.parent):
        info = directory.stat()
        assert [info.st_dev, info.st_ino] in before_completion

    assert _finish(staging, evidence)["kind"] == "completed"
    synced = [json.loads(line) for line in trace.read_text().splitlines()]
    for path in (
        evidence / "intent.json",
        evidence / "request.json",
        evidence / "producer-results.json",
        attempt / "request.json",
        attempt / "launch-attempt.json",
        attempt / "worker.json",
        attempt / "receipt.json",
    ):
        info = path.stat()
        file_sync = synced.index([info.st_dev, info.st_ino])
        parent = path.parent.stat()
        assert [parent.st_dev, parent.st_ino] in synced[file_sync + 1 :], path.name


@pytest.mark.parametrize("remote", [False, True])
def test_worker_has_detached_session_and_persistent_stdio(
    staging, source_repo, tmp_path, fake_transport, observed_git, remote
):
    evidence, destination, started = _stage(staging, source_repo, tmp_path, remote=remote)
    assert _finish(staging, evidence)["kind"] == "completed"
    records = _transport_calls(observed_git)
    target_commands = [
        record for record in records if record["cwd"] == str(destination) or record["argv"][0] == "clone"
    ]
    assert target_commands
    attempt = tmp_path / "remote attempt" if remote else evidence / "target"
    log_inode = (attempt / "worker.log").stat().st_ino
    for record in target_commands:
        assert record["sid"] == started["pid"]
        assert record["sid"] != os.getsid(0)
        assert record["stdio"][0]["device"] == os.stat(os.devnull).st_rdev
        assert record["stdio"][2]["inode"] == log_inode
        if record["argv"][0] in {"clone", "checkout"}:
            assert record["stdio"][1]["inode"] == log_inode
        else:
            assert stat.S_ISFIFO(record["stdio"][1]["mode"])
    assert "durable git stderr" in (attempt / "worker.log").read_text()


@pytest.mark.parametrize("field", ["attempt_id", "destination", "commit", "bundle_sha256", "head", "clean"])
def test_completed_receipt_with_wrong_binding_is_not_success(staging, source_repo, tmp_path, fake_transport, field):
    evidence, _destination, _started = _stage(staging, source_repo, tmp_path)
    assert _finish(staging, evidence)["kind"] == "completed"
    receipt_path = evidence / "target" / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt[field] = False if field == "clean" else "incorrect-binding"
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises((RuntimeError, ValueError)):
        staging.check_runtime(evidence)


@pytest.mark.parametrize(
    "mode", ["eof", "empty", "truncated", "wrong_binding", "wrong_head", "missing_steps", "ssh255"]
)
def test_check_requires_complete_bound_success_without_retry(
    staging, source_repo, tmp_path, fake_transport, monkeypatch, mode
):
    evidence, _destination, _started = _stage(staging, source_repo, tmp_path, remote=True)
    assert _finish(staging, evidence)["kind"] == "completed"
    before = _transport_calls(fake_transport)
    monkeypatch.setenv("STAGE_TEST_SSH_MODE", mode)
    with pytest.raises((RuntimeError, ValueError, subprocess.CalledProcessError)):
        staging.check_runtime(evidence)
    after = _transport_calls(fake_transport)
    assert len(after) == len(before) + 1
    assert after[-1]["kind"] == "ssh"


@pytest.mark.parametrize("drift", ["head", "tracked", "untracked"])
def test_check_rejects_changed_runtime_without_repair(staging, source_repo, tmp_path, drift):
    evidence, destination, _started = _stage(staging, source_repo, tmp_path)
    assert _finish(staging, evidence)["kind"] == "completed"
    if drift == "head":
        _git(destination, "fetch", str(source_repo.root), source_repo.head)
        _git(destination, "checkout", "--detach", source_repo.head)
    else:
        (destination / ("tracked.txt" if drift == "tracked" else "foreign.txt")).write_text("keep this change\n")
    head = _git(destination, "rev-parse", "HEAD")
    status = _git(destination, "status", "--porcelain")
    with pytest.raises((RuntimeError, ValueError)):
        staging.check_runtime(evidence)
    assert _git(destination, "rev-parse", "HEAD") == head
    assert _git(destination, "status", "--porcelain") == status


@pytest.mark.parametrize("remote", [False, True])
def test_check_rejects_attached_head_even_at_exact_clean_commit(staging, source_repo, tmp_path, fake_transport, remote):
    evidence, destination, _started = _stage(staging, source_repo, tmp_path, remote=remote)
    assert _finish(staging, evidence)["kind"] == "completed"
    _git(destination, "checkout", "-b", "reattached-head")
    assert _git(destination, "rev-parse", "HEAD") == source_repo.commit
    assert _git(destination, "status", "--porcelain") == ""

    with pytest.raises(RuntimeError, match="Transport failed"):
        staging.check_runtime(evidence)

    assert _git(destination, "symbolic-ref", "HEAD") == "refs/heads/reattached-head"
    assert _git(destination, "rev-parse", "HEAD") == source_repo.commit


@pytest.mark.parametrize("mode", ["attached", "symbolic_error"])
def test_worker_requires_explicit_detached_head_evidence(
    staging, source_repo, tmp_path, observed_git, monkeypatch, mode
):
    if mode == "attached":
        monkeypatch.setenv("STAGE_TEST_ATTACH_HEAD", "1")
    else:
        monkeypatch.setenv("STAGE_TEST_GIT_FAILURE", "symbolic-ref")
        monkeypatch.setenv("STAGE_TEST_GIT_RETURN_CODE", "128")
    evidence, destination, _started = _stage(staging, source_repo, tmp_path)

    result = _finish(staging, evidence)

    assert result["kind"] == "failed"
    assert result["returncode"] != 0
    assert result["steps"][-1] == {
        "argv": ["git", "symbolic-ref", "--quiet", "HEAD"],
        "returncode": 0 if mode == "attached" else 128,
    }
    assert _git(destination, "rev-parse", "HEAD") == source_repo.commit
    assert _git(destination, "status", "--porcelain") == ""


def test_check_does_not_treat_symbolic_ref_error_as_detached(staging, source_repo, tmp_path, observed_git, monkeypatch):
    evidence, _destination, _started = _stage(staging, source_repo, tmp_path)
    assert _finish(staging, evidence)["kind"] == "completed"
    monkeypatch.setenv("STAGE_TEST_GIT_FAILURE", "symbolic-ref")
    monkeypatch.setenv("STAGE_TEST_GIT_RETURN_CODE", "128")
    with pytest.raises(RuntimeError, match="Transport failed"):
        staging.check_runtime(evidence)


def test_missing_receipt_stays_pending_even_when_head_exists(staging, source_repo, tmp_path, fake_transport):
    evidence, destination, _started = _stage(staging, source_repo, tmp_path, remote=True)
    assert _finish(staging, evidence)["kind"] == "completed"
    attempt = tmp_path / "remote attempt"
    launch = (attempt / "launch-attempt.json").read_bytes()
    (attempt / "receipt.json").unlink()
    before = _transport_calls(fake_transport)

    assert staging.check_runtime(evidence)["kind"] == "pending"
    checked = subprocess.run(
        [sys.executable, str(SCRIPT), "check", "--evidence-dir", str(evidence)],
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert checked.returncode == 2
    assert json.loads(checked.stdout)["kind"] == "pending"
    extra = _transport_calls(fake_transport)[len(before) :]
    assert len(extra) == 2
    assert all(call["kind"] == "ssh" and "_check" in call["argv"][-1] for call in extra)
    assert (attempt / "launch-attempt.json").read_bytes() == launch
    assert not (attempt / "receipt.json").exists()
    assert _git(destination, "rev-parse", "HEAD") == source_repo.commit


@pytest.mark.parametrize("remote", [False, True])
def test_completed_check_leaves_git_index_bytes_and_mtime_unchanged(
    staging, source_repo, tmp_path, fake_transport, remote
):
    evidence, destination, _started = _stage(staging, source_repo, tmp_path, remote=remote)
    assert _finish(staging, evidence)["kind"] == "completed"
    tracked = destination / "tracked.txt"
    previous = tracked.stat()
    os.utime(tracked, ns=(previous.st_atime_ns, previous.st_mtime_ns + 1_000_000_000))
    index = destination / ".git" / "index"
    before = (index.read_bytes(), index.stat().st_mtime_ns)

    assert staging.check_runtime(evidence)["kind"] == "completed"

    assert (index.read_bytes(), index.stat().st_mtime_ns) == before


def test_two_attempts_for_one_destination_launch_at_most_once(staging, source_repo, tmp_path, monkeypatch):
    destination = tmp_path / "runtime destination"
    evidence_dirs = [tmp_path / "first evidence", tmp_path / "second evidence"]
    produce = staging._produce_bundle
    ready = Barrier(2)

    def produce_together(*args):
        bundle = produce(*args)
        ready.wait(timeout=10)
        return bundle

    monkeypatch.setattr(staging, "_produce_bundle", produce_together)

    def start(evidence):
        try:
            return staging.stage_runtime(source_repo.root, source_repo.commit, destination, evidence)
        except RuntimeError as error:
            return error

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(start, evidence_dirs))
    assert sum(isinstance(outcome, RuntimeError) for outcome in outcomes) == 1
    winner = next(index for index, outcome in enumerate(outcomes) if isinstance(outcome, dict))
    assert outcomes[winner]["kind"] == "started"
    assert _finish(staging, evidence_dirs[winner])["kind"] == "completed"
    assert sum((evidence / "target" / "launch-attempt.json").exists() for evidence in evidence_dirs) == 1
    assert _git(destination, "rev-parse", "HEAD") == source_repo.commit
    assert (destination / "tracked.txt").read_text() == "requested commit\n"


def test_worker_cli_propagates_failed_receipt_exit(staging, monkeypatch, capsys):
    monkeypatch.setattr(staging, "_worker", lambda _attempt: {"kind": "failed", "returncode": 19})
    assert staging.main(["_worker", "/unused-test-attempt"]) == 1
    assert json.loads(capsys.readouterr().out) == {"kind": "failed", "returncode": 19}


@pytest.mark.parametrize("existing", ["destination", "evidence", "remote_attempt"])
def test_existing_paths_are_not_reused(staging, source_repo, tmp_path, fake_transport, existing):
    paths = {
        "destination": tmp_path / "runtime destination",
        "evidence": tmp_path / "evidence",
        "remote_attempt": tmp_path / "remote attempt",
    }
    paths[existing].mkdir()
    marker = paths[existing] / "existing.txt"
    marker.write_text("keep existing content\n")
    with pytest.raises((FileExistsError, ValueError, RuntimeError, subprocess.CalledProcessError)):
        _stage(staging, source_repo, tmp_path, remote=True)
    assert marker.read_text() == "keep existing content\n"


@pytest.mark.parametrize("remote", [False, True])
def test_repeated_attempt_never_starts_another_worker(staging, source_repo, tmp_path, fake_transport, remote):
    evidence, destination, _started = _stage(staging, source_repo, tmp_path, remote=remote)
    completed = _finish(staging, evidence)
    before = _transport_calls(fake_transport)
    with pytest.raises((FileExistsError, ValueError, RuntimeError, subprocess.CalledProcessError)):
        _stage(staging, source_repo, tmp_path, remote=remote)
    assert _transport_calls(fake_transport) == before
    assert _finish(staging, evidence) == completed
    assert _git(destination, "rev-parse", "HEAD") == source_repo.commit


def test_stage_cli_started_is_not_shell_success(staging, source_repo, tmp_path, fake_transport):
    evidence = tmp_path / "evidence"
    destination = tmp_path / "runtime destination"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "stage",
            "--source-repo",
            str(source_repo.root),
            "--commit",
            source_repo.commit,
            "--destination",
            str(destination),
            "--evidence-dir",
            str(evidence),
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 2, result.stderr
    assert json.loads(result.stdout)["kind"] == "started"
    assert _finish(staging, evidence)["kind"] == "completed"
    checked = subprocess.run(
        [sys.executable, str(SCRIPT), "check", "--evidence-dir", str(evidence)],
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert checked.returncode == 0, checked.stderr
    assert json.loads(checked.stdout)["kind"] == "completed"
