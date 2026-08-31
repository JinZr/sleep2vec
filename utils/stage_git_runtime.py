"""Stage one committed Git runtime; never activate it or run experiment checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import uuid

IDENTITY_KEYS = (
    "attempt_id",
    "attempt_dir",
    "destination",
    "commit",
    "bundle_sha256",
    "source_repo",
    "source_origin_sha256",
    "worker_sha256",
)
_CLAIM_PROGRAM = """import json, os, pathlib, sys
request = json.loads(sys.argv[1])
attempt = pathlib.Path(request['attempt_dir'])
attempt.mkdir()
pathlib.Path(request['destination']).mkdir()
with (attempt / 'request.json').open('x') as stream:
    json.dump(request, stream)
    stream.flush()
    os.fsync(stream.fileno())
identity = {key: request[key] for key in json.loads(sys.argv[2])}
print(json.dumps(dict(identity, kind='claimed')), flush=True)
"""


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _request_identity(request):
    return {key: request[key] for key in IDENTITY_KEYS}


def _write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _git_env():
    # Caller Git overrides must not redirect fixed source/destination operations.
    return dict(
        ((key, value) for key, value in os.environ.items() if not key.startswith("GIT_")),
        GIT_OPTIONAL_LOCKS="0",
    )


def _git(argv, *, cwd, steps, log, capture=False):
    log.write("Running: " + shlex.join(["git", *argv]) + "\n")
    log.flush()
    result = subprocess.run(
        ["git", *argv],
        cwd=cwd,
        env=_git_env(),
        text=True,
        stdout=subprocess.PIPE if capture else log,
        stderr=log,
    )
    steps.append({"argv": ["git", *argv], "returncode": result.returncode})
    if capture:
        log.write(result.stdout)
    log.flush()
    result.check_returncode()
    return result.stdout.strip() if capture else ""


def _produce_bundle(source_repo, commit, evidence_dir):
    evidence = Path(evidence_dir)
    repository = evidence / "producer.git"
    bundle = evidence / "runtime.bundle"
    steps: list[dict[str, list[str] | int]] = []
    with (evidence / "producer.log").open("x") as log:
        try:
            _git(["init", "--bare", str(repository)], cwd=evidence, steps=steps, log=log)
            _git(
                ["fetch", "--no-tags", source_repo, f"{commit}:refs/heads/runtime-staging"],
                cwd=repository,
                steps=steps,
                log=log,
            )
            actual = _git(
                ["rev-parse", "refs/heads/runtime-staging"], cwd=repository, steps=steps, log=log, capture=True
            )
            if actual != commit:
                raise ValueError("Bundle producer did not fetch the requested commit.")
            _git(["bundle", "create", str(bundle), "refs/heads/runtime-staging"], cwd=repository, steps=steps, log=log)
            _git(["bundle", "verify", str(bundle)], cwd=repository, steps=steps, log=log)
        finally:
            _write_json(evidence / "producer-results.json", steps)
    return bundle


def _positive_reply(result, request, kinds):
    if result.returncode != 0:
        raise RuntimeError(f"Transport failed with exit {result.returncode}; inspect this attempt, never replay it.")
    try:
        reply = json.loads(result.stdout)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Transport returned no complete operation reply; do not retry this attempt.") from exc
    if not isinstance(reply, dict) or any(reply.get(key) != request[key] for key in IDENTITY_KEYS):
        raise RuntimeError("Operation reply does not match the frozen attempt.")
    if reply.get("kind") not in kinds:
        raise RuntimeError("Operation reply does not establish the requested outcome.")
    if reply["kind"] == "started" and (type(reply.get("pid")) is not int or reply["pid"] <= 0):
        raise RuntimeError("Started reply has no positive worker PID.")
    if reply["kind"] == "pending" and "recorded_pid" in reply:
        if type(reply["recorded_pid"]) is not int or reply["recorded_pid"] <= 0:
            raise RuntimeError("Pending reply has an invalid recorded worker PID.")
    return reply


def _on_target(request, argv):
    if request["host"]:
        argv = ["ssh", "--", request["host"], shlex.join(argv)]
    return subprocess.run(argv, capture_output=True, text=True, timeout=30)


def stage_runtime(
    source_repo,
    commit,
    destination,
    evidence_dir,
    *,
    host=None,
    remote_python=None,
    remote_attempt_dir=None,
):
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("commit must be a complete lowercase 40-character SHA.")
    paths = [source_repo, destination, evidence_dir]
    if host:
        if not remote_python or not remote_attempt_dir:
            raise ValueError("Remote staging requires remote_python and remote_attempt_dir.")
        paths.append(remote_attempt_dir)
    elif remote_python is not None or remote_attempt_dir is not None:
        raise ValueError("Remote options require host.")
    if any(not Path(path).is_absolute() for path in paths):
        raise ValueError("Use explicit absolute source, destination and evidence paths.")
    if not host and os.path.lexists(destination):
        raise FileExistsError(f"Destination already exists: {destination}")
    evidence = Path(evidence_dir)
    evidence.mkdir()
    request = {
        "attempt_id": uuid.uuid4().hex,
        "attempt_dir": str(remote_attempt_dir if host else evidence / "target"),
        "destination": str(destination),
        "commit": commit,
        "source_repo": str(source_repo),
        "host": host,
        "python": remote_python if host else sys.executable,
    }
    _write_json(evidence / "intent.json", request)
    origin = subprocess.run(
        ["git", "-C", str(source_repo), "config", "--get", "remote.origin.url"],
        env=_git_env(),
        capture_output=True,
        text=True,
    )
    if origin.returncode not in (0, 1):
        origin.check_returncode()
    # Hash the exact origin rather than exposing credentials embedded in its URL.
    request["source_origin_sha256"] = hashlib.sha256(origin.stdout.rstrip("\n").encode()).hexdigest()
    bundle = _produce_bundle(str(source_repo), commit, evidence)
    request["bundle_sha256"] = _sha256(bundle)
    request["worker_sha256"] = _sha256(__file__)
    _write_json(evidence / "request.json", request)
    claim = _on_target(
        request,
        [request["python"], "-c", _CLAIM_PROGRAM, json.dumps(request), json.dumps(IDENTITY_KEYS)],
    )
    _positive_reply(claim, request, {"claimed"})
    attempt = Path(request["attempt_dir"])
    worker = attempt / "stage_git_runtime.py"
    if host:
        transfer = subprocess.run(
            [
                "scp",
                "-O",
                "--",
                str(bundle),
                str(Path(__file__).absolute()),
                f"{host}:{shlex.quote(str(attempt) + '/')}",
            ]
        )
        transfer.check_returncode()
    else:
        shutil.copyfile(bundle, attempt / "runtime.bundle")
        shutil.copyfile(__file__, worker)
    result = _on_target(request, [request["python"], str(worker), "_launch", str(attempt)])
    return _positive_reply(result, request, {"started"})


def _load_request(attempt):
    return json.loads((Path(attempt) / "request.json").read_text())


def _launch(attempt):
    attempt = Path(attempt)
    request = _load_request(attempt)
    identity = _request_identity(request)
    if _sha256(__file__) != request["worker_sha256"] or _sha256(attempt / "runtime.bundle") != request["bundle_sha256"]:
        raise ValueError("Transferred worker or bundle does not match its frozen hash.")
    _write_json(attempt / "launch-attempt.json", identity)
    with (attempt / "worker.log").open("xb", buffering=0) as log:
        child = subprocess.Popen(
            [sys.executable, str(Path(__file__).absolute()), "_worker", str(attempt)],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    return dict(identity, kind="started", pid=child.pid)


def _publish_receipt(attempt, receipt):
    descriptor, temporary = tempfile.mkstemp(prefix=".receipt-", dir=attempt)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(receipt, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        # Link publishes atomically but never replaces an existing terminal receipt.
        os.link(temporary, Path(attempt) / "receipt.json")
        parent = os.open(attempt, os.O_RDONLY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    finally:
        os.unlink(temporary)


def _worker(attempt):
    attempt = Path(attempt)
    request = _load_request(attempt)
    receipt = dict(_request_identity(request), kind="failed", returncode=1, steps=[])
    _write_json(attempt / "worker.json", dict(_request_identity(request), pid=os.getpid()))
    destination = request["destination"]
    try:
        bundle = attempt / "runtime.bundle"
        if _sha256(bundle) != request["bundle_sha256"]:
            raise ValueError("Bundle changed before the worker consumed it.")
        _git(
            ["clone", "--no-checkout", "--no-hardlinks", str(bundle), destination],
            cwd=attempt,
            steps=receipt["steps"],
            log=sys.stdout,
        )
        _git(["checkout", "--detach", request["commit"]], cwd=destination, steps=receipt["steps"], log=sys.stdout)
        head = _git(["rev-parse", "HEAD"], cwd=destination, steps=receipt["steps"], log=sys.stdout, capture=True)
        dirty = _git(
            ["status", "--porcelain", "--untracked-files=all", "--ignored"],
            cwd=destination,
            steps=receipt["steps"],
            log=sys.stdout,
            capture=True,
        )
        if head != request["commit"] or dirty:
            raise ValueError("Staged runtime is not the exact clean requested commit.")
        receipt.update(kind="completed", returncode=0, head=head, clean=True)
    except Exception as exc:
        receipt["error"] = str(exc)
        if isinstance(exc, subprocess.CalledProcessError):
            receipt["returncode"] = exc.returncode
    _publish_receipt(attempt, receipt)
    return receipt


def _validate_receipt(request, receipt):
    if _request_identity(receipt) != _request_identity(request):
        raise ValueError("Terminal receipt does not match this attempt.")
    if receipt["kind"] == "completed":
        expected = [
            [
                "git",
                "clone",
                "--no-checkout",
                "--no-hardlinks",
                str(Path(request["attempt_dir"]) / "runtime.bundle"),
                request["destination"],
            ],
            ["git", "checkout", "--detach", request["commit"]],
            ["git", "rev-parse", "HEAD"],
            ["git", "status", "--porcelain", "--untracked-files=all", "--ignored"],
        ]
        if (
            type(receipt.get("returncode")) is not int
            or receipt["returncode"] != 0
            or receipt.get("head") != request["commit"]
            or receipt.get("clean") is not True
            or receipt.get("steps") != [{"argv": argv, "returncode": 0} for argv in expected]
            or any(type(step["returncode"]) is not int for step in receipt["steps"])
        ):
            raise ValueError("Terminal receipt lacks successful fixed Git operation evidence.")
    elif receipt["kind"] != "failed" or type(receipt.get("returncode")) is not int or receipt["returncode"] == 0:
        raise ValueError("Malformed terminal receipt.")


def _check(attempt):
    attempt = Path(attempt)
    request = _load_request(attempt)
    path = attempt / "receipt.json"
    if not path.exists():
        pending = dict(_request_identity(request), kind="pending")
        worker_path = attempt / "worker.json"
        if worker_path.exists():
            worker = json.loads(worker_path.read_text())
            if _request_identity(worker) != _request_identity(request):
                raise ValueError("Recorded worker does not match this attempt.")
            if type(worker.get("pid")) is not int or worker["pid"] <= 0:
                raise ValueError("Recorded worker has no positive PID.")
            # This is a durable lookup handle, not evidence that the PID is still alive.
            pending["recorded_pid"] = worker["pid"]
        return pending
    receipt = json.loads(path.read_text())
    _validate_receipt(request, receipt)
    if receipt["kind"] == "completed":
        head = subprocess.check_output(
            ["git", "-C", request["destination"], "rev-parse", "HEAD"], env=_git_env(), text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "-C", request["destination"], "status", "--porcelain", "--untracked-files=all", "--ignored"],
            env=_git_env(),
            text=True,
        )
        if head != request["commit"] or dirty:
            raise ValueError("Runtime has changed since its completed staging receipt.")
    return receipt


def check_runtime(evidence_dir):
    request = json.loads((Path(evidence_dir) / "request.json").read_text())
    worker = str(Path(request["attempt_dir"]) / "stage_git_runtime.py")
    result = _on_target(request, [request["python"], worker, "_check", request["attempt_dir"]])
    receipt = _positive_reply(result, request, {"completed", "failed", "pending"})
    if receipt["kind"] != "pending":
        _validate_receipt(request, receipt)
    return receipt


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    internal = {"_launch": _launch, "_worker": _worker, "_check": _check}
    parser = argparse.ArgumentParser(description=__doc__)
    if argv and argv[0] in internal:
        command = argv[0]
        parser.add_argument("attempt")
        args = vars(parser.parse_args(argv[1:]))
    else:
        commands = parser.add_subparsers(dest="command", required=True)
        stage = commands.add_parser(
            "stage", help="Start one detached staging attempt; started returns exit 2, not success."
        )
        for name in ("source-repo", "commit", "destination", "evidence-dir"):
            stage.add_argument("--" + name, required=True)
        for name in ("host", "remote-python", "remote-attempt-dir"):
            stage.add_argument("--" + name)
        check = commands.add_parser("check", help="Read the original attempt; only verified completion returns exit 0.")
        check.add_argument("--evidence-dir", required=True)
        args = vars(parser.parse_args(argv))
        command = args.pop("command")
    try:
        if command == "stage":
            result = stage_runtime(**args)
        elif command == "check":
            result = check_runtime(**args)
        else:
            result = internal[command](args["attempt"])
        print(json.dumps(result), flush=True)
        if command == "_worker":
            return 0 if result["kind"] == "completed" else 1
        if command.startswith("_"):
            return 0
        return {"completed": 0, "failed": 1}.get(result["kind"], 2)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
