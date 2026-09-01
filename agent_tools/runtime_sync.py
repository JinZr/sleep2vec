from __future__ import annotations

import json
from pathlib import Path
import subprocess
from typing import Any

from . import python_programs, transport
from .models import is_full_git_object_id
from .runtime_lock import runtime_lock


def sync_runtime(
    workdir: str | Path,
    *,
    host: str | None = None,
    remote_python: str = "python3",
    execute: bool = False,
) -> dict[str, Any]:
    if host:
        return _sync_remote(str(workdir), host, remote_python=remote_python, execute=execute)
    checkout = str(workdir)
    with runtime_lock(checkout):
        return _sync_local(checkout, execute=execute)


def _sync_local(checkout: str, *, execute: bool) -> dict[str, Any]:
    root = _git(checkout, "rev-parse", "--show-toplevel").stdout.strip()
    before = _commit(_git(checkout, "rev-parse", "HEAD").stdout, "runtime HEAD")
    _require_clean_runtime(checkout)

    if execute:
        _git(checkout, "fetch", "--no-tags", "origin", "main", timeout=60)
        upstream = _commit(_git(checkout, "rev-parse", "FETCH_HEAD").stdout, "fetched origin/main")
    else:
        result = _git(checkout, "ls-remote", "--exit-code", "origin", "refs/heads/main", timeout=60)
        fields = result.stdout.split()
        if len(fields) != 2 or fields[1] != "refs/heads/main":
            raise RuntimeError("origin/main lookup returned malformed output.")
        upstream = _commit(fields[0], "origin/main")

    if before == upstream:
        status = "unchanged"
        after = before
    elif not execute:
        # ls-remote proves a different tip, but dry-run intentionally does not fetch enough objects to prove ancestry.
        status = "update_available"
        after = before
    else:
        ancestor = _git_result(checkout, "merge-base", "--is-ancestor", before, upstream)
        if ancestor.returncode == 1:
            raise RuntimeError(
                f"Runtime HEAD {before} has diverged from origin/main {upstream}; refusing a non-fast-forward update."
            )
        if ancestor.returncode != 0:
            _raise_git_error(ancestor)
        # Rolling runtimes move only through a normal fast-forward; never rewrite local history.
        _git(checkout, "merge", "--ff-only", "--no-edit", upstream, timeout=60)
        after = _commit(_git(checkout, "rev-parse", "HEAD").stdout, "updated runtime HEAD")
        if after != upstream:
            raise RuntimeError(f"Runtime update ended at {after}, expected {upstream}.")
        _require_clean_runtime(checkout)
        status = "fast_forwarded"

    return {
        "status": status,
        "executed": execute,
        "host": "",
        "workdir": root,
        "before_commit": before,
        "upstream_commit": upstream,
        "after_commit": after,
    }


def _sync_remote(workdir: str, host: str, *, remote_python: str, execute: bool) -> dict[str, Any]:
    # The target checkout may predate runtime-sync, so bootstrap it with the manager's self-contained program.
    argv = [remote_python, "-c", python_programs.source("runtime_sync.sync"), workdir, "1" if execute else "0"]
    command = " ".join(transport.sh(part) for part in argv)
    result = transport.run_shell(host, command, timeout=120)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or f"exit code {result.returncode}"
        raise RuntimeError(f"Remote runtime sync failed on {host}: {detail}")
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Remote runtime sync returned malformed evidence on {host}.") from exc
    fields = {
        "status",
        "executed",
        "host",
        "workdir",
        "before_commit",
        "upstream_commit",
        "after_commit",
    }
    if (
        not isinstance(payload, dict)
        or set(payload) != fields
        or payload["status"] not in {"unchanged", "update_available", "fast_forwarded"}
        or type(payload["executed"]) is not bool
        or payload["executed"] is not execute
        or payload["host"] != ""
        or not isinstance(payload["workdir"], str)
        or not Path(payload["workdir"]).is_absolute()
        or any(
            not is_full_git_object_id(payload[field]) for field in ("before_commit", "upstream_commit", "after_commit")
        )
    ):
        raise RuntimeError(f"Remote runtime sync returned malformed evidence on {host}.")
    before = payload["before_commit"]
    upstream = payload["upstream_commit"]
    after = payload["after_commit"]
    status = payload["status"]
    valid_status = (
        (status == "unchanged" and before == upstream == after)
        or (status == "update_available" and not execute and before != upstream and after == before)
        or (status == "fast_forwarded" and execute and before != upstream and after == upstream)
    )
    if not valid_status:
        raise RuntimeError(f"Remote runtime sync returned inconsistent evidence on {host}.")
    payload["host"] = host
    return payload


def _require_clean_runtime(workdir: str) -> None:
    dirty = _git(workdir, "status", "--porcelain", "--untracked-files=no").stdout.strip()
    if dirty:
        raise RuntimeError("Runtime checkout has tracked worktree changes; refusing to update it.")
    for flags in (("--others", "--exclude-standard"), ("--others", "--ignored", "--exclude-standard")):
        args = ["ls-files", *flags, "--", "*.py", "*.pyi", "*.pyc", "*.so"]
        paths = _git(workdir, *args).stdout.splitlines()
        if any(_is_importable_code(path, workdir) for path in paths):
            raise RuntimeError("Runtime checkout has untracked or ignored importable code; refusing to update it.")


def _is_importable_code(raw_path: str, workdir: str) -> bool:
    path = Path(raw_path)
    if path.suffix == ".pyc":
        return (
            "__pycache__" not in path.parts
            and path.stem.isidentifier()
            and all(part.isidentifier() for part in path.parts[:-1])
            and not (Path(workdir) / path).with_suffix(".py").exists()
        )
    module_name = path.name.split(".", 1)[0]
    return module_name.isidentifier() and all(part.isidentifier() for part in path.parts[:-1])


def _commit(value: str, label: str) -> str:
    commit = value.strip().lower()
    if not is_full_git_object_id(commit):
        raise RuntimeError(f"{label} is not a full Git object ID: {commit!r}")
    return commit


def _git(
    workdir: str,
    *args: str,
    timeout: float = transport.SSH_TIMEOUT_SECONDS,
) -> subprocess.CompletedProcess:
    result = _git_result(workdir, *args, timeout=timeout)
    if result.returncode != 0:
        _raise_git_error(result)
    return result


def _git_result(
    workdir: str,
    *args: str,
    timeout: float = transport.SSH_TIMEOUT_SECONDS,
) -> subprocess.CompletedProcess:
    command = " ".join(transport.sh(part) for part in ("git", "-c", "core.hooksPath=/dev/null", "-C", workdir, *args))
    return transport.run_shell(None, command, timeout=timeout)


def _raise_git_error(result: subprocess.CompletedProcess) -> None:
    detail = result.stderr.strip() or result.stdout.strip() or f"exit code {result.returncode}"
    raise RuntimeError(f"Runtime Git command failed: {detail}")
