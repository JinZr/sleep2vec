from __future__ import annotations

import json
from pathlib import Path
import subprocess
from typing import Any

from . import python_programs, transport
from .models import is_full_git_object_id
from .runtime_lock import runtime_lock

# The embedded Git steps are individually bounded, but lock contention is not. Wait for definitive remote evidence.
REMOTE_SYNC_TIMEOUT_SECONDS: float | None = None
GIT_REPOSITORY_ENV = (
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_COMMON_DIR",
    "GIT_CONFIG",
    "GIT_CONFIG_COUNT",
    "GIT_CONFIG_PARAMETERS",
    "GIT_DIR",
    "GIT_GRAFT_FILE",
    "GIT_IMPLICIT_WORK_TREE",
    "GIT_INDEX_FILE",
    "GIT_NO_REPLACE_OBJECTS",
    "GIT_OBJECT_DIRECTORY",
    "GIT_PREFIX",
    "GIT_REPLACE_REF_BASE",
    "GIT_SHALLOW_FILE",
    "GIT_WORK_TREE",
)


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
    checkout = root
    before = _commit(_git(checkout, "rev-parse", "HEAD").stdout, "runtime HEAD")
    _require_clean_runtime(root)

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
        _require_update_keeps_bytecode_sources(root, upstream)
        # Rolling runtimes move only through a normal fast-forward; never rewrite local history.
        _git(checkout, "merge", "--ff-only", "--no-edit", upstream, timeout=60)
        after = _commit(_git(checkout, "rev-parse", "HEAD").stdout, "updated runtime HEAD")
        if after != upstream:
            raise RuntimeError(f"Runtime update ended at {after}, expected {upstream}.")
        _require_clean_runtime(root)
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
    result = transport.run_shell(host, command, timeout=REMOTE_SYNC_TIMEOUT_SECONDS)
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
    _require_no_hidden_tracked_changes(workdir)
    dirty = _git(workdir, "status", "--porcelain", "--untracked-files=no").stdout.strip()
    if dirty:
        raise RuntimeError("Runtime checkout has tracked worktree changes; refusing to update it.")
    for flags in (("--others", "--exclude-standard"), ("--others", "--ignored", "--exclude-standard")):
        args = ["ls-files", *flags, "--", "*.py", "*.pyi", "*.pyc", "*.so"]
        paths = _git(workdir, *args).stdout.splitlines()
        if any(_is_importable_code(path, workdir) for path in paths):
            raise RuntimeError("Runtime checkout has untracked or ignored importable code; refusing to update it.")
        directories = _git(workdir, "ls-files", *flags, "--directory", "--no-empty-directory").stdout.splitlines()
        if any(_is_importable_package_symlink(path, workdir) for path in directories):
            raise RuntimeError("Runtime checkout has untracked or ignored importable code; refusing to update it.")


def _require_no_hidden_tracked_changes(workdir: str) -> None:
    entries = _git(workdir, "ls-files", "-v", "-z").stdout.split("\0")
    hidden_paths = [entry[2:] for entry in entries if len(entry) > 2 and entry[0] in {"h", "s", "S"}]
    for path in hidden_paths:
        index_entry = _git(workdir, "ls-files", "--stage", "--", path).stdout.split()
        candidate = Path(workdir) / path
        if len(index_entry) < 3 or index_entry[2] != "0" or candidate.is_symlink() or not candidate.is_file():
            raise RuntimeError("Runtime checkout has index-hidden tracked worktree changes; refusing to update it.")
        worktree_hash = _git(workdir, "hash-object", "--filters", f"--path={path}", path).stdout.strip()
        if worktree_hash != index_entry[1]:
            raise RuntimeError("Runtime checkout has index-hidden tracked worktree changes; refusing to update it.")


def _require_update_keeps_bytecode_sources(workdir: str, upstream: str) -> None:
    source_paths = set()
    for flags in (("--others", "--exclude-standard"), ("--others", "--ignored", "--exclude-standard")):
        paths = _git(workdir, "ls-files", *flags, "--", "*.pyc").stdout.splitlines()
        for raw_path in paths:
            path = Path(raw_path)
            if (
                "__pycache__" not in path.parts
                and path.stem.isidentifier()
                and all(part.isidentifier() for part in path.parts[:-1])
                and (Path(workdir) / path).with_suffix(".py").exists()
            ):
                source_paths.add(path.with_suffix(".py").as_posix())
    if not source_paths:
        return
    upstream_sources = set(
        _git(workdir, "ls-tree", "-r", "--name-only", upstream, "--", *sorted(source_paths)).stdout.splitlines()
    )
    if source_paths - upstream_sources:
        raise RuntimeError(
            "Runtime update would leave untracked or ignored sourceless bytecode; refusing to update it."
        )


def _is_importable_code(raw_path: str, workdir: str) -> bool:
    path = Path(raw_path)
    candidate = Path(workdir) / path
    if path.suffix == ".pyc":
        return (
            "__pycache__" not in path.parts
            and path.stem.isidentifier()
            and all(part.isidentifier() for part in path.parts[:-1])
            and not candidate.with_suffix(".py").exists()
        )
    module_name = path.name.split(".", 1)[0]
    return (
        path.suffix in {".py", ".pyi", ".so"}
        and module_name.isidentifier()
        and all(part.isidentifier() for part in path.parts[:-1])
    )


def _is_importable_package_symlink(raw_path: str, workdir: str) -> bool:
    path = Path(raw_path)
    candidate = Path(workdir) / path
    return (
        candidate.is_symlink()
        and candidate.is_dir()
        and path.name.isidentifier()
        and all(part.isidentifier() for part in path.parts[:-1])
    )


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
    sanitized = tuple(part for name in GIT_REPOSITORY_ENV for part in ("-u", name))
    command = " ".join(
        transport.sh(part)
        for part in ("env", *sanitized, "git", "-c", "core.hooksPath=/dev/null", "-C", workdir, *args)
    )
    return transport.run_shell(None, command, timeout=timeout)


def _raise_git_error(result: subprocess.CompletedProcess) -> None:
    detail = result.stderr.strip() or result.stdout.strip() or f"exit code {result.returncode}"
    raise RuntimeError(f"Runtime Git command failed: {detail}")
