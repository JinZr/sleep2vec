"""Parity between ``runtime_lock`` and the inline ``runtime_guard`` copy embedded programs carry.

Both must resolve the same checkout-root lock file and refuse the same inputs; otherwise a
``runtime_sync`` fast-forward and a managed launch could hold different locks.
"""

from __future__ import annotations

import fcntl
import os
from pathlib import Path
import threading

import pytest

from agent_tools import python_programs
from agent_tools.runtime_lock import runtime_lock


def _embedded_runtime_guard():
    fragment = python_programs._SOURCE_ROOT / python_programs._FRAGMENTS["runtime_guard"]
    namespace: dict[str, object] = {}
    exec(compile(fragment.read_text(encoding="utf-8"), str(fragment), "exec"), namespace)
    return namespace["runtime_guard"]


LOCKS = {"runtime_lock": runtime_lock, "runtime_guard": _embedded_runtime_guard()}


def _assert_held(path: Path) -> None:
    descriptor = os.open(path, os.O_RDWR)
    try:
        with pytest.raises(BlockingIOError):
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        os.close(descriptor)


@pytest.mark.parametrize("marker", ["directory", "worktree_file"])
@pytest.mark.parametrize("name", sorted(LOCKS))
def test_lock_resolves_the_checkout_root(tmp_path: Path, name: str, marker: str):
    outer = tmp_path / "outer"
    (outer / ".git").mkdir(parents=True)
    checkout = outer / "checkout"
    workdir = checkout / "nested" / "dir"
    workdir.mkdir(parents=True)
    if marker == "directory":
        (checkout / ".git").mkdir()
    else:
        (checkout / ".git").write_text("gitdir: /elsewhere/.git/worktrees/checkout\n")

    with LOCKS[name](workdir):
        _assert_held(checkout / ".agent-tools-runtime.lock")

    assert not (outer / ".agent-tools-runtime.lock").exists()
    assert not (workdir / ".agent-tools-runtime.lock").exists()


@pytest.mark.parametrize(
    ("layout", "message"),
    [
        ("malformed_marker", "Malformed Git worktree marker"),
        ("no_checkout", "Cannot locate the runtime checkout root"),
        ("fifo_lock", "Runtime lock is not a regular file"),
    ],
)
@pytest.mark.parametrize("name", sorted(LOCKS))
def test_lock_refuses_the_same_layouts(tmp_path: Path, name: str, layout: str, message: str):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    if layout == "malformed_marker":
        (checkout / ".git").write_text("not a gitdir line\n")
    elif layout == "fifo_lock":
        (checkout / ".git").mkdir()
        os.mkfifo(checkout / ".agent-tools-runtime.lock")

    with pytest.raises(RuntimeError, match=message):
        with LOCKS[name](checkout):
            pass


@pytest.mark.parametrize("name", sorted(LOCKS))
def test_lock_refuses_a_symlinked_lock_file(tmp_path: Path, name: str):
    checkout = tmp_path / "checkout"
    (checkout / ".git").mkdir(parents=True)
    outside = tmp_path / "outside.lock"
    outside.write_text("")
    (checkout / ".agent-tools-runtime.lock").symlink_to(outside)

    with pytest.raises(OSError):
        with LOCKS[name](checkout):
            pass


@pytest.mark.parametrize(("holder", "waiter"), [("runtime_lock", "runtime_guard"), ("runtime_guard", "runtime_lock")])
def test_locks_exclude_each_other(tmp_path: Path, holder: str, waiter: str):
    checkout = tmp_path / "checkout"
    (checkout / ".git").mkdir(parents=True)
    acquired = threading.Event()

    def wait_for_lock() -> None:
        with LOCKS[waiter](checkout):
            acquired.set()

    with LOCKS[holder](checkout):
        thread = threading.Thread(target=wait_for_lock, daemon=True)
        thread.start()
        assert not acquired.wait(0.3)
    thread.join(5)
    assert acquired.is_set()
