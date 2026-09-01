from __future__ import annotations

from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import stat
from typing import Iterator


@contextmanager
def runtime_lock(workdir: str | Path) -> Iterator[int]:
    checkout = Path(workdir).resolve()
    path = _checkout_root(checkout) / ".agent-tools-runtime.lock"
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise RuntimeError(f"Runtime lock is not a regular file: {path}")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield descriptor
    finally:
        os.close(descriptor)


def _checkout_root(checkout: Path) -> Path:
    for root in (checkout, *checkout.parents):
        marker = root / ".git"
        if marker.is_dir():
            return root
        if marker.is_file():
            # Linked worktrees keep a .git marker file but still own an independent checkout-root lock.
            line = marker.read_text().strip()
            if line.startswith("gitdir: "):
                return root
            raise RuntimeError(f"Malformed Git worktree marker: {marker}")
    raise RuntimeError(f"Cannot locate the runtime checkout root from {checkout}.")
