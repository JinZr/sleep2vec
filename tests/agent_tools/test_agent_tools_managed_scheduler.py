import errno
import fcntl
import os
from types import SimpleNamespace

from agent_tools import experiment_io, managed_scheduler


def test_managed_run_lock_retries_transient_file_lock_eio(tmp_path, monkeypatch):
    attempts = 0
    delays = []
    real_flock = fcntl.flock

    def flaky_flock(file_descriptor, operation):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError(errno.EIO, os.strerror(errno.EIO))
        real_flock(file_descriptor, operation)

    monkeypatch.setattr(experiment_io, "fcntl", SimpleNamespace(LOCK_EX=fcntl.LOCK_EX, flock=flaky_flock))
    monkeypatch.setattr(experiment_io, "time", SimpleNamespace(sleep=delays.append))

    with managed_scheduler.managed_run_lock(tmp_path):
        assert (tmp_path / "run_manifest.tsv.lock").exists()

    assert attempts == 2
    assert delays == [0.1]
