from __future__ import annotations

import errno
import os
from pathlib import Path

import pytest

from agent_tools import run_evidence


@pytest.mark.parametrize("consumer", ["tail", "failure"])
@pytest.mark.parametrize("encoding", ["utf-8", "latin-1"])
@pytest.mark.parametrize("error_number", [errno.EIO, errno.EACCES, errno.ENOENT])
def test_local_log_read_errors_propagate_after_open(tmp_path, monkeypatch, consumer, encoding, error_number):
    path = tmp_path / "log.txt"
    path.write_bytes(b"ordinary record\n" * 10000)
    real_open = Path.open
    opened = []
    error = OSError(error_number, "injected read failure")

    def fail_read(*_args, **_kwargs):
        raise error

    def open_log(target, *args, **kwargs):
        source = real_open(target, *args, encoding=encoding, **kwargs)
        opened.append(source)
        if encoding == "utf-8":
            monkeypatch.setattr(source.buffer, "read", fail_read)
        else:
            monkeypatch.setattr(source, "read", fail_read)
        return source

    monkeypatch.setattr(Path, "open", open_log)
    with pytest.raises(OSError, match="injected read failure") as caught:
        if consumer == "tail":
            run_evidence.log_tail(path)
        else:
            run_evidence.log_has_failure(path, require_exit_code=True)
    assert caught.value is error
    assert len(opened) == 1
    assert opened[0].closed


@pytest.mark.parametrize("consumer", ["tail", "failure"])
def test_local_nonseekable_log_keeps_text_read_fallback(tmp_path, monkeypatch, consumer):
    path = tmp_path / "log.txt"
    payload = b"old\r\ninvalid \xff\r\nAGENT_TOOLS_EXIT_CODE=0\r\n"
    path.write_bytes(payload)
    expected = "\n".join(path.read_text(errors="replace").splitlines()[-8:])
    read_fd, write_fd = os.pipe()
    try:
        os.write(write_fd, payload)
    finally:
        os.close(write_fd)
    source = os.fdopen(read_fd, "r", errors="replace")
    real_read = source.read
    reads = []

    def read_text(*args):
        reads.append(args)
        return real_read(*args)

    def unexpected_seek(*_args):
        pytest.fail("A nonseekable log must retain the full text-read path")

    monkeypatch.setattr(source, "read", read_text)
    monkeypatch.setattr(source.buffer, "seek", unexpected_seek)
    monkeypatch.setattr(Path, "open", lambda *_args, **_kwargs: source)
    if consumer == "tail":
        assert run_evidence.log_tail(path) == expected
    else:
        assert run_evidence.log_has_failure(path, require_exit_code=True) is False
    assert reads == [()]
    assert source.closed


@pytest.mark.parametrize("marker", ["Traceback", "RuntimeError", "CUDA out of memory", "Error executing job"])
@pytest.mark.parametrize("following_lines", [99, 100])
@pytest.mark.parametrize("separator", ["\n", "\u2028"])
def test_local_failure_markers_use_exact_last_100_splitlines(tmp_path, marker, following_lines, separator):
    path = tmp_path / "log.txt"
    records = ["old record"] * 20000 + [marker] + ["normal record"] * following_lines
    path.write_text(separator.join(records) + separator)
    old_tail = "\n".join(path.read_text(errors="replace").splitlines()[-100:])
    assert run_evidence.log_has_failure(path) is (marker in old_tail)
    assert (marker in old_tail) is (following_lines == 99)


@pytest.mark.parametrize("consumer", ["tail", "failure"])
def test_local_log_rotation_after_open_reads_the_opened_file(tmp_path, monkeypatch, consumer):
    path = tmp_path / "log.txt"
    rotated = tmp_path / "rotated.txt"
    path.write_text("old record\n" * 20000 + "RuntimeError in opened log\n")
    expected = "\n".join(path.read_text(errors="replace").splitlines()[-8:])
    real_open = Path.open
    opened = []

    def open_log(target, *args, **kwargs):
        source = real_open(target, *args, **kwargs)
        opened.append(source)
        os.replace(path, rotated)
        with real_open(path, "w") as replacement:
            replacement.write("new file\nAGENT_TOOLS_EXIT_CODE=0\n")
        return source

    monkeypatch.setattr(Path, "open", open_log)
    if consumer == "tail":
        assert run_evidence.log_tail(path) == expected
    else:
        assert run_evidence.log_has_failure(path) is True
    assert len(opened) == 1
    assert opened[0].closed


def test_local_log_truncation_before_first_buffer_read_can_reach_new_bof(tmp_path, monkeypatch):
    path = tmp_path / "log.txt"
    path.write_bytes(b"old record\n" * 100000)
    replacement = b"new first\nnew last\n"
    real_open = Path.open
    reads = []

    def open_log(target, *args, **kwargs):
        source = real_open(target, *args, **kwargs)
        real_read = source.buffer.read

        def truncate_then_read(size):
            if not reads:
                with real_open(path, "wb") as writer:
                    writer.write(replacement)
            reads.append(size)
            return real_read(size)

        monkeypatch.setattr(source.buffer, "read", truncate_then_read)
        return source

    monkeypatch.setattr(Path, "open", open_log)
    # This controlled race checks one EOF shrink, not an atomic snapshot during arbitrary concurrent writes.
    assert run_evidence.log_tail(path, lines=1) == "new last"
    assert len(reads) > 1
