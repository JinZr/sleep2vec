from __future__ import annotations

import os
from pathlib import Path

import pytest

from agent_tools import run_evidence


@pytest.mark.parametrize(
    "separator", ["\n", "\r", "\r\n", "\v", "\f", "\x1c", "\x1d", "\x1e", "\x85", "\u2028", "\u2029"]
)
@pytest.mark.parametrize("trailing", [False, True])
@pytest.mark.parametrize("lines", [1, 8, 100, 0, -2])
def test_local_tail_preserves_splitlines(tmp_path, separator, trailing, lines):
    path = tmp_path / "log.txt"
    text = separator.join(["  α😀  ", "", "middle", "last  "] * 9000)
    path.write_bytes((text + (separator if trailing else "")).encode("utf-8"))
    expected = "\n".join(path.read_text(errors="replace").splitlines()[-lines:])
    assert run_evidence.log_tail(path, lines=lines) == expected


@pytest.mark.parametrize(
    "payload",
    [b"", b"one", b"one\n", b"one\n\n", b"\xffbad\r\nlast\xf0\x9f", b"x" * 200000, b"x" * 200000 + b"\nend"],
)
@pytest.mark.parametrize("lines", [1, 8, 100])
def test_local_tail_preserves_empty_long_and_invalid_text(tmp_path, payload, lines):
    path = tmp_path / "log.txt"
    path.write_bytes(payload)
    assert run_evidence.log_tail(path, lines=lines) == "\n".join(path.read_text(errors="replace").splitlines()[-lines:])


@pytest.mark.parametrize("boundary", ["😀", "\r\n", "\x85", "\u2028", "\u2029", "\ufffd"])
@pytest.mark.parametrize("offset", range(1, 5))
def test_local_tail_window_can_split_utf8_and_separators(tmp_path, boundary, offset):
    path = tmp_path / "log.txt"
    suffix = b"a\nb\nc\n" + b"z" * (65536 - offset - 6)
    path.write_bytes(b"old\n" * 20000 + boundary.encode("utf-8") + suffix)
    assert run_evidence.log_tail(path, lines=2) == "\n".join(path.read_text(errors="replace").splitlines()[-2:])


def test_local_tail_and_failure_read_only_bounded_suffix(tmp_path, monkeypatch):
    path = tmp_path / "log.txt"
    path.write_bytes(b"old log record\n" * 100000 + b"AGENT_TOOLS_EXIT_CODE=0\n")
    real_open = Path.open
    sizes = []

    class Buffer:
        def __init__(self, source):
            self.source = source

        def seek(self, *args):
            return self.source.seek(*args)

        def read(self, size):
            sizes.append(size)
            return self.source.read(size)

    class Source:
        def __init__(self, source):
            self.source = source
            self.buffer = Buffer(source.buffer)

        def __getattr__(self, name):
            return getattr(self.source, name)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.source.close()

    monkeypatch.setattr(Path, "open", lambda target, *args, **kwargs: Source(real_open(target, *args, **kwargs)))
    assert run_evidence.log_tail(path).endswith("AGENT_TOOLS_EXIT_CODE=0")
    assert run_evidence.log_has_failure(path, require_exit_code=True) is False
    assert sizes == [65536, 65536]


def test_local_tail_keeps_non_utf8_text_encoding(tmp_path, monkeypatch):
    path = tmp_path / "log.txt"
    path.write_bytes(b"caf\xe9\n" * 20000 + b"fin\xe9")
    expected = "\n".join(path.read_text(encoding="latin-1", errors="replace").splitlines()[-8:])
    real_open = Path.open

    def open_latin1(target, *args, **kwargs):
        return real_open(target, *args, encoding="latin-1", **kwargs)

    monkeypatch.setattr(Path, "open", open_latin1)
    assert run_evidence.log_tail(path) == expected


def test_local_tail_reads_zero_size_virtual_file(tmp_path, monkeypatch):
    path = tmp_path / "log.txt"
    path.write_text("first\nlast\n")
    real_fstat = os.fstat

    def zero_size(fd):
        values = list(real_fstat(fd))
        values[6] = 0
        return os.stat_result(values)

    monkeypatch.setattr(os, "fstat", zero_size)
    assert run_evidence.log_tail(path, lines=1) == "last"


@pytest.mark.parametrize("error", [PermissionError("denied"), OSError("read failed"), FileNotFoundError("removed")])
def test_local_tail_does_not_hide_open_errors(tmp_path, monkeypatch, error):
    path = tmp_path / "log.txt"
    path.write_text("present")

    def fail_open(*_args, **_kwargs):
        raise error

    monkeypatch.setattr(Path, "open", fail_open)
    with pytest.raises(type(error), match=str(error)):
        run_evidence.log_tail(path)
    with pytest.raises(type(error), match=str(error)):
        run_evidence.log_has_failure(path)


def test_local_tail_missing_keeps_failure_policy(tmp_path):
    path = tmp_path / "missing.log"
    assert run_evidence.log_tail(path) == ""
    assert run_evidence.log_has_failure(path) is False
    assert run_evidence.log_has_failure(path, require_exit_code=True) is True


@pytest.mark.parametrize(
    "exit_line,failed",
    [("AGENT_TOOLS_EXIT_CODE=0", False), ("AGENT_TOOLS_EXIT_CODE=17", True), ("0: AGENT_TOOLS_EXIT_CODE=0", False)],
)
def test_local_failure_tail_keeps_terminal_marker_priority(tmp_path, exit_line, failed):
    path = tmp_path / "log.txt"
    path.write_text("RuntimeError old\n" + "normal\n" * 20000 + exit_line + "\n")
    assert run_evidence.log_has_failure(path, {"scheduler_type": "slurm"}, require_exit_code=True) is failed
