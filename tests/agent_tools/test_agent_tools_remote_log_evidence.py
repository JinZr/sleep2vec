from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap

import pytest

from agent_tools import managed_scheduler, python_programs, run_evidence, transport

ROW = {"target": "ssh", "host": "fake-log-host", "scheduler_type": "slurm", "status": "running"}


@pytest.fixture
def fake_remote_tools(tmp_path, monkeypatch):
    binary_dir = tmp_path / "bin"
    binary_dir.mkdir()
    events = tmp_path / "events.jsonl"
    real_tail = shutil.which("tail")
    assert real_tail
    script = f"#!{sys.executable}\n" + textwrap.dedent("""\
        import base64
        import json
        import os
        from pathlib import Path
        import subprocess
        import sys

        tool = Path(sys.argv[0]).name
        with open(os.environ["LOG_PROBE_TEST_EVENTS"], "a") as stream:
            stream.write(json.dumps({"tool": tool, "args": sys.argv[1:]}) + "\\n")
        if tool == "ssh":
            assert sys.argv[1] == "fake-log-host"
            mode = os.environ.get("LOG_PROBE_TEST_PAIR_FAILURE")
            if mode and sys.argv[2].startswith("python3 -c "):
                if mode == "complete255":
                    result = subprocess.run(["/bin/sh", "-c", sys.argv[2]], capture_output=True)
                    sys.stdout.buffer.write(result.stdout)
                    sys.exit(255)
                if mode == "diagnostic0":
                    print("remote child failed")
                elif mode == "partial0":
                    print('[[0,"cGFydGlhbA==",""]')
                else:
                    assert mode == "empty0"
                sys.exit(0)
            os.execv("/bin/sh", ["sh", "-c", sys.argv[2]])
        if tool == "date":
            assert sys.argv[1:] == ["+%s"]
            print(os.environ["LOG_PROBE_TEST_NOW"])
        elif tool == "stat":
            assert sys.argv[1:3] == ["-c", "%Y"]
            if os.environ.get("LOG_PROBE_TEST_STAT_FAILURE"):
                print("stat: Permission denied", file=sys.stderr)
                sys.exit(1)
            try:
                print(int(Path(sys.argv[3]).stat().st_mtime))
            except FileNotFoundError:
                print("stat: No such file", file=sys.stderr)
                sys.exit(1)
        elif tool == "tail":
            if "LOG_PROBE_TEST_TAIL_STDOUT" in os.environ:
                stdout = base64.b64decode(os.environ["LOG_PROBE_TEST_TAIL_STDOUT"])
                stderr = base64.b64decode(os.environ.get("LOG_PROBE_TEST_TAIL_STDERR", ""))
                returncode = int(os.environ.get("LOG_PROBE_TEST_TAIL_RC", "0"))
            else:
                result = subprocess.run(
                    [os.environ["LOG_PROBE_TEST_REAL_TAIL"], *sys.argv[1:]], capture_output=True
                )
                stdout, stderr, returncode = result.stdout, result.stderr, result.returncode
            sys.stdout.buffer.write(stdout)
            sys.stderr.buffer.write(stderr)
            if os.environ.get("LOG_PROBE_TEST_ROTATE"):
                path = Path(sys.argv[-1])
                path.rename(str(path) + ".old")
                path.write_bytes(b"replacement log\\n")
                os.utime(path, (500, 500))
            sys.exit(returncode)
        else:
            raise AssertionError(tool)
        """)
    for name in ("ssh", "tail", "date", "stat"):
        binary = binary_dir / name
        binary.write_text(script)
        binary.chmod(0o700)
    (binary_dir / "python3").symlink_to(sys.executable)
    monkeypatch.setenv("PATH", f"{binary_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("LOG_PROBE_TEST_EVENTS", str(events))
    monkeypatch.setenv("LOG_PROBE_TEST_REAL_TAIL", real_tail)
    monkeypatch.setenv("LOG_PROBE_TEST_NOW", "1000")
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    return events


@pytest.fixture(autouse=True)
def reject_real_ssh(monkeypatch, tmp_path):
    real_popen = subprocess.Popen

    def guarded_popen(args, *positional, **kwargs):
        if isinstance(args, (list, tuple)) and Path(args[0]).name == "ssh":
            environment = kwargs.get("env") or os.environ
            executable = shutil.which(str(args[0]), path=environment.get("PATH"))
            assert executable == str(tmp_path / "bin" / "ssh"), "Real SSH is forbidden in log evidence tests"
        return real_popen(args, *positional, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", guarded_popen)


def _events(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def _payload(tail=b"paired tail\n", age=b"13\n", *, tail_rc=0, age_rc=0, tail_stderr=b"", age_stderr=b""):
    return json.dumps(
        [
            [returncode, base64.b64encode(stdout).decode(), base64.b64encode(stderr).decode()]
            for returncode, stdout, stderr in [(tail_rc, tail, tail_stderr), (age_rc, age, age_stderr)]
        ]
    )


@pytest.mark.parametrize("mtime", [987.75, 1000.75, 1234.75])
@pytest.mark.parametrize("content", [b"  first \n last \n\n", b"first\rlast\r\n", "旧行\n  café😀 \n".encode()])
def test_remote_pair_matches_existing_probes_and_uses_one_ssh(tmp_path, monkeypatch, fake_remote_tools, mtime, content):
    path = tmp_path / "log ' $ quoted ; file.txt"
    path.write_bytes(content)
    os.utime(path, (mtime, mtime))
    expected = run_evidence.log_tail(path, ROW, 1), run_evidence.log_age_seconds(path, ROW)
    fake_remote_tools.write_text("")

    assert run_evidence.log_tail_and_age(path, ROW, 1) == expected
    assert expected[1] == 1000 - int(mtime)
    calls = _events(fake_remote_tools)
    assert [call["tool"] for call in calls] == ["ssh", "tail", "date", "stat"]
    assert calls[1]["args"] == ["-n", "1", str(path)]
    assert calls[3]["args"] == ["-c", "%Y", str(path)]


def test_embedded_log_program_preserves_raw_component_bytes(tmp_path, fake_remote_tools):
    path = tmp_path / "log.txt"
    path.write_bytes(b"first\ninvalid \xff\r\n")
    os.utime(path, (987.75, 987.75))
    result = subprocess.run(
        [sys.executable, "-c", python_programs.source("run_evidence.log_tail_and_age"), str(path), "1"],
        capture_output=True,
        check=True,
    )
    payload = json.loads(result.stdout)
    assert payload == json.loads(_payload(b"invalid \xff\r\n", b"13\n"))
    assert result.stderr == b""
    assert [call["tool"] for call in _events(fake_remote_tools)] == ["tail", "date", "stat"]


@pytest.mark.parametrize("failure", ["tail", "stat", "both", "missing"])
def test_remote_pair_keeps_component_failures_independent(tmp_path, monkeypatch, fake_remote_tools, failure):
    path = tmp_path / "log.txt"
    if failure != "missing":
        path.write_text("visible tail\n")
        os.utime(path, (987, 987))
    if failure in {"tail", "both"}:
        monkeypatch.setenv("LOG_PROBE_TEST_TAIL_STDOUT", base64.b64encode(b"partial stdout").decode())
        monkeypatch.setenv("LOG_PROBE_TEST_TAIL_STDERR", base64.b64encode(b"tail: Permission denied\n").decode())
        monkeypatch.setenv("LOG_PROBE_TEST_TAIL_RC", "1")
    if failure in {"stat", "both"}:
        monkeypatch.setenv("LOG_PROBE_TEST_STAT_FAILURE", "1")

    expected_tail = "visible tail" if failure == "stat" else ""
    expected_age = 13 if failure == "tail" else None
    assert run_evidence.log_tail_and_age(path, ROW) == (expected_tail, expected_age)
    assert [call["tool"] for call in _events(fake_remote_tools)] == ["ssh", "tail", "date", "stat"]


def test_remote_pair_stats_path_after_tail_rotation(tmp_path, monkeypatch, fake_remote_tools):
    path = tmp_path / "log.txt"
    path.write_text("original tail\n")
    os.utime(path, (900, 900))
    monkeypatch.setenv("LOG_PROBE_TEST_ROTATE", "1")

    assert run_evidence.log_tail_and_age(path, ROW) == ("original tail", 500)
    assert path.read_text() == "replacement log\n"
    assert Path(str(path) + ".old").read_text() == "original tail\n"
    assert [call["tool"] for call in _events(fake_remote_tools)] == ["ssh", "tail", "date", "stat"]


@pytest.mark.parametrize("health", [False, True])
def test_slurm_health_alone_uses_one_fresh_paired_probe(tmp_path, monkeypatch, fake_remote_tools, health):
    path = tmp_path / "log.txt"
    path.write_text("first observation\n")
    os.utime(path, (987, 987))
    monkeypatch.setattr(run_evidence, "runtime_artifacts", lambda _row: ("", {}, []))
    row = {**ROW, "log_path": str(path)}
    observed = managed_scheduler._slurm_artifact_observation(dict(row), health=health)
    assert observed["log_tail"] == "first observation"
    if health:
        assert observed["log_age_seconds"] == 13
        assert observed["health_status"] == "scheduler_running"
    else:
        assert "log_age_seconds" not in observed
    assert [call["tool"] for call in _events(fake_remote_tools)] == (
        ["ssh", "tail", "date", "stat"] if health else ["ssh", "tail"]
    )
    path.write_text("next observation\n")
    os.utime(path, (999, 999))
    fake_remote_tools.write_text("")
    observed = managed_scheduler._slurm_artifact_observation(dict(row), health=health)
    assert observed["log_tail"] == "next observation"
    if health:
        assert observed["log_age_seconds"] == 1
    assert sum(call["tool"] == "ssh" for call in _events(fake_remote_tools)) == 1


@pytest.mark.parametrize(
    "response,returncode",
    [
        ("", 0),
        ("child failed but SSH reported success", 0),
        ("[]", 0),
        ("null", 0),
        ('{"tail": "", "tail": "other", "age": null}', 0),
        ('[[0,"",""]]', 0),
        ('[[0,"",""],[0,"",""],[0,"",""]]', 0),
        ('[[0,""],[0,"",""]]', 0),
        ('[[0,"","","extra"],[0,"",""]]', 0),
        ('[[true,"",""],[0,"",""]]', 0),
        ('[[0.0,"",""],[0,"",""]]', 0),
        ('[["0","",""],[0,"",""]]', 0),
        ('[[0,null,""],[0,"",""]]', 0),
        ('[[0,"",{}],[0,"",""]]', 0),
        ('[[0,"***",""],[0,"",""]]', 0),
        ('[[0,"a",""],[0,"",""]]', 0),
        ('[[0,"",""],{}]', 0),
        (_payload() + " diagnostic", 0),
        (_payload()[:-1], 0),
        (_payload(), 255),
        (_payload(), 124),
    ],
)
def test_invalid_remote_pair_falls_back_to_each_exact_probe_once(monkeypatch, response, returncode):
    path = "log ' $ quoted ; file.txt"
    calls = []
    pair_command = transport.remote_python_program_command("run_evidence.log_tail_and_age", path, 3)
    tail_command = f"tail -n 3 {transport.sh(path)}"
    age_command = f"now=$(date +%s); m=$(stat -c %Y {transport.sh(path)} 2>/dev/null) || exit 1; echo $((now-m))"

    def run_command(row, command):
        assert row == ROW
        calls.append(command)
        if command == pair_command:
            return subprocess.CompletedProcess(command, returncode, response, "")
        if command == tail_command:
            return subprocess.CompletedProcess(command, 0, " fallback tail \n", "")
        assert command == age_command
        return subprocess.CompletedProcess(command, 0, "37\n", "")

    monkeypatch.setattr(run_evidence, "run_row_command", run_command)
    assert run_evidence.log_tail_and_age(path, ROW, 3) == ("fallback tail", 37)
    assert calls == [pair_command, tail_command, age_command]


def test_pair_timeout_uses_transport_timeout_boundary_then_exact_fallback(tmp_path, monkeypatch, fake_remote_tools):
    path = tmp_path / "log.txt"
    path.write_text("fallback\n")
    os.utime(path, (987, 987))
    real_run = subprocess.run
    calls = []

    def run(args, **kwargs):
        calls.append(args)
        if len(calls) == 1:
            assert args[:2] == ["ssh", ROW["host"]]
            raise subprocess.TimeoutExpired(args, kwargs["timeout"])
        return real_run(args, **kwargs)

    monkeypatch.setattr(subprocess, "run", run)
    assert run_evidence.log_tail_and_age(path, ROW) == ("fallback", 13)
    assert len(calls) == 3
    assert [call["tool"] for call in _events(fake_remote_tools)] == ["ssh", "tail", "ssh", "date", "stat"]


@pytest.mark.parametrize("mode", ["complete255", "empty0", "diagnostic0", "partial0"])
def test_fake_ssh_failure_discards_pair_and_runs_two_exact_fallbacks(tmp_path, monkeypatch, fake_remote_tools, mode):
    path = tmp_path / "log.txt"
    path.write_text("fallback tail\n")
    os.utime(path, (987, 987))
    monkeypatch.setenv("LOG_PROBE_TEST_PAIR_FAILURE", mode)
    assert run_evidence.log_tail_and_age(path, ROW) == ("fallback tail", 13)
    calls = _events(fake_remote_tools)
    assert sum(call["tool"] == "ssh" for call in calls) == 3
    assert [call["tool"] for call in calls][-5:] == ["ssh", "tail", "ssh", "date", "stat"]


@pytest.mark.parametrize("field", ["tail", "tail_stderr", "age", "age_stderr"])
@pytest.mark.parametrize("returncode", [0, 1])
def test_paired_component_decoding_stays_strict_without_fallback(monkeypatch, field, returncode):
    calls = []
    payload = _payload(**{field: b"\xff", "tail_rc": returncode, "age_rc": returncode})
    monkeypatch.setattr(run_evidence.locale, "getpreferredencoding", lambda _setlocale: "utf-8")

    def run_command(_row, command):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, payload, "")

    monkeypatch.setattr(run_evidence, "run_row_command", run_command)
    with pytest.raises(UnicodeDecodeError):
        run_evidence.log_tail_and_age("log.txt", ROW)
    assert len(calls) == 1


@pytest.mark.parametrize("stream", ["STDOUT", "STDERR"])
@pytest.mark.parametrize("returncode", [0, 1])
def test_pair_matches_original_strict_tail_decoding(tmp_path, monkeypatch, fake_remote_tools, stream, returncode):
    path = tmp_path / "log.txt"
    path.write_text("log\n")
    monkeypatch.setenv("LOG_PROBE_TEST_TAIL_STDOUT", "")
    monkeypatch.setenv(f"LOG_PROBE_TEST_TAIL_{stream}", base64.b64encode(b"\xff").decode())
    monkeypatch.setenv("LOG_PROBE_TEST_TAIL_RC", str(returncode))
    monkeypatch.setattr(run_evidence.locale, "getpreferredencoding", lambda _setlocale: "utf-8")
    with pytest.raises(UnicodeDecodeError):
        run_evidence.log_tail(path, ROW)
    fake_remote_tools.write_text("")
    with pytest.raises(UnicodeDecodeError):
        run_evidence.log_tail_and_age(path, ROW)
    assert sum(call["tool"] == "ssh" for call in _events(fake_remote_tools)) == 1


def test_pair_uses_controller_text_encoding(monkeypatch):
    monkeypatch.setattr(run_evidence.locale, "getpreferredencoding", lambda _setlocale: "latin-1")
    monkeypatch.setattr(
        run_evidence,
        "run_row_command",
        lambda _row, command: subprocess.CompletedProcess(command, 0, _payload(b"  caf\xe9\r\n next \r\n"), ""),
    )
    assert run_evidence.log_tail_and_age("log.txt", ROW) == ("café\n next", 13)


@pytest.mark.parametrize("path,row", [("local.log", {}), (None, ROW), ("", ROW)])
def test_local_or_empty_log_pair_reuses_existing_operations(monkeypatch, path, row):
    calls = []
    monkeypatch.setattr(run_evidence, "log_tail", lambda *args: calls.append(("tail", args)) or "tail")
    monkeypatch.setattr(run_evidence, "log_age_seconds", lambda *args: calls.append(("age", args)) or 7)
    monkeypatch.setattr(run_evidence, "run_row_command", lambda *_args: pytest.fail("No SSH for local or empty logs"))
    assert run_evidence.log_tail_and_age(path, row, 3) == ("tail", 7)
    assert calls == [("tail", (path, row, 3)), ("age", (path, row))]


def test_direct_health_keeps_tail_progress_io_age_order(tmp_path, monkeypatch):
    calls = []
    row = {**ROW, "scheduler_type": "", "log_path": "log.txt", "run_id": "run-000"}
    monkeypatch.setattr(run_evidence, "read_pid", lambda *_args: 123)
    monkeypatch.setattr(run_evidence, "process_running", lambda *_args: True)
    monkeypatch.setattr(run_evidence, "runtime_artifacts", lambda *_args: ("", {}, []))
    monkeypatch.setattr(run_evidence, "log_tail", lambda *_args: calls.append("tail") or "tail")
    monkeypatch.setattr(run_evidence, "read_run_progress", lambda *_args: calls.append("progress") or {})
    monkeypatch.setattr(run_evidence, "proc_io", lambda *_args: calls.append("io") or {})
    monkeypatch.setattr(run_evidence, "log_age_seconds", lambda *_args: calls.append("age") or 7)
    monkeypatch.setattr(run_evidence, "gpu_summary", lambda *_args: calls.append("gpu") or "")
    monkeypatch.setattr(run_evidence, "log_tail_and_age", lambda *_args: pytest.fail("Direct health must not use pair"))
    observed = run_evidence.status_row(tmp_path, row, script_commits_terminal_status=True, health=True)
    assert observed["log_tail"] == "tail"
    assert observed["log_age_seconds"] == 7
    assert calls == ["tail", "progress", "io", "age", "gpu"]
