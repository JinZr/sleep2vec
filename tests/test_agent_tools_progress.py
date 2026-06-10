from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from agent_tools.progress import read_progress, write_progress


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "agent_tools", *args], text=True, capture_output=True)


def test_progress_helper_writes_valid_json_and_cli_reads_it(tmp_path: Path):
    write_progress(
        tmp_path,
        status="running",
        task="unit_task",
        processed=2,
        total=4,
        success=2,
        failed=0,
        start_time=1.0,
        current_item="sample.npz",
    )

    data = json.loads((tmp_path / "status" / "progress.json").read_text())
    assert data["task"] == "unit_task"
    assert data["processed"] == 2
    assert read_progress(tmp_path)["status"] == "running"

    result = _run("progress", "--run-dir", str(tmp_path))

    assert result.returncode == 0
    assert "unit_task running" in result.stdout
    assert "2 / 4 done" in result.stdout


def test_progress_remote_uses_single_ssh_cat(monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(
            command,
            0,
            '{"status": "running", "task": "remote_task", "processed": 1, "total": 2}\n',
            "",
        )

    monkeypatch.setattr("agent_tools.progress.subprocess.run", fake_run)

    data = read_progress("/wujidata/run", remote="baichuan3")

    assert data["task"] == "remote_task"
    assert calls == [["ssh", "baichuan3", "cat /wujidata/run/status/progress.json"]]


def test_progress_missing_returns_structured_status(tmp_path: Path):
    data = read_progress(tmp_path)

    assert data["status"] == "missing"
    assert "progress file not found" in data["message"]
