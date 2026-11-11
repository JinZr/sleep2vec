from __future__ import annotations

import json
from pathlib import Path

from agent_tools.progress import read_progress, write_progress


def test_progress_helper_writes_valid_json_and_reader_reads_it(tmp_path: Path):
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


def test_progress_missing_returns_structured_status(tmp_path: Path):
    data = read_progress(tmp_path)

    assert data["status"] == "missing"
    assert "progress file not found" in data["message"]


def test_progress_empty_or_invalid_json_returns_unknown(tmp_path: Path):
    progress = tmp_path / "status" / "progress.json"
    progress.parent.mkdir()
    progress.write_text("")

    data = read_progress(tmp_path)

    assert data["status"] == "unknown"
    assert "empty" in data["message"]

    progress.write_text("{")
    data = read_progress(tmp_path)

    assert data["status"] == "unknown"
    assert "not valid JSON" in data["message"]
