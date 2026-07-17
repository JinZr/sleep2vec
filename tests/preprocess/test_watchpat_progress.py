from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "module_name",
    [
        "preprocess.watchpat_zzp_to_edf",
        "sleep2expert.preprocess.watchpat_zzp_to_edf",
        "sleep2vec2.preprocess.watchpat_zzp_to_edf",
    ],
)
def test_watchpat_batch_conversion_writes_progress_status(tmp_path: Path, monkeypatch, module_name: str):
    module = importlib.import_module(module_name)
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    input_root.mkdir()
    files = [input_root / "a.zzp", input_root / "b.zzp"]
    for path in files:
        path.write_bytes(b"")

    def fake_convert_zzp_to_edf(**kwargs):
        Path(kwargs["output_path"]).write_text("edf")

    monkeypatch.setattr(module, "convert_zzp_to_edf", fake_convert_zzp_to_edf)
    args = argparse.Namespace(
        output_edf=str(output_root),
        json_summary=None,
        skip_existing=False,
        writer="manual",
        include_internal_1hz=False,
        no_pulse_rate=False,
        verbose=False,
    )

    exit_code = module._run_batch_conversion(args, input_root, files)

    assert exit_code == 0
    progress = json.loads((output_root / "status" / "progress.json").read_text())
    assert progress["task"] == "watchpat_zzp_to_edf"
    assert progress["status"] == "completed"
    assert progress["processed"] == 2
