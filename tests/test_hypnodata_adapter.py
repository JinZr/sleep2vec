from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from hypnodata.config import load_config
from hypnodata.pipeline import run_pipeline
from tests.hypnodata_test_helpers import write_tiny_edf


def test_custom_adapter_hooks_cover_multifile_metadata_header_scoring_and_annotations(
    tmp_path: Path,
    monkeypatch,
):
    adapter_module = tmp_path / "toy_adapter.py"
    adapter_module.write_text("""
from dataclasses import replace
from pathlib import Path

import pandas as pd

from hypnodata.annotations import AnnotationResult, read_stage_csv
from hypnodata.edf import EdfInventory
from hypnodata.records import RecordTask


class ToyAdapter:
    def collect_records(self, config):
        frame = pd.read_csv(config.adapter_options["index"], low_memory=False)
        records = []
        for _, row in frame.iterrows():
            records.append(
                RecordTask(
                    record_id=str(row["record_id"]),
                    center=config.center,
                    files={
                        "left": Path(row["left_edf"]),
                        "right": Path(row["right_edf"]),
                        "stage": Path(row["stage_csv"]),
                    },
                    metadata={"source": row["source"], "split": row["split"]},
                    source_row=row.to_dict(),
                )
            )
        return records

    def resolve_metadata(self, record, config):
        return {
            "subject_id": record.files["left"].parent.name,
            "session_id": record.source_row["session"],
        }

    def fix_header(self, record, inventories, config):
        fixed = {}
        for key, inventory in inventories.items():
            signals = [
                replace(signal, unit=config.adapter_options["unit_override"])
                if signal.unit is None and signal.raw_label == config.adapter_options["preferred_label"]
                else signal
                for signal in inventory.signals
            ]
            fixed[key] = EdfInventory(
                path=inventory.path,
                signals=signals,
                duration=inventory.duration,
                warnings=inventory.warnings,
            )
        return fixed

    def score_channel_candidate(self, record, canonical, spec, candidate, signal, config):
        return 10 if signal.raw_label == config.adapter_options["preferred_label"] else 0

    def read_annotations(self, record, config, duration_sec):
        signal = read_stage_csv(
            record.files["stage"],
            duration_sec=duration_sec,
            epoch_sec=config.adapter_options["stage_epoch_sec"],
            mapping=config.adapter_options["stage_mapping"],
            label_column="Type",
            start_column="Start",
            duration_column="Duration",
        )
        return AnnotationResult([signal])


def make_adapter(config):
    return ToyAdapter()
""")
    monkeypatch.syspath_prepend(str(tmp_path))
    subject_dir = tmp_path / "subject-a"
    subject_dir.mkdir()
    left = write_tiny_edf(subject_dir / "left.edf", label="C3-A1", duration_sec=10)
    right = write_tiny_edf(subject_dir / "right.edf", label="C3-A2", duration_sec=10, unit="")
    stage_csv = tmp_path / "stage.csv"
    pd.DataFrame(
        [
            {"Start": 0, "Duration": 5, "Type": "Wake"},
            {"Start": 5, "Duration": 5, "Type": "N1"},
        ]
    ).to_csv(stage_csv, index=False)
    adapter_index = tmp_path / "adapter_index.csv"
    pd.DataFrame(
        [
            {
                "record_id": "night1",
                "left_edf": str(left),
                "right_edf": str(right),
                "stage_csv": str(stage_csv),
                "source": "toy_source",
                "split": "train",
                "session": "adapter-session",
            }
        ]
    ).to_csv(adapter_index, index=False)
    config_path = tmp_path / "hypnodata.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "center": "toy_adapter_center",
                "record_discovery": {"type": "custom", "adapter": "toy_adapter:make_adapter"},
                "backend": {"type": "npz"},
                "adapter_options": {
                    "index": str(adapter_index),
                    "preferred_label": "C3-A2",
                    "unit_override": "uV",
                    "stage_epoch_sec": 5,
                    "stage_mapping": {"Wake": 0, "N1": 1},
                },
                "signals": {
                    "eeg": {
                        "kind": "eeg",
                        "required": True,
                        "target_sfreq": 10,
                        "target_unit": "uV",
                        "candidates": [{"regex": "^C3-", "priority": 1}],
                    },
                    "stage5": {
                        "kind": "stage",
                        "required": False,
                        "target_sfreq": 0.2,
                        "candidates": [],
                    },
                },
            }
        )
    )
    output_dir = tmp_path / "out"

    run_pipeline(load_config(config_path), output_dir=output_dir)

    signal_manifest = pd.read_csv(output_dir / "manifest" / "signal_manifest.csv")
    eeg = signal_manifest[signal_manifest["canonical_channel"] == "eeg"].iloc[0]
    assert eeg["raw_label"] == "C3-A2"
    assert eeg["raw_unit"] == "uV"
    assert "adapter_score=10" in eeg["selection_reason"]
    stage = signal_manifest[signal_manifest["canonical_channel"] == "stage5"].iloc[0]
    assert stage["selection_reason"] == "annotation"
    assert stage["mask_column"] == "stage_mask"

    record_manifest = pd.read_csv(output_dir / "manifest" / "record_manifest.csv")
    assert record_manifest.loc[0, "subject_id"] == "subject-a"
    assert record_manifest.loc[0, "session_id"] == "adapter-session"
    assert record_manifest.loc[0, "stage_mask"] == 1
    with np.load(output_dir / "backends" / "npz" / "records" / "night1.npz") as npz:
        np.testing.assert_array_equal(npz["stage5"], np.asarray([0, 1], dtype=np.int64))
