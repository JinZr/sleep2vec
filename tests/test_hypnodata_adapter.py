from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from data.utils import load_builtin_ahi_metadata
from hypnodata.config import load_config
from hypnodata.pipeline import run_pipeline
from tests.hypnodata_test_helpers import write_tiny_edf


def _write_annotation_only_adapter(tmp_path: Path, monkeypatch) -> str:
    adapter_module = tmp_path / "annotation_only_adapter.py"
    adapter_module.write_text("""
from pathlib import Path

from hypnodata.annotations import AnnotationResult, read_stage_csv
from hypnodata.records import RecordTask


class AnnotationOnlyAdapter:
    def collect_records(self, config):
        metadata = {"source": "annotation_source", "split": "test"}
        if config.adapter_options.get("include_duration", True):
            metadata["duration"] = config.adapter_options["duration"]
        files = {}
        if "stage_csv" in config.adapter_options:
            files["stage"] = Path(config.adapter_options["stage_csv"])
        return [RecordTask(record_id="night1", center=config.center, files=files, metadata=metadata)]

    def read_annotations(self, record, config, duration_sec):
        if config.adapter_options.get("mode") == "empty":
            return AnnotationResult()
        stage = read_stage_csv(
            record.files["stage"],
            duration_sec=duration_sec,
            epoch_sec=config.adapter_options["epoch_sec"],
            label_column="Type",
            start_column="Start",
            duration_column="Duration",
        )
        return AnnotationResult([stage])


def make_adapter(config):
    return AnnotationOnlyAdapter()
""")
    monkeypatch.syspath_prepend(str(tmp_path))
    return "annotation_only_adapter:make_adapter"


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
                        "candidates": ["C3-A2", "C3-A1"],
                    },
                    "stage5": {
                        "kind": "stage",
                        "required": False,
                        "epoch_sec": 5,
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
    assert eeg["selection_reason"] == "label:C3-A2"
    stage = signal_manifest[signal_manifest["canonical_channel"] == "stage5"].iloc[0]
    assert stage["selection_reason"] == "annotation"
    assert stage["mask_column"] == "stage_mask"

    record_manifest = pd.read_csv(output_dir / "manifest" / "record_manifest.csv")
    assert record_manifest.loc[0, "subject_id"] == "subject-a"
    assert record_manifest.loc[0, "session_id"] == "adapter-session"
    assert record_manifest.loc[0, "stage_mask"] == 1
    with np.load(output_dir / "backends" / "npz" / "records" / "night1.npz") as npz:
        np.testing.assert_allclose(npz["eeg"][:5], np.arange(5, dtype=np.float32))
        np.testing.assert_array_equal(npz["stage5"], np.asarray([0, 1], dtype=np.int64))


def test_custom_adapter_materializes_event_table_dense_and_anchor_annotations(tmp_path: Path, monkeypatch):
    adapter_module = tmp_path / "event_adapter.py"
    adapter_module.write_text("""
from pathlib import Path

import pandas as pd

from hypnodata.annotations import (
    AnnotationResult,
    filter_events_to_sleep_stages,
    materialize_anchor_events,
    materialize_dense_events,
    materialize_event_table,
    read_event_csv,
    read_stage_csv,
)
from hypnodata.records import RecordTask


class EventAdapter:
    def collect_records(self, config):
        frame = pd.read_csv(config.adapter_options["index"], low_memory=False)
        return [
            RecordTask(
                record_id=str(row["record_id"]),
                center=config.center,
                files={
                    "edf": Path(row["edf_path"]),
                    "stage": Path(row["stage_csv"]),
                    "events": Path(row["event_csv"]),
                },
            )
            for _, row in frame.iterrows()
        ]

    def read_annotations(self, record, config, duration_sec):
        stage = read_stage_csv(
            record.files["stage"],
            duration_sec=duration_sec,
            epoch_sec=5,
            label_column="Type",
            start_column="Start",
            duration_column="Duration",
        )
        events = read_event_csv(record.files["events"], mapping={"Apnea": 0, "Hypopnea": 1})
        events = filter_events_to_sleep_stages(events, stage.data, epoch_sec=5)
        steps = ["event_csv", "stage_sleep_filter"]
        return AnnotationResult(
            [
                stage,
                materialize_event_table(
                    events,
                    canonical_channel="ah_event_table",
                    raw_file=str(record.files["events"]),
                    raw_label="Type/Start/Duration",
                    steps=steps,
                ),
                materialize_dense_events(
                    events,
                    duration_sec=duration_sec,
                    interval_sec=1,
                    canonical_channel="ah_event",
                    raw_file=str(record.files["events"]),
                    raw_label="Type/Start/Duration",
                    steps=steps,
                ),
                materialize_anchor_events(
                    events,
                    duration_sec=duration_sec,
                    window_sec=10,
                    anchor_num=2,
                    canonical_channel="arousal_anchor",
                    raw_file=str(record.files["events"]),
                    raw_label="Type/Start/Duration",
                    steps=steps,
                ),
            ]
        )


def make_adapter(config):
    return EventAdapter()
""")
    monkeypatch.syspath_prepend(str(tmp_path))
    edf_path = write_tiny_edf(tmp_path / "night.edf", duration_sec=20)
    stage_csv = tmp_path / "stage.csv"
    pd.DataFrame(
        [
            {"Start": 0, "Duration": 5, "Type": "Wake"},
            {"Start": 5, "Duration": 5, "Type": "N1"},
            {"Start": 10, "Duration": 5, "Type": "N2"},
            {"Start": 15, "Duration": 5, "Type": "Wake"},
        ]
    ).to_csv(stage_csv, index=False)
    event_csv = tmp_path / "events.csv"
    pd.DataFrame(
        [
            {"Start": 2, "Duration": 2, "Type": "Apnea"},
            {"Start": 7, "Duration": 4, "Type": "Hypopnea"},
        ]
    ).to_csv(event_csv, index=False)
    adapter_index = tmp_path / "event_index.csv"
    pd.DataFrame(
        [
            {
                "record_id": "night1",
                "edf_path": str(edf_path),
                "stage_csv": str(stage_csv),
                "event_csv": str(event_csv),
            }
        ]
    ).to_csv(adapter_index, index=False)
    config_path = tmp_path / "hypnodata.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "center": "toy_event_center",
                "record_discovery": {"type": "custom", "adapter": "event_adapter:make_adapter"},
                "backend": {"type": "npz"},
                "adapter_options": {"index": str(adapter_index)},
                "signals": {
                    "eeg": {
                        "kind": "eeg",
                        "required": True,
                        "target_sfreq": 10,
                        "target_unit": "uV",
                        "candidates": ["EEG C3"],
                    },
                    "stage5": {"kind": "stage", "required": False, "epoch_sec": 5, "candidates": []},
                    "ah_event_table": {"kind": "event_table", "required": False, "candidates": []},
                    "ah_event": {"kind": "event_dense", "required": False, "interval_sec": 1, "candidates": []},
                    "arousal_anchor": {
                        "kind": "event_anchor",
                        "required": False,
                        "window_sec": 10,
                        "candidates": [],
                    },
                },
            }
        )
    )
    output_dir = tmp_path / "out"

    run_pipeline(load_config(config_path), output_dir=output_dir)

    with np.load(output_dir / "backends" / "npz" / "records" / "night1.npz") as npz:
        assert sorted(npz.files) == ["ah_event", "ah_event_table", "arousal_anchor", "eeg", "stage5"]
        np.testing.assert_array_equal(npz["stage5"], np.asarray([0, 1, 2, 0], dtype=np.int64))
        np.testing.assert_allclose(npz["ah_event_table"], np.asarray([[1, 7, 4]], dtype=np.float32))
        expected_dense = np.zeros(20, dtype=np.float32)
        expected_dense[7:11] = 1
        np.testing.assert_array_equal(npz["ah_event"], expected_dense)
        np.testing.assert_allclose(npz["arousal_anchor"][0], np.asarray([1, 0.7, 1, 0, 0, 0], dtype=np.float32))
        np.testing.assert_allclose(npz["arousal_anchor"][1], np.asarray([1, 0, 0.1, 0, 0, 0], dtype=np.float32))

    record_manifest = pd.read_csv(output_dir / "manifest" / "record_manifest.csv")
    assert record_manifest.loc[0, "ah_event_table_mask"] == 1
    assert record_manifest.loc[0, "ah_event_mask"] == 1
    assert record_manifest.loc[0, "arousal_anchor_mask"] == 1

    signal_manifest = pd.read_csv(output_dir / "manifest" / "signal_manifest.csv")
    table = signal_manifest[signal_manifest["canonical_channel"] == "ah_event_table"].iloc[0]
    assert table["kind"] == "event_table"
    assert table["selection_reason"] == "annotation"
    assert pd.isna(table["raw_sfreq"])
    assert pd.isna(table["target_sfreq"])
    assert table["preprocess_steps"] == "event_csv,stage_sleep_filter"
    dense = signal_manifest[signal_manifest["canonical_channel"] == "ah_event"].iloc[0]
    assert dense["raw_sfreq"] == 1.0
    assert dense["target_sfreq"] == 1.0
    assert dense["mask_column"] == "ah_event_mask"
    anchor = signal_manifest[signal_manifest["canonical_channel"] == "arousal_anchor"].iloc[0]
    assert anchor["preprocess_steps"] == "event_csv,stage_sleep_filter,anchor:10s:2"


def test_custom_adapter_materializes_builtin_ahi_output(tmp_path: Path, monkeypatch):
    adapter_module = tmp_path / "ahi_adapter.py"
    adapter_module.write_text("""
from pathlib import Path

import pandas as pd

from hypnodata.annotations import AnnotationResult, materialize_ahi_from_events, read_event_csv, read_stage_csv
from hypnodata.records import RecordTask


class AhiAdapter:
    def collect_records(self, config):
        frame = pd.read_csv(config.adapter_options["index"], low_memory=False)
        return [
            RecordTask(
                record_id=str(row["record_id"]),
                center=config.center,
                files={"stage": Path(row["stage_csv"]), "events": Path(row["event_csv"])},
                metadata={"duration": float(row["duration"]), "split": "train"},
            )
            for _, row in frame.iterrows()
        ]

    def read_annotations(self, record, config, duration_sec):
        stage = read_stage_csv(
            record.files["stage"],
            duration_sec=duration_sec,
            epoch_sec=30,
            label_column="Type",
            start_column="Start",
            duration_column="Duration",
        )
        events = read_event_csv(record.files["events"], mapping={"Apnea": 0, "Hypopnea": 1})
        ahi = materialize_ahi_from_events(
            events,
            stage.data,
            duration_sec=duration_sec,
            epoch_sec=30,
            interval_sec=1,
            raw_file=str(record.files["events"]),
            raw_label="Type/Start/Duration",
        )
        return AnnotationResult([stage, ahi])


def make_adapter(config):
    return AhiAdapter()
""")
    monkeypatch.syspath_prepend(str(tmp_path))
    stage_csv = tmp_path / "stage.csv"
    pd.DataFrame(
        [
            {"Start": 0, "Duration": 30, "Type": "Wake"},
            {"Start": 30, "Duration": 30, "Type": "N1"},
            {"Start": 60, "Duration": 30, "Type": "N2"},
            {"Start": 90, "Duration": 30, "Type": "Wake"},
        ]
    ).to_csv(stage_csv, index=False)
    event_csv = tmp_path / "events.csv"
    pd.DataFrame(
        [
            {"Start": 35, "Duration": 9, "Type": "Apnea"},
            {"Start": 35, "Duration": 10, "Type": "Apnea"},
            {"Start": 5, "Duration": 20, "Type": "Hypopnea"},
            {"Start": 85, "Duration": 20, "Type": "Hypopnea"},
        ]
    ).to_csv(event_csv, index=False)
    adapter_index = tmp_path / "ahi_index.csv"
    pd.DataFrame(
        [{"record_id": "night1", "stage_csv": str(stage_csv), "event_csv": str(event_csv), "duration": 120}]
    ).to_csv(adapter_index, index=False)
    config_path = tmp_path / "hypnodata_ahi.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "center": "toy_ahi_center",
                "record_discovery": {"type": "custom", "adapter": "ahi_adapter:make_adapter"},
                "backend": {"type": "npz"},
                "adapter_options": {"index": str(adapter_index)},
                "signals": {
                    "stage5": {"kind": "stage", "required": True, "epoch_sec": 30, "candidates": []},
                    "ahi": {"kind": "ahi", "required": True, "interval_sec": 1, "candidates": []},
                },
            }
        )
    )
    output_dir = tmp_path / "out_ahi"

    run_pipeline(load_config(config_path), output_dir=output_dir)

    npz_path = output_dir / "backends" / "npz" / "records" / "night1.npz"
    with np.load(npz_path) as npz:
        assert sorted(npz.files) == ["ah_event", "ahi", "stage5", "tst"]
        expected = np.zeros(120, dtype=np.float32)
        expected[35:45] = 1
        expected[85:105] = 1
        np.testing.assert_array_equal(npz["ah_event"], expected)
        np.testing.assert_array_equal(npz["stage5"], np.asarray([0, 1, 2, 0], dtype=np.int64))
        ahi_value, tst_value = load_builtin_ahi_metadata(npz)
    assert ahi_value == pytest.approx(120.0)
    assert tst_value == pytest.approx(60 / 3600)

    record_manifest = pd.read_csv(output_dir / "manifest" / "record_manifest.csv")
    assert record_manifest.loc[0, "stage_mask"] == 1
    assert record_manifest.loc[0, "ah_event_mask"] == 1
    assert record_manifest.loc[0, "duration"] == 120

    signal_manifest = pd.read_csv(output_dir / "manifest" / "signal_manifest.csv")
    ahi = signal_manifest[signal_manifest["canonical_channel"] == "ahi"].iloc[0]
    assert ahi["kind"] == "ahi"
    assert ahi["output_key"] == "ah_event"
    assert ahi["mask_column"] == "ah_event_mask"
    assert ahi["target_sfreq"] == 1.0
    assert "ahi_from_events" in ahi["preprocess_steps"]


def test_pipeline_writes_annotation_only_record_with_metadata_duration(tmp_path: Path, monkeypatch):
    adapter_ref = _write_annotation_only_adapter(tmp_path, monkeypatch)
    stage_csv = tmp_path / "stage.csv"
    pd.DataFrame(
        [
            {"Start": 0, "Duration": 5, "Type": "Wake"},
            {"Start": 5, "Duration": 5, "Type": "N1"},
        ]
    ).to_csv(stage_csv, index=False)
    config_path = tmp_path / "annotation_only.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "center": "annotation_center",
                "record_discovery": {"type": "custom", "adapter": adapter_ref},
                "backend": {"type": "npz"},
                "adapter_options": {
                    "duration": 10,
                    "epoch_sec": 5,
                    "stage_csv": str(stage_csv),
                },
                "signals": {
                    "stage5": {
                        "kind": "stage",
                        "required": True,
                        "epoch_sec": 5,
                        "candidates": [],
                    },
                },
            }
        )
    )
    output_dir = tmp_path / "annotation_only_out"

    run_pipeline(load_config(config_path), output_dir=output_dir)

    with np.load(output_dir / "backends" / "npz" / "records" / "night1.npz") as npz:
        assert sorted(npz.files) == ["stage5"]
        np.testing.assert_array_equal(npz["stage5"], np.asarray([0, 1], dtype=np.int64))

    record_manifest = pd.read_csv(output_dir / "manifest" / "record_manifest.csv")
    assert record_manifest.loc[0, "duration"] == 10.0
    assert record_manifest.loc[0, "stage_mask"] == 1
    signal_manifest = pd.read_csv(output_dir / "manifest" / "signal_manifest.csv")
    stage = signal_manifest[signal_manifest["canonical_channel"] == "stage5"].iloc[0]
    assert stage["selection_reason"] == "annotation"


def test_pipeline_rejects_annotation_only_record_without_metadata_duration(tmp_path: Path, monkeypatch):
    adapter_ref = _write_annotation_only_adapter(tmp_path, monkeypatch)
    stage_csv = tmp_path / "stage.csv"
    pd.DataFrame([{"Start": 0, "Duration": 5, "Type": "Wake"}]).to_csv(stage_csv, index=False)
    config_path = tmp_path / "annotation_only_missing_duration.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "center": "annotation_center",
                "record_discovery": {"type": "custom", "adapter": adapter_ref},
                "backend": {"type": "npz"},
                "adapter_options": {
                    "include_duration": False,
                    "epoch_sec": 5,
                    "stage_csv": str(stage_csv),
                },
                "signals": {
                    "stage5": {
                        "kind": "stage",
                        "required": True,
                        "epoch_sec": 5,
                        "candidates": [],
                    },
                },
            }
        )
    )
    output_dir = tmp_path / "annotation_only_missing_duration_out"

    run_pipeline(load_config(config_path), output_dir=output_dir)

    failures = pd.read_csv(output_dir / "manifest" / "failures.csv")
    assert "record.metadata['duration'] is required" in failures.loc[0, "message"]


def test_pipeline_rejects_empty_optional_raw_without_annotations(tmp_path: Path, monkeypatch):
    adapter_ref = _write_annotation_only_adapter(tmp_path, monkeypatch)
    config_path = tmp_path / "empty_optional_raw.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "center": "annotation_center",
                "record_discovery": {"type": "custom", "adapter": adapter_ref},
                "backend": {"type": "npz"},
                "adapter_options": {
                    "duration": 10,
                    "mode": "empty",
                },
                "signals": {
                    "eeg": {
                        "kind": "eeg",
                        "required": False,
                        "target_sfreq": 10,
                        "target_unit": "uV",
                        "candidates": ["EEG C3"],
                    },
                },
            }
        )
    )
    output_dir = tmp_path / "empty_optional_raw_out"

    run_pipeline(load_config(config_path), output_dir=output_dir)

    failures = pd.read_csv(output_dir / "manifest" / "failures.csv")
    assert failures.loc[0, "message"] == "No available signals to write."


@pytest.mark.parametrize(
    ("mode", "match"),
    [
        ("undeclared", "must be declared"),
        ("duplicate", "Duplicate annotation channel"),
        ("raw_duplicate", "duplicates a raw signal output"),
        ("zero_anchor", "must have 3 columns per anchor"),
        ("long_dense", "does not match record duration"),
        ("short_dense", "does not match record duration"),
        ("event_table_beyond_duration", "exceeds record duration"),
    ],
)
def test_pipeline_rejects_invalid_annotation_channels(tmp_path: Path, monkeypatch, mode: str, match: str):
    adapter_module = tmp_path / "bad_annotation_adapter.py"
    adapter_module.write_text("""
from pathlib import Path

import numpy as np
import pandas as pd

from hypnodata.annotations import AnnotationResult, AnnotationSignal, materialize_dense_events, materialize_event_table
from hypnodata.records import RecordTask


class BadAnnotationAdapter:
    def collect_records(self, config):
        frame = pd.read_csv(config.adapter_options["index"], low_memory=False)
        return [
            RecordTask(
                record_id=str(row["record_id"]),
                center=config.center,
                files={"edf": Path(row["edf_path"])},
            )
            for _, row in frame.iterrows()
        ]

    def read_annotations(self, record, config, duration_sec):
        rows = np.asarray([[0, 1, 2]], dtype=np.float32)
        if config.adapter_options["mode"] == "undeclared":
            return AnnotationResult(
                [
                    materialize_dense_events(
                        rows,
                        duration_sec=duration_sec,
                        interval_sec=1,
                        canonical_channel="missing_event",
                    )
                ]
            )
        if config.adapter_options["mode"] == "duplicate":
            first = materialize_dense_events(
                rows,
                duration_sec=duration_sec,
                interval_sec=1,
                canonical_channel="ah_event",
            )
            second = materialize_dense_events(
                rows,
                duration_sec=duration_sec,
                interval_sec=1,
                canonical_channel="ah_event",
            )
            return AnnotationResult([first, second])
        if config.adapter_options["mode"] == "zero_anchor":
            return AnnotationResult(
                [
                    AnnotationSignal(
                        canonical_channel="arousal_anchor",
                        data=np.zeros((10, 0), dtype=np.float32),
                        sfreq=0.1,
                        raw_file="events.csv",
                        raw_label="events",
                        materialization="event_anchor",
                    )
                ]
            )
        if config.adapter_options["mode"] == "long_dense":
            return AnnotationResult(
                [
                    AnnotationSignal(
                        canonical_channel="ah_event",
                        data=np.zeros(11, dtype=np.float32),
                        sfreq=1.0,
                        raw_file="events.csv",
                        raw_label="events",
                        materialization="event_dense",
                    )
                ]
            )
        if config.adapter_options["mode"] == "short_dense":
            return AnnotationResult(
                [
                    AnnotationSignal(
                        canonical_channel="ah_event",
                        data=np.zeros(9, dtype=np.float32),
                        sfreq=1.0,
                        raw_file="events.csv",
                        raw_label="events",
                        materialization="event_dense",
                    )
                ]
            )
        if config.adapter_options["mode"] == "event_table_beyond_duration":
            return AnnotationResult(
                [
                    materialize_event_table(
                        np.asarray([[0, duration_sec - 1, 2]], dtype=np.float32),
                        canonical_channel="ah_event_table",
                    )
                ]
            )
        return AnnotationResult(
            [
                materialize_dense_events(
                    rows,
                    duration_sec=duration_sec,
                    interval_sec=1,
                    canonical_channel="eeg",
                )
            ]
        )


def make_adapter(config):
    return BadAnnotationAdapter()
""")
    monkeypatch.syspath_prepend(str(tmp_path))
    edf_path = write_tiny_edf(tmp_path / "night.edf", duration_sec=10)
    adapter_index = tmp_path / "bad_index.csv"
    pd.DataFrame([{"record_id": "night1", "edf_path": str(edf_path)}]).to_csv(adapter_index, index=False)
    config_path = tmp_path / f"hypnodata_{mode}.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "center": "toy_bad_annotation",
                "record_discovery": {"type": "custom", "adapter": "bad_annotation_adapter:make_adapter"},
                "backend": {"type": "npz"},
                "adapter_options": {"index": str(adapter_index), "mode": mode},
                "signals": {
                    "eeg": {
                        "kind": "eeg",
                        "required": True,
                        "target_sfreq": 10,
                        "target_unit": "uV",
                        "candidates": ["EEG C3"],
                    },
                    "ah_event": {"kind": "event_dense", "required": False, "interval_sec": 1, "candidates": []},
                    "ah_event_table": {"kind": "event_table", "required": False, "candidates": []},
                    "arousal_anchor": {
                        "kind": "event_anchor",
                        "required": False,
                        "window_sec": 10,
                        "candidates": [],
                    },
                },
            }
        )
    )
    output_dir = tmp_path / f"out_{mode}"

    run_pipeline(load_config(config_path), output_dir=output_dir)

    failures = pd.read_csv(output_dir / "manifest" / "failures.csv")
    assert match in failures.loc[0, "message"]
