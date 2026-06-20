from __future__ import annotations

from pathlib import Path

from agent_tool_test_helpers import config_payload, survival_config_payload, write_survival_sidecars, write_yaml
import pandas as pd

from agent_tools.index_csv import index_summary


def test_index_summary_counts_splits_masks_and_labels(tmp_path: Path):
    index = tmp_path / "index.csv"
    pd.DataFrame(
        [
            {
                "path": "a.npz",
                "split": "train",
                "duration": 60,
                "source": "s1",
                "age": 50,
                "custom_label": 1,
                "wake_frac": 0.2,
                "sleep_hours": 1.0,
                "ppg_mask": 1,
            },
            {
                "path": "b.npz",
                "split": "val",
                "duration": 90,
                "source": "s1",
                "age": None,
                "custom_label": 0,
                "wake_frac": 0.3,
                "sleep_hours": 1.5,
                "ppg_mask": 0,
            },
            {
                "path": "c.npz",
                "split": "test",
                "duration": 120,
                "source": "s2",
                "age": 60,
                "custom_label": 1,
                "wake_frac": 0.8,
                "sleep_hours": 0.8,
                "ppg_mask": 1,
            },
        ]
    ).to_csv(index, index=False)
    config = write_yaml(tmp_path / "config.yaml", config_payload(index))

    summary = index_summary([index], config=config, label_name="custom_label")

    assert summary["rows"] == 3
    assert summary["split_counts"]["train"] == 1
    assert summary["label_presence"]["age"]["non_null"] == 2
    assert summary["mask_columns"]["ppg_mask"]["true_count"] == 2
    assert summary["channel_coverage_from_config"]["ppg"]["available_rows"] == 2
    assert summary["label_presence"]["custom_label"]["non_null"] == 3
    assert "custom_label" in summary["split_source_label_counts"]
    assert summary["channel_mask_coverage_by_split_source"]["ppg_mask"]
    assert "wake_frac" in summary["numeric_shift_metrics"]
    assert "sleep_hours" in summary["numeric_shift_metrics"]


def test_index_summary_reports_survival_key_column(tmp_path: Path):
    index = tmp_path / "index.csv"
    pd.DataFrame(
        [
            {"path": "a.npz", "split": "train", "duration": 60, "eid": "001", "ppg_mask": 1},
            {"path": "b.npz", "split": "train", "duration": 60, "eid": "001", "ppg_mask": 1},
            {"path": "c.npz", "split": "val", "duration": 60, "eid": "NA", "ppg_mask": 1},
        ]
    ).to_csv(index, index=False)
    sidecars = write_survival_sidecars(tmp_path)
    Path(sidecars["event_time_index"]).write_text("eid,d1,d2\n001,10,20\nNA,30,40\n")
    Path(sidecars["is_event_index"]).write_text("eid,d1,d2\n001,1,0\nNA,0,1\n")
    Path(sidecars["has_label_index"]).write_text("eid,d1,d2\n001,1,1\nNA,1,1\n")
    config = write_yaml(
        tmp_path / "survival.yaml",
        survival_config_payload(index, sidecars),
    )

    summary = index_summary([index], config=config)

    assert summary["survival_key"] == {
        "key_column": "eid",
        "exists": True,
        "non_null_rows": 3,
        "missing_rows": 0,
        "unique_keys": 2,
        "sidecar_key_count": 2,
        "missing_from_sidecars": 0,
        "missing_from_sidecars_examples": [],
    }
    assert "Index CSV contains empty survival key values in column: eid" not in summary["blocking_issues"]


def test_index_summary_blocks_missing_survival_key_column(tmp_path: Path):
    index = tmp_path / "index.csv"
    pd.DataFrame([{"path": "a.npz", "split": "train", "duration": 60, "ppg_mask": 1}]).to_csv(index, index=False)
    config = write_yaml(
        tmp_path / "survival.yaml",
        survival_config_payload(index, write_survival_sidecars(tmp_path)),
    )

    summary = index_summary([index], config=config)

    assert summary["survival_key"]["exists"] is False
    assert "Index CSV missing required survival key column: eid" in summary["blocking_issues"]


def test_index_summary_blocks_survival_keys_missing_from_sidecars(tmp_path: Path):
    index = tmp_path / "index.csv"
    pd.DataFrame(
        [
            {"path": "a.npz", "split": "train", "duration": 60, "eid": "001", "ppg_mask": 1},
            {"path": "b.npz", "split": "val", "duration": 60, "eid": "003", "ppg_mask": 1},
        ]
    ).to_csv(index, index=False)
    config = write_yaml(
        tmp_path / "survival.yaml",
        survival_config_payload(index, write_survival_sidecars(tmp_path)),
    )

    summary = index_summary([index], config=config)

    assert summary["survival_key"]["missing_from_sidecars"] == 1
    assert summary["survival_key"]["missing_from_sidecars_examples"] == ["003"]
    assert (
        "Index CSV contains survival key values missing from sidecars in column eid: 1 missing (examples: 003)"
        in summary["blocking_issues"]
    )


def test_index_summary_blocks_empty_survival_keys(tmp_path: Path):
    index = tmp_path / "index.csv"
    index.write_text("path,split,duration,eid,ppg_mask\na.npz,train,60,001,1\nb.npz,val,60,,1\n")
    config = write_yaml(
        tmp_path / "survival.yaml",
        survival_config_payload(index, write_survival_sidecars(tmp_path)),
    )

    summary = index_summary([index], config=config)

    assert summary["survival_key"]["exists"] is True
    assert summary["survival_key"]["missing_rows"] == 1
    assert "Index CSV contains empty survival key values in column: eid" in summary["blocking_issues"]
