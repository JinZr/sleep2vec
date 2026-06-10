from __future__ import annotations

from pathlib import Path

from agent_tool_test_helpers import config_payload, write_yaml
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
