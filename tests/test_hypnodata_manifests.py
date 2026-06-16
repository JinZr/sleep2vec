from pathlib import Path

import pandas as pd

from hypnodata.config import BackendConfig, DiscoveryConfig, HypnodataConfig, SignalSpec
from hypnodata.manifests import mask_column_for_channel, output_key_for_channel, write_manifests


def _config(tmp_path: Path) -> HypnodataConfig:
    return HypnodataConfig(
        path=tmp_path / "config.yaml",
        center="toy",
        record_discovery=DiscoveryConfig(type="glob", root=tmp_path),
        backend=BackendConfig(type="npz"),
        signals={
            "eeg": SignalSpec(
                name="eeg",
                kind="eeg",
                required=True,
                target_sfreq=10,
                target_unit="uV",
                candidates=["EEG"],
            ),
            "stage5": SignalSpec(
                name="stage5",
                kind="stage",
                required=False,
                target_sfreq=None,
                target_unit=None,
                candidates=["Stage"],
            ),
        },
    )


def test_mask_column_for_channel_matches_downstream_contract():
    assert mask_column_for_channel("eeg") == "eeg_mask"
    assert mask_column_for_channel("stage5") == "stage_mask"
    assert mask_column_for_channel("ahi") == "ah_event_mask"
    assert mask_column_for_channel("ah_event") == "ah_event_mask"


def test_output_key_for_channel_matches_downstream_contract():
    assert output_key_for_channel("eeg") == "eeg"
    assert output_key_for_channel("ahi") == "ah_event"
    assert output_key_for_channel("ah_event") == "ah_event"


def test_write_manifests_preserves_required_columns(tmp_path: Path):
    output_dir = tmp_path / "out"
    write_manifests(
        output_dir,
        _config(tmp_path),
        record_rows=[
            {
                "record_id": "r1",
                "center": "toy",
                "source": "src",
                "subject_id": "sub",
                "session_id": "ses",
                "split": "train",
                "path": "backends/npz/records/r1.npz",
                "duration": 10.0,
                "backend": "npz",
                "qc_status": "ok",
                "eeg_mask": 1,
                "stage_mask": 0,
            }
        ],
        signal_rows=[],
        qc_rows=[],
        failure_rows=[],
        dry_run=False,
    )

    record_manifest = pd.read_csv(output_dir / "manifest" / "record_manifest.csv")
    assert "duration" in record_manifest.columns
    assert "duration_sec" not in record_manifest.columns
    assert record_manifest.loc[0, "stage_mask"] == 0

    signal_manifest = pd.read_csv(output_dir / "manifest" / "signal_manifest.csv")
    assert list(signal_manifest.columns) == [
        "record_id",
        "center",
        "canonical_channel",
        "kind",
        "available",
        "required",
        "raw_file",
        "raw_label",
        "selection_reason",
        "raw_sfreq",
        "target_sfreq",
        "raw_unit",
        "target_unit",
        "scale_applied",
        "polarity_applied",
        "raw_n_samples",
        "output_n_samples",
        "preprocess_steps",
        "qc_status",
        "output_key",
        "mask_column",
    ]
