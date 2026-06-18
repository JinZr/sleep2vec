import json
from pathlib import Path

from tests.hypnodata_test_helpers import run_tiny_hypnodata


def test_backend_manifest_describes_npz_product_without_schema_version(tmp_path: Path):
    output_dir = run_tiny_hypnodata(tmp_path)

    manifest = json.loads((output_dir / "manifest" / "backend_manifest.json").read_text())

    assert "schema_version" not in manifest
    assert manifest["enabled_backends"] == ["npz"]
    assert manifest["record_manifest"] == "manifest/record_manifest.csv"
    assert manifest["signal_manifest"] == "manifest/signal_manifest.csv"
    assert manifest["npz_records_dir"] == "backends/npz/records"
    assert manifest["channels"]["eeg"] == {
        "kind": "eeg",
        "target_sfreq": 1.0,
        "target_unit": "uV",
        "mask_column": "eeg_mask",
        "output_key": "eeg",
    }
