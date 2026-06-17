import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

kaldi_native_io = pytest.importorskip("kaldi_native_io")

from preprocess.convert_npz_to_kaldi import convert, parse_args
from tests.hypnodata_test_helpers import run_tiny_hypnodata


def test_npz_to_kaldi_converter_consumes_hypnodata_outputs(tmp_path: Path):
    output_dir = run_tiny_hypnodata(tmp_path)
    config_path = tmp_path / "kaldi_config.yaml"
    config_path.write_text(yaml.safe_dump({"model": {"channels": [{"name": "eeg", "input_dim": 30}]}}))
    kaldi_dir = tmp_path / "kaldi"

    convert(
        parse_args(
            [
                "--index",
                str(output_dir / "manifest" / "record_manifest.csv"),
                "--config",
                str(config_path),
                "--output-dir",
                str(kaldi_dir),
                "--max-tokens",
                "2",
                "--stride-tokens",
                "2",
                "--token-sec",
                "30",
                "--channels-from-config",
                "--source-field",
                "source",
                "--min-channels",
                "1",
                "--num-workers",
                "1",
                "--no-compress-ark",
            ]
        )
    )

    manifest = pd.read_csv(kaldi_dir / "manifests" / "test.csv", low_memory=False)
    assert manifest["sample_key"].tolist() == ["toy_source_ses1_000000_000002"]
    assert manifest.loc[0, "record_id"] == "night1"
    assert json.loads(manifest.loc[0, "available_channels"]) == ["eeg"]

    key = manifest.loc[0, "sample_key"]
    with kaldi_native_io.RandomAccessFloatMatrixReader(f"scp:{kaldi_dir / 'channels' / 'test' / 'eeg.scp'}") as reader:
        matrix = np.asarray(reader[key], dtype=np.float32)
    assert matrix.shape == (2, 30)
