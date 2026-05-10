from __future__ import annotations

from pathlib import Path

import numpy as np

from data.kaldi_io import KaldiChannelSpec, KaldiReaderPool
from sleep2wave.data.kaldi_io import KaldiWaveChannelSpec, KaldiWaveReaderPool


def test_reader_pool_opens_sorted_scp_reader(tmp_path: Path, monkeypatch) -> None:
    opened_specs = []

    class FakeReader:
        def __contains__(self, key: str) -> bool:
            return key == "sample-a"

        def __getitem__(self, key: str) -> np.ndarray:
            return np.ones((2, 3), dtype=np.float32)

    class FakeKaldiNativeIO:
        def RandomAccessFloatMatrixReader(self, spec: str) -> FakeReader:
            opened_specs.append(spec)
            return FakeReader()

    monkeypatch.setattr(KaldiReaderPool, "_import_kaldi_native_io", lambda self: FakeKaldiNativeIO())

    pool = KaldiReaderPool(
        tmp_path,
        {"eeg": KaldiChannelSpec(name="eeg", input_dim=3, scp_path=Path("channels/eeg.scp"))},
    )

    pool.read_matrix("eeg", "sample-a")

    assert opened_specs == [f"s,scp:{tmp_path / 'channels' / 'eeg.scp'}"]


def test_sleep2wave_reader_pool_opens_sorted_scp_reader(tmp_path: Path, monkeypatch) -> None:
    opened_specs = []

    class FakeReader:
        def __contains__(self, key: str) -> bool:
            return key == "sample-a"

        def __getitem__(self, key: str) -> np.ndarray:
            return np.ones((2, 3840), dtype=np.float32)

    class FakeKaldiNativeIO:
        def RandomAccessFloatMatrixReader(self, spec: str) -> FakeReader:
            opened_specs.append(spec)
            return FakeReader()

    monkeypatch.setattr(KaldiWaveReaderPool, "_import_kaldi_native_io", lambda self: FakeKaldiNativeIO())

    pool = KaldiWaveReaderPool(
        tmp_path,
        {
            "eeg": KaldiWaveChannelSpec(
                name="eeg",
                frames_per_epoch=3840,
                scp_path=Path("channels/train/eeg.scp"),
            )
        },
    )

    pool.read_matrix("eeg", "sample-a")

    assert opened_specs == [f"s,scp:{tmp_path / 'channels' / 'train' / 'eeg.scp'}"]
