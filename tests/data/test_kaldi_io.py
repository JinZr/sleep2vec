from __future__ import annotations

from pathlib import Path
import pickle

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

kaldi_native_io = pytest.importorskip("kaldi_native_io")

from data.kaldi_io import KaldiChannelSpec, KaldiReaderPool


def _write_channel(root: Path, channel: str, matrices: dict[str, np.ndarray]) -> Path:
    channels_dir = root / "channels"
    channels_dir.mkdir(parents=True, exist_ok=True)
    ark_path = channels_dir / f"{channel}.ark"
    scp_path = channels_dir / f"{channel}.scp"
    with kaldi_native_io.FloatMatrixWriter(f"ark,scp:{ark_path},{scp_path}") as writer:
        for key, matrix in matrices.items():
            writer.write(key, np.asarray(matrix, dtype=np.float32))
    return scp_path


def _write_compressed_channel(root: Path, channel: str, matrices: dict[str, np.ndarray]) -> Path:
    channels_dir = root / "channels"
    channels_dir.mkdir(parents=True, exist_ok=True)
    ark_path = channels_dir / f"{channel}.ark"
    scp_path = channels_dir / f"{channel}.scp"
    method = kaldi_native_io.CompressionMethod.kTwoByteAuto
    with kaldi_native_io.CompressedMatrixWriter(f"ark,scp:{ark_path},{scp_path}") as writer:
        for key, matrix in matrices.items():
            writer.write(key, np.asarray(matrix, dtype=np.float32), method=method)
    return scp_path


def _pool(root: Path, channel: str, input_dim: int, scp_path: Path) -> KaldiReaderPool:
    return KaldiReaderPool(
        root,
        {channel: KaldiChannelSpec(name=channel, input_dim=input_dim, scp_path=scp_path.relative_to(root))},
    )


class _KaldiPoolDataset(Dataset):
    def __init__(self, pool: KaldiReaderPool, keys: list[str]) -> None:
        self.pool = pool
        self.keys = keys

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.from_numpy(self.pool.read_matrix("eeg", self.keys[index]))


def test_reader_pool_reads_matrix_copy(tmp_path: Path) -> None:
    matrix = np.arange(6, dtype=np.float32).reshape(2, 3)
    scp_path = _write_channel(tmp_path, "eeg", {"sample-a": matrix})
    pool = _pool(tmp_path, "eeg", 3, scp_path)

    actual = pool.read_matrix("eeg", "sample-a")

    np.testing.assert_array_equal(actual, matrix)
    assert actual.dtype == np.float32
    actual[0, 0] = -999.0
    np.testing.assert_array_equal(pool.read_matrix("eeg", "sample-a"), matrix)


def test_reader_pool_reads_compressed_matrix(tmp_path: Path) -> None:
    matrix = np.linspace(-1.0, 1.0, num=12, dtype=np.float32).reshape(4, 3)
    scp_path = _write_compressed_channel(tmp_path, "eeg", {"sample-a": matrix})
    pool = _pool(tmp_path, "eeg", 3, scp_path)

    actual = pool.read_matrix("eeg", "sample-a")

    np.testing.assert_allclose(actual, matrix, rtol=1e-3, atol=1e-3)


def test_reader_pool_rejects_unknown_channel(tmp_path: Path) -> None:
    scp_path = _write_channel(tmp_path, "eeg", {"sample-a": np.ones((2, 3), dtype=np.float32)})
    pool = _pool(tmp_path, "eeg", 3, scp_path)

    with pytest.raises(KeyError, match="Unknown Kaldi channel"):
        pool.read_matrix("ppg", "sample-a")


def test_reader_pool_rejects_missing_key(tmp_path: Path) -> None:
    scp_path = _write_channel(tmp_path, "eeg", {"sample-a": np.ones((2, 3), dtype=np.float32)})
    pool = _pool(tmp_path, "eeg", 3, scp_path)

    with pytest.raises(KeyError, match="Missing Kaldi key"):
        pool.read_matrix("eeg", "sample-b")


def test_reader_pool_rejects_input_dim_mismatch(tmp_path: Path) -> None:
    scp_path = _write_channel(tmp_path, "eeg", {"sample-a": np.ones((2, 4), dtype=np.float32)})
    pool = _pool(tmp_path, "eeg", 3, scp_path)

    with pytest.raises(ValueError, match="expected input_dim=3"):
        pool.read_matrix("eeg", "sample-a")


def test_reader_pool_drops_handles_during_pickle(tmp_path: Path) -> None:
    matrix = np.arange(6, dtype=np.float32).reshape(2, 3)
    scp_path = _write_channel(tmp_path, "eeg", {"sample-a": matrix})
    pool = _pool(tmp_path, "eeg", 3, scp_path)

    np.testing.assert_array_equal(pool.read_matrix("eeg", "sample-a"), matrix)
    assert pool._readers

    restored = pickle.loads(pickle.dumps(pool))

    assert restored._readers == {}
    np.testing.assert_array_equal(restored.read_matrix("eeg", "sample-a"), matrix)


def test_reader_pool_works_in_dataloader_workers(tmp_path: Path) -> None:
    matrices = {
        "sample-a": np.full((2, 3), 1.0, dtype=np.float32),
        "sample-b": np.full((2, 3), 2.0, dtype=np.float32),
        "sample-c": np.full((2, 3), 3.0, dtype=np.float32),
        "sample-d": np.full((2, 3), 4.0, dtype=np.float32),
    }
    scp_path = _write_channel(tmp_path, "eeg", matrices)
    pool = _pool(tmp_path, "eeg", 3, scp_path)
    pool.read_matrix("eeg", "sample-a")

    loader = DataLoader(_KaldiPoolDataset(pool, list(matrices)), batch_size=1, shuffle=False, num_workers=2)

    batches = [batch.squeeze(0).numpy() for batch in loader]

    for actual, expected in zip(batches, matrices.values(), strict=True):
        np.testing.assert_array_equal(actual, expected)
