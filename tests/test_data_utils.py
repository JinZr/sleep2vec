from __future__ import annotations

import numpy as np

from data.default_dataset import SampleIndex
from data.utils import default_extractor, default_tokenizer, filter_valid_sample_indices


class _FakeNpz(dict):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_filter_valid_sample_indices_opens_each_path_once(monkeypatch):
    open_counts: dict[str, int] = {}

    def fake_load_npz(path: str):
        open_counts[path] = open_counts.get(path, 0) + 1
        return _FakeNpz({"eeg": np.arange(16, dtype=np.float32)})

    monkeypatch.setattr("data.utils.load_npz", fake_load_npz)

    data = [
        SampleIndex(id=0, path="same.npz", start=0, end=2),
        SampleIndex(id=1, path="same.npz", start=2, end=4),
        SampleIndex(id=2, path="other.npz", start=0, end=2),
    ]
    extractors = {"eeg": default_extractor("eeg", 4)}
    tokenizers = {"eeg": default_tokenizer(4)}

    filtered = filter_valid_sample_indices(
        data,
        extractors,
        tokenizers,
        allow_missing_channels=False,
        channel_names=["eeg"],
        min_channels=1,
        max_workers=1,
    )

    assert filtered == data
    assert open_counts == {"same.npz": 1, "other.npz": 1}
