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


def test_filter_valid_sample_indices_drops_builtin_ahi_samples_without_any_valid_labels(monkeypatch):
    npz_by_path = {
        "invalid.npz": _FakeNpz(
            {
                "ah_event": np.full(60, -1.0, dtype=np.float32),
                "ahi": np.asarray(3.0, dtype=np.float32),
                "tst": np.asarray(5.0, dtype=np.float32),
            }
        ),
        "valid.npz": _FakeNpz(
            {
                "ah_event": np.concatenate(
                    [np.full(30, -1.0, dtype=np.float32), np.zeros(30, dtype=np.float32)],
                    axis=0,
                ),
                "ahi": np.asarray(1.0, dtype=np.float32),
                "tst": np.asarray(6.0, dtype=np.float32),
            }
        ),
    }

    monkeypatch.setattr("data.utils.load_npz", lambda path: npz_by_path[path])

    data = [
        SampleIndex(id=0, path="invalid.npz", start=0, end=2, metadata={}),
        SampleIndex(id=1, path="valid.npz", start=0, end=2, metadata={}),
    ]
    extractors = {"ahi": default_extractor("ahi", 30, source_name="ah_event")}
    tokenizers = {"ahi": default_tokenizer(30)}

    filtered = filter_valid_sample_indices(
        data,
        extractors,
        tokenizers,
        allow_missing_channels=False,
        channel_names=["ahi"],
        min_channels=1,
        max_workers=1,
    )

    assert [sample.id for sample in filtered] == [1]
    assert filtered[0].metadata["ahi"] == 1.0
    assert filtered[0].metadata["tst"] == 6.0
