from __future__ import annotations

import numpy as np
import pytest

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
    assert [sample.payload["available_channels"] for sample in filtered] == [["eeg"], ["eeg"], ["eeg"]]


def test_filter_valid_sample_indices_records_available_channels_in_strict_mode(monkeypatch):
    monkeypatch.setattr("data.utils.load_npz", lambda path: _FakeNpz({"eeg": np.arange(8, dtype=np.float32)}))

    data = [SampleIndex(id=0, path="same.npz", start=0, end=2)]
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
    assert filtered[0].payload["available_channels"] == ["eeg"]


def test_filter_valid_sample_indices_records_configured_channels_for_alias_keys(monkeypatch):
    monkeypatch.setattr(
        "data.utils.load_npz",
        lambda path: _FakeNpz(
            {
                "psg_breath": np.arange(8, dtype=np.float32),
                "bcg_heartbeat": np.arange(8, dtype=np.float32),
            }
        ),
    )

    data = [SampleIndex(id=0, path="same.npz", start=0, end=2)]
    extractors = {
        "breath": default_extractor("breath", 4, source_names=["psg_breath"]),
        "heartbeat": default_extractor("heartbeat", 4, source_names=["bcg_heartbeat"]),
    }
    tokenizers = {
        "breath": default_tokenizer(4),
        "heartbeat": default_tokenizer(4),
    }

    filtered = filter_valid_sample_indices(
        data,
        extractors,
        tokenizers,
        allow_missing_channels=True,
        channel_names=["breath", "heartbeat"],
        min_channels=2,
        max_workers=1,
        channel_aliases={"breath": ["psg_breath"], "heartbeat": ["bcg_heartbeat"]},
    )

    assert filtered == data
    assert filtered[0].payload["available_channels"] == ["breath", "heartbeat"]


def test_default_extractor_falls_back_to_channel_alias():
    npz = _FakeNpz({"psg_breath": np.ones(8, dtype=np.float32)})

    extracted = default_extractor("breath", 4, source_names=["psg_breath"])(npz, 0, 2)

    assert extracted.tolist() == [1.0] * 8


def test_default_extractor_does_not_use_undeclared_channel_alias():
    npz = _FakeNpz({"psg_breath": np.ones(8, dtype=np.float32)})

    with pytest.raises(KeyError):
        default_extractor("breath", 4)(npz, 0, 2)


def test_default_extractor_prefers_canonical_key_before_alias():
    npz = _FakeNpz(
        {
            "breath": np.zeros(8, dtype=np.float32),
            "psg_breath": np.ones(8, dtype=np.float32),
        }
    )

    extracted = default_extractor("breath", 4, source_names=["psg_breath"])(npz, 0, 2)

    assert extracted.tolist() == [0.0] * 8


def test_variant_default_extractors_fall_back_to_channel_alias():
    from sleep2expert.data.utils import default_extractor as expert_default_extractor
    from sleep2vec2.data.utils import default_extractor as sleep2vec2_default_extractor

    npz = _FakeNpz({"bcg_heartbeat": np.ones(8, dtype=np.float32)})

    assert sleep2vec2_default_extractor("heartbeat", 4, source_names=["bcg_heartbeat"])(npz, 0, 2).tolist() == [1.0] * 8
    assert expert_default_extractor("heartbeat", 4, source_names=["bcg_heartbeat"])(npz, 0, 2).tolist() == [1.0] * 8


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
