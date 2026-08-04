import importlib
from pathlib import Path

import numpy as np
import pytest
import torch

DATA_PACKAGES = ("data", "sleep2vec2.data", "sleep2expert.data")


def _write_arousal_npz(path: Path, metadata_keys: tuple[str, ...]) -> None:
    payload = {
        "arousal_event": np.zeros((60, 4), dtype=np.float32),
        **{key: np.asarray(1.0, dtype=np.float32) for key in metadata_keys},
    }
    payload["tst"] = np.asarray(3.0, dtype=np.float32)
    np.savez(path, **payload)


@pytest.mark.parametrize("package", DATA_PACKAGES)
@pytest.mark.parametrize("allow_missing_channels", [False, True])
def test_arousal_filter_validates_full_raster_once_per_recording(
    package: str,
    allow_missing_channels: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    utils_module = importlib.import_module(f"{package}.utils")
    dataset_module = importlib.import_module(f"{package}.default_dataset")
    path = tmp_path / f"{package.replace('.', '-')}.npz"
    _write_arousal_npz(path, utils_module.AROUSAL_METADATA_KEYS)
    samples = [
        dataset_module.SampleIndex(id="window-0", path=str(path), start=0, end=1),
        dataset_module.SampleIndex(id="window-1", path=str(path), start=1, end=2),
    ]
    original_load = utils_module._load_builtin_arousal_events
    load_count = 0

    def counted_load(npz):
        nonlocal load_count
        load_count += 1
        return original_load(npz)

    monkeypatch.setattr(utils_module, "_load_builtin_arousal_events", counted_load)

    filtered = utils_module.filter_valid_sample_indices(
        samples,
        {"arousal": utils_module.builtin_arousal_extractor},
        {"arousal": utils_module.builtin_arousal_tokenizer},
        allow_missing_channels=allow_missing_channels,
        channel_names=["arousal"],
        min_channels=1,
        max_workers=1,
    )

    assert filtered == samples
    assert load_count == 1
    assert [sample.payload["available_channels"] for sample in filtered] == [["arousal"], ["arousal"]]


@pytest.mark.parametrize("package", DATA_PACKAGES)
def test_arousal_runtime_load_validates_full_raster_once(
    package: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    utils_module = importlib.import_module(f"{package}.utils")
    dataset_module = importlib.import_module(f"{package}.default_dataset")
    path = tmp_path / f"{package.replace('.', '-')}.npz"
    _write_arousal_npz(path, utils_module.AROUSAL_METADATA_KEYS)
    original_load = utils_module._load_builtin_arousal_events
    load_count = 0

    def counted_load(npz):
        nonlocal load_count
        load_count += 1
        return original_load(npz)

    monkeypatch.setattr(utils_module, "_load_builtin_arousal_events", counted_load)
    monkeypatch.setattr(dataset_module, "_load_builtin_arousal_events", counted_load)
    dataset = object.__new__(dataset_module.DefaultDataset)
    dataset.extractors = {"arousal": utils_module.builtin_arousal_extractor}
    dataset.tokenizers = {"arousal": utils_module.builtin_arousal_tokenizer}
    dataset.mask_generators = {
        "arousal": lambda tokens: torch.zeros(tokens.shape[0], dtype=torch.bool),
    }
    src = dataset_module.SampleIndex(id="window-0", path=str(path), start=0, end=2)

    payload, tokens, masks, metadata = dataset._load_tokens_for_src(src, ["arousal"])

    assert load_count == 1
    assert payload["arousal"].shape == (60, 4)
    assert tokens["arousal"].shape == (2, 120)
    assert masks["arousal"].shape == (2,)
    assert metadata["tst"] == 3.0
