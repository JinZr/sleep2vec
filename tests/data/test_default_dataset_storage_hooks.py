from __future__ import annotations

import torch

from data.default_dataset import DefaultDataset, SampleIndex


class _HookDataset(DefaultDataset):
    def __init__(self) -> None:
        self.channel_names = ["a", "b"]
        self.randomly_select_channels = False
        self.generative = False
        self.min_channels = 2
        self.allow_missing_channels = True
        self.bucket_by_available_channels = False
        self.train_pair_probs = None
        self.train_pair_track_unique_samples = False
        self.is_train_set = False
        self.filter_calls = 0
        self.available_calls = 0
        self.loaded: list[tuple[int | str, tuple[str, ...]]] = []

        data = [
            SampleIndex(
                id=0,
                path="/tmp/no-npz-read-0.npz",
                start=0,
                end=2,
                metadata={"age": 40, "sex": 1, "source": "center-0", "path": "record-0", "split": "train"},
            ),
            SampleIndex(
                id=1,
                path="/tmp/no-npz-read-1.npz",
                start=1,
                end=3,
                metadata={"age": 41, "sex": 0, "source": "center-1", "path": "record-1", "split": "train"},
            ),
        ]

        super().__init__(
            save_preset_path=None,
            load_preset_path=None,
            data=data,
            split=["train"],
            extractors={},
            tokenizers={},
            mask_generators={},
            dataloader_config={"batch_size": 2, "shuffle": False, "num_workers": 0},
        )

    def _filter_valid_sample_indices(self, data, *, filter_max_workers):
        self.filter_calls += 1
        return list(data)

    def _get_available_channels_for_src(self, src: SampleIndex) -> set[str]:
        self.available_calls += 1
        return {"a", "b"}

    def _load_tokens_for_src(self, src: SampleIndex, chosen_channels: list[str]):
        self.loaded.append((src.id, tuple(chosen_channels)))
        length = src.end - src.start
        dims = {"a": 3, "b": 1}
        tokens = {
            channel: torch.full((length, dims[channel]), float(int(src.id) + offset))
            for offset, channel in enumerate(chosen_channels)
        }
        masks = {channel: torch.zeros(length, dtype=torch.bool) for channel in chosen_channels}
        return {"loaded_id": src.id}, tokens, masks, dict(src.metadata)


def test_default_dataset_collate_uses_storage_hooks_without_npz(monkeypatch):
    def fail_load_npz(path):
        raise AssertionError(f"Unexpected NPZ read: {path}")

    monkeypatch.setattr("data.default_dataset.load_npz", fail_load_npz)

    dataset = _HookDataset()
    batch = next(iter(dataset.dataloader(device="cpu")))

    assert dataset.filter_calls == 1
    assert dataset.available_calls == 2
    assert dataset.loaded == [(0, ("a", "b")), (1, ("a", "b"))]
    assert batch["id"] == [0, 1]
    assert batch["length"].tolist() == [2, 2]
    assert batch["token_start"].tolist() == [0, 1]
    assert batch["pair"] == ("a", "b")
    assert batch["tokens"]["a"].shape == (2, 2, 3)
    assert batch["tokens"]["b"].shape == (2, 2, 1)
    assert batch["mlm_mask"]["a"].dtype == torch.bool
    assert batch["mlm_mask"]["b"].dtype == torch.bool
    assert batch["metadata"]["age"].tolist() == [40.0, 41.0]
    assert batch["metadata"]["sex"].tolist() == [1, 0]
    assert batch["metadata"]["source"] == ["center-0", "center-1"]
    assert batch["metadata"]["path"] == ["record-0", "record-1"]
    assert batch["w"].shape == (2, 2)
    assert batch["h"].shape == (2, 2)
