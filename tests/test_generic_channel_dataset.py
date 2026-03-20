from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from data.psg_pretrain_dataset import PSGPretrainDataset


def test_psg_dataset_supports_custom_channel_input_dims(tmp_path: Path):
    npz_path = tmp_path / "sample.npz"
    np.savez(npz_path, ppg=np.arange(8, dtype=np.float32))

    index_path = tmp_path / "index.csv"
    pd.DataFrame(
        [
            {
                "path": str(npz_path),
                "split": "train",
                "duration": 60,
                "age": 40,
                "sex": 1,
            }
        ]
    ).to_csv(index_path, index=False)

    dataset = PSGPretrainDataset(
        channel_names=["ppg"],
        channel_input_dims={"ppg": 4},
        save_preset_path=None,
        load_preset_path=None,
        index=str(index_path),
        split=["train"],
        max_tokens=2,
        mask_rate=0.0,
        randomly_select_channels=False,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    batch = next(iter(dataset.dataloader(device="cpu")))
    assert batch["tokens"]["ppg"].shape == (1, 2, 4)
