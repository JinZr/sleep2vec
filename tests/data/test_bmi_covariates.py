from importlib import import_module
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

PACKAGES = [("data", "sleep2vec"), ("sleep2vec2.data", "sleep2vec2")]


def _write_multilabel_fixture(tmp_path):
    signal_path = tmp_path / "signal.npz"
    np.savez(signal_path, wearable=np.arange(8, dtype=np.float32))
    index_path = tmp_path / "index.csv"
    pd.DataFrame(
        {
            "path": [str(signal_path)] * 3,
            "eid": ["p001", "p002", "p003"],
            "split": ["train"] * 3,
            "duration": [60] * 3,
            "age": [40, 67, 55],
            "sex": [0, 1, 1],
            "bmi": [23.75, 26.125, float("nan")],
            "bmi_missing": [0, 1, 1],
        }
    ).to_csv(index_path, index=False)
    label_path = tmp_path / "labels.csv"
    mask_path = tmp_path / "masks.csv"
    columns_path = tmp_path / "columns.txt"
    eids = ["p001", "p002", "p003"]
    pd.DataFrame({"eid": eids, "disease": [0, 1, 0], "second": [1, 0, 0], "third": [0, 0, 1]}).to_csv(
        label_path, index=False
    )
    pd.DataFrame({"eid": eids, "disease": [1] * 3, "second": [1] * 3, "third": [1] * 3}).to_csv(mask_path, index=False)
    columns_path.write_text("disease\nsecond\nthird\n")
    label_config = SimpleNamespace(
        key_column="eid",
        label_index=str(label_path),
        has_label_index=str(mask_path),
        disease_columns_index=str(columns_path),
        covariates=["age", "sex", "bmi", "bmi_missing"],
    )
    return index_path, label_config


def _loader_args(index_path, preset_path, label_config):
    return SimpleNamespace(
        label_name="disease",
        data_channel_names=["wearable"],
        channel_input_dims={"wearable": 4},
        finetune_preset_path=preset_path,
        finetune_data_index=str(index_path),
        max_tokens=2,
        batch_size=2,
        num_workers=0,
        device="cpu",
        is_classification=True,
        is_multilabel=True,
        multilabel=label_config,
        output_dim=3,
    )


@pytest.mark.parametrize(("data_package", "variant"), PACKAGES)
def test_bmi_covariates_reach_the_multilabel_batch_from_index_and_preset(tmp_path, data_package, variant):
    index_path, label_config = _write_multilabel_fixture(tmp_path)
    preset_path = tmp_path / "preset.pkl"
    import_module(f"{data_package}.psg_pretrain_dataset").PSGPretrainDataset(
        channel_names=["wearable"],
        channel_input_dims={"wearable": 4},
        index=str(index_path),
        split=["train"],
        max_tokens=2,
        mask_rate=0.0,
        randomly_select_channels=False,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        multilabel_label_config=label_config,
        multilabel_output_dim=3,
        save_preset_path=str(preset_path),
        load_preset_path=None,
    )
    build_loader = import_module(f"{variant}.utils")._build_finetune_loader

    for preset in (None, str(preset_path)):
        args = _loader_args(index_path, preset, label_config)
        loader = build_loader(args, split=["train"], sources=[], shuffle=False, is_train_set=True)
        metadata = next(iter(loader))["metadata"]

        # The row with a missing (NaN) bmi is dropped by required-covariate filtering; bmi keeps its fraction.
        torch.testing.assert_close(metadata["bmi"], torch.tensor([23.75, 26.125]))
        assert metadata["bmi_missing"].tolist() == [0, 1]
        torch.testing.assert_close(metadata["age"], torch.tensor([40.0, 67.0]))
        assert metadata["disease_label"].tolist() == [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]


@pytest.mark.parametrize(("data_package", "variant"), PACKAGES)
def test_bmi_covariates_fail_fast_on_an_index_without_bmi(tmp_path, data_package, variant):
    index_path, label_config = _write_multilabel_fixture(tmp_path)
    pd.read_csv(index_path).drop(columns=["bmi", "bmi_missing"]).to_csv(index_path, index=False)
    args = _loader_args(index_path, None, label_config)

    with pytest.raises(ValueError, match="No samples remain after required metadata filtering"):
        import_module(f"{variant}.utils")._build_finetune_loader(
            args, split=["train"], sources=[], shuffle=False, is_train_set=True
        )


@pytest.mark.parametrize(("data_package", "variant"), PACKAGES)
def test_process_metadata_always_emits_bmi_without_truncation(data_package, variant):
    process_metadata = import_module(f"{data_package}.metadata").process_metadata

    processed = process_metadata(
        [
            SimpleNamespace(metadata={"bmi": 23.75, "bmi_missing": 0}),
            SimpleNamespace(metadata={"bmi": "unused", "bmi_missing": 0.5}),
            SimpleNamespace(metadata={}),
        ],
        [],
    )

    assert processed["bmi"].dtype == torch.float
    assert processed["bmi"][0].item() == 23.75
    assert processed["bmi"][1:].isnan().all()
    assert processed["bmi_missing"].tolist() == [0, -1, -1]


@pytest.mark.parametrize(
    ("name", "value", "valid"),
    [
        ("age", 0, True),
        ("age", -1, False),
        ("sex", "1", True),
        ("sex", 2, False),
        ("bmi", 18.5, True),
        ("bmi", "22.25", True),
        ("bmi", float("inf"), False),
        ("bmi", None, False),
        ("bmi_missing", 1.0, True),
        ("bmi_missing", 0.5, False),
        ("eid", "001", True),
        ("eid", float("nan"), False),
    ],
)
@pytest.mark.parametrize("data_package", ["data", "sleep2vec2.data"])
def test_required_covariate_is_valid(data_package, name, value, valid):
    assert import_module(f"{data_package}.metadata").required_covariate_is_valid(name, value) is valid
