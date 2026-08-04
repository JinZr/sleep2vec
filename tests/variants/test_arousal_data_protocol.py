import importlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch


@pytest.mark.parametrize("namespace", ["sleep2vec2", "sleep2expert"])
def test_variant_arousal_registry_and_preset_contract_are_package_local(namespace: str) -> None:
    dataset_module = importlib.import_module(f"{namespace}.data.psg_pretrain_dataset")
    preset_module = importlib.import_module(f"{namespace}.preprocess.save_dataset_presets")
    converter_module = importlib.import_module(f"{namespace}.preprocess.convert_npz_to_kaldi")

    registry = dataset_module._build_channel_registry(
        channel_names=["arousal"],
        channel_input_dims={},
        mask_rate=0.9,
    )
    tokens = registry["arousal"][1](torch.arange(240, dtype=torch.float32).view(60, 4) % 2)
    channels, dims = preset_module._resolve_validation_channels(
        model_channels=["ppg"],
        channel_input_dims={"ppg": 8},
        preset_required_channels=None,
        selected_channels=["ppg", "arousal"],
    )

    assert registry["arousal"][0].__module__ == f"{namespace}.data.utils"
    assert registry["arousal"][1].__module__ == f"{namespace}.data.utils"
    assert tokens.shape == (2, 120)
    assert torch.equal(tokens.view(2, 30, 4), (torch.arange(240).view(2, 30, 4) % 2).to(torch.float32))
    assert not registry["arousal"][2](tokens).any()
    assert channels == ["ppg", "arousal", "stage5"]
    assert dims == {"ppg": 8, "arousal": 120, "stage5": 1}
    assert preset_module._resolve_effective_min_channels(
        channel_names=channels,
        cli_min_channels=1,
        preset_min_channels=1,
    ) == len(channels)
    assert "arousal" in converter_module.UNCOMPRESSED_BUILTIN_CHANNELS
    with pytest.raises(ValueError, match="input_dim=120"):
        dataset_module._build_channel_registry(
            channel_names=["arousal"],
            channel_input_dims={"arousal": 4},
            mask_rate=0.0,
        )
    with pytest.raises(ValueError, match=r"exactly 60 rows.*got 30"):
        registry["arousal"][0]({"arousal_event": torch.zeros(30, 4).numpy()}, 0, 2)


@pytest.mark.parametrize("namespace", ["sleep2vec2", "sleep2expert"])
def test_variant_arousal_converter_supports_multiple_ark_shards(
    namespace: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    converter_module = importlib.import_module(f"{namespace}.preprocess.convert_npz_to_kaldi")
    config_path = tmp_path / "config.yaml"
    index_path = tmp_path / "index.csv"
    npz_path = tmp_path / "sample.npz"
    config_path.write_text("model:\n  channels:\n    - name: ppg\n      input_dim: 8\n")
    payload = {
        "arousal_event": np.zeros((60, 4), dtype=np.float32),
        "stage5": np.asarray([1.0, 2.0], dtype=np.float32),
        **{key: np.asarray(0.0, dtype=np.float32) for key in converter_module.AROUSAL_METADATA_KEYS},
        "tst": np.asarray(3.0, dtype=np.float32),
    }
    np.savez(npz_path, **payload)
    index_path.write_text(
        "path,duration,split,dataset,source,session_id,arousal_event_mask,stage_mask\n"
        f"{npz_path},60,train,center-a,center-a,record-1,1,1\n"
    )
    output_dir = tmp_path / "kaldi"
    args = converter_module.parse_args(
        [
            "--index",
            str(index_path),
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
            "--max-tokens",
            "1",
            "--ark-shards",
            "2",
            "--num-workers",
            "1",
            "--extra-channels",
            "arousal",
        ]
    )
    monkeypatch.setattr(
        converter_module,
        "_resolve_channels",
        lambda unused_args: (["arousal", "stage5"], {"arousal": 120, "stage5": 1}, 2),
    )

    converter_module.convert(args)

    channel_dir = output_dir / "channels" / "train"
    assert len((channel_dir / "arousal.1.scp").read_text().splitlines()) == 1
    assert len((channel_dir / "arousal.2.scp").read_text().splitlines()) == 1
    assert len((channel_dir / "arousal.scp").read_text().splitlines()) == 2


@pytest.mark.parametrize("namespace", ["sleep2vec2", "sleep2expert"])
@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("token_sec", "token_sec=30"),
        ("input_dim", "input_dim=120"),
        ("storage", "ark_storage='float_matrix'"),
    ],
)
def test_variant_arousal_kaldi_manifest_is_strict(namespace: str, mutation: str, match: str, tmp_path: Path) -> None:
    kaldi_module = importlib.import_module(f"{namespace}.data.kaldi_psg_dataset")
    arousal_spec = {
        "input_dim": 120,
        "scp": "channels/train/arousal.scp",
        "ark_storage": "float_matrix",
    }
    manifest = {
        "token_sec": 30,
        "splits": {
            "train": {
                "manifest": "manifests/train.csv",
                "channels": {"arousal": arousal_spec},
            }
        },
    }
    if mutation == "token_sec":
        manifest["token_sec"] = 30.9
    elif mutation == "input_dim":
        arousal_spec["input_dim"] = 120.9
    else:
        arousal_spec["ark_storage"] = "compressed_matrix"
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest) + "\n")

    with pytest.raises(ValueError, match=match):
        kaldi_module.KaldiPSGDataset(
            channel_names=["arousal"],
            channel_input_dims={"arousal": 120},
            kaldi_data_root=tmp_path,
            manifest=manifest_path,
            split=["train"],
            max_tokens=2,
            mask_rate=0.0,
        )
