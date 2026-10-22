import argparse
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

torch = pytest.importorskip("torch")

from sleep2vec2.checkpoints import load_pretrain_init_weights  # noqa: E402
from sleep2vec2.config import ChannelConfig, load_finetune_config, load_pretrain_config  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
FINETUNE_CONFIG = REPO_ROOT / "configs" / "sleep2vec2" / "ppg_ahi_finetune.yaml"
PRETRAIN_CONFIG = REPO_ROOT / "configs" / "sleep2vec2" / "sleep2vec_dense_pretrain.yaml"


def _write_with_channel_field(tmp_path: Path, source: Path, **fields) -> Path:
    data = yaml.safe_load(source.read_text())
    data["model"]["channels"][0].update(fields)
    path = tmp_path / source.name
    path.write_text(yaml.safe_dump(data))
    return path


def test_finetune_config_parses_init_from(tmp_path: Path):
    bundle = load_finetune_config(_write_with_channel_field(tmp_path, FINETUNE_CONFIG, init_from="eeg_original"))

    assert bundle.model.channels[0].init_from == "eeg_original"


@pytest.mark.parametrize(
    ("value", "match"),
    [("", "init_from must be a non-empty string"), ("ppg", "init_from must name a different channel")],
)
def test_finetune_config_rejects_invalid_init_from(tmp_path: Path, value: str, match: str):
    path = _write_with_channel_field(tmp_path, FINETUNE_CONFIG, init_from=value)

    with pytest.raises(ValueError, match=match):
        load_finetune_config(path)


def test_pretrain_config_rejects_init_from(tmp_path: Path):
    path = _write_with_channel_field(tmp_path, PRETRAIN_CONFIG, init_from="ppg")

    with pytest.raises(ValueError, match="only supported for finetune initialization"):
        load_pretrain_config(path)


class _TinyBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer_mapping = torch.nn.ModuleDict(
            {"zmax_eegl": torch.nn.Linear(4, 3), "zmax_eegr": torch.nn.Linear(4, 3)}
        )
        self.mask_embed = torch.nn.ParameterDict(
            {name: torch.nn.Parameter(torch.zeros(3)) for name in ("zmax_eegl", "zmax_eegr")}
        )


def _save_source_checkpoint(path: Path, extra: dict | None = None) -> dict:
    source = {
        "tokenizer_mapping.eeg_original.weight": torch.randn(3, 4),
        "tokenizer_mapping.eeg_original.bias": torch.randn(3),
        "mask_embed.eeg_original": torch.randn(3),
        **(extra or {}),
    }
    torch.save({"state_dict": {f"model.{key}": value for key, value in source.items()}}, path)
    return source


def test_channel_init_from_copies_source_into_independent_targets(tmp_path: Path):
    ckpt_path = tmp_path / "pretrain.ckpt"
    source = _save_source_checkpoint(ckpt_path)
    module = _TinyBackbone()

    result = load_pretrain_init_weights(
        module,
        ckpt_path,
        device="cpu",
        prefixes=("model.",),
        channel_init_from={"zmax_eegl": "eeg_original", "zmax_eegr": "eeg_original"},
    )

    for name in ("zmax_eegl", "zmax_eegr"):
        tokenizer = module.tokenizer_mapping[name]
        assert torch.equal(tokenizer.weight, source["tokenizer_mapping.eeg_original.weight"])
        assert torch.equal(tokenizer.bias, source["tokenizer_mapping.eeg_original.bias"])
        assert torch.equal(module.mask_embed[name], source["mask_embed.eeg_original"])
    assert result.missing_keys == []

    with torch.no_grad():
        module.tokenizer_mapping["zmax_eegl"].weight.add_(1.0)
        module.mask_embed["zmax_eegl"].add_(1.0)
    assert torch.equal(module.tokenizer_mapping["zmax_eegr"].weight, source["tokenizer_mapping.eeg_original.weight"])
    assert torch.equal(module.mask_embed["zmax_eegr"], source["mask_embed.eeg_original"])


def test_without_channel_init_from_targets_stay_missing(tmp_path: Path):
    ckpt_path = tmp_path / "pretrain.ckpt"
    _save_source_checkpoint(ckpt_path, extra={"tokenizer_mapping.zmax_eegl.weight": torch.randn(3, 4)})
    module = _TinyBackbone()

    result = load_pretrain_init_weights(module, ckpt_path, device="cpu", prefixes=("model.",))

    assert "tokenizer_mapping.zmax_eegr.weight" in result.missing_keys
    assert "mask_embed.zmax_eegr" in result.missing_keys
    assert result.loaded_keys == 4


def test_channel_init_from_rejects_unknown_source(tmp_path: Path):
    ckpt_path = tmp_path / "pretrain.ckpt"
    _save_source_checkpoint(ckpt_path)

    with pytest.raises(ValueError, match=r"no tokenizer for 'eeg_missing'.*\['eeg_original'\]"):
        load_pretrain_init_weights(
            _TinyBackbone(),
            ckpt_path,
            device="cpu",
            prefixes=("model.",),
            channel_init_from={"zmax_eegl": "eeg_missing"},
        )


def test_channel_init_from_resolves_sources_against_the_checkpoint_not_other_targets(tmp_path: Path):
    ckpt_path = tmp_path / "pretrain.ckpt"
    _save_source_checkpoint(ckpt_path)

    # zmax_eegl is an init_from target, not a checkpoint channel, so zmax_eegr cannot chain through it
    # even though the mapping lists zmax_eegl first.
    with pytest.raises(ValueError, match=r"no tokenizer for 'zmax_eegl'.*\['eeg_original'\]"):
        load_pretrain_init_weights(
            _TinyBackbone(),
            ckpt_path,
            device="cpu",
            prefixes=("model.",),
            channel_init_from={"zmax_eegl": "eeg_original", "zmax_eegr": "zmax_eegl"},
        )


def test_channel_init_from_rejects_target_already_in_checkpoint(tmp_path: Path):
    ckpt_path = tmp_path / "pretrain.ckpt"
    _save_source_checkpoint(ckpt_path, extra={"tokenizer_mapping.zmax_eegl.weight": torch.randn(3, 4)})

    with pytest.raises(ValueError, match="already holds a pretrained tokenizer for 'zmax_eegl'"):
        load_pretrain_init_weights(
            _TinyBackbone(),
            ckpt_path,
            device="cpu",
            prefixes=("model.",),
            channel_init_from={"zmax_eegl": "eeg_original"},
        )


class _MismatchedTokenizerBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer_mapping = torch.nn.ModuleDict(
            {"zmax_eegl": torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.Linear(3, 3))}
        )
        self.mask_embed = torch.nn.ParameterDict({"zmax_eegl": torch.nn.Parameter(torch.zeros(3))})


def test_channel_init_from_rejects_mismatched_tokenizer_layout(tmp_path: Path):
    ckpt_path = tmp_path / "pretrain.ckpt"
    _save_source_checkpoint(ckpt_path)

    with pytest.raises(ValueError, match="does not match this model's 'zmax_eegl' tokenizer layout"):
        load_pretrain_init_weights(
            _MismatchedTokenizerBackbone(),
            ckpt_path,
            device="cpu",
            prefixes=("model.",),
            channel_init_from={"zmax_eegl": "eeg_original"},
        )


def _downstream_model_with_init_from_targets(downstream_model):
    model = downstream_model.Sleep2vecDownstreamModel.__new__(downstream_model.Sleep2vecDownstreamModel)
    torch.nn.Module.__init__(model)
    model.backbone = torch.nn.Identity()
    model.model_config = SimpleNamespace(
        cls=None,
        channels=[
            ChannelConfig(name="zmax_eegl", input_dim=3840, init_from="eeg_original"),
            ChannelConfig(name="zmax_eegr", input_dim=3840, init_from="eeg_original"),
            ChannelConfig(name="ppg", input_dim=3840),
        ],
    )
    return model


def _checkpoint_with_channels(*channels: dict) -> dict:
    return {"model_config_yaml": yaml.safe_dump({"channels": list(channels)})}


_LINEAR_TOKENIZER = {"name": "linear", "out_dim": None, "kwargs": {}}


def test_load_pretrained_backbone_passes_channel_init_from(monkeypatch: pytest.MonkeyPatch):
    from sleep2vec2 import downstream_model

    calls = []

    def fake_load(module, ckpt_path, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(loaded_keys=0, missing_keys=[], unexpected_keys=[])

    ckpt = _checkpoint_with_channels({"name": "eeg_original", "input_dim": 3840, "tokenizer": _LINEAR_TOKENIZER})
    monkeypatch.setattr(downstream_model, "load_checkpoint", lambda path, device: ckpt)
    monkeypatch.setattr(downstream_model, "load_pretrain_init_weights", fake_load)
    model = _downstream_model_with_init_from_targets(downstream_model)

    model.load_pretrained_backbone("pretrain.ckpt")

    assert calls[0]["channel_init_from"] == {"zmax_eegl": "eeg_original", "zmax_eegr": "eeg_original"}


@pytest.mark.parametrize(
    ("ckpt_channel", "match"),
    [
        (
            {
                "name": "eeg_original",
                "input_dim": 3840,
                "tokenizer": {**_LINEAR_TOKENIZER, "kwargs": {"residual_mixing": "ffn_scaled"}},
            },
            "input_dim/tokenizer differ from the checkpoint's 'eeg_original' channel",
        ),
        (
            {"name": "eeg_original", "input_dim": 1920, "tokenizer": _LINEAR_TOKENIZER},
            "input_dim/tokenizer differ from the checkpoint's 'eeg_original' channel",
        ),
        (
            {"name": "ppg", "input_dim": 3840, "tokenizer": _LINEAR_TOKENIZER},
            "has no channel named 'eeg_original'",
        ),
    ],
)
def test_load_pretrained_backbone_rejects_init_from_semantic_mismatch(
    monkeypatch: pytest.MonkeyPatch, ckpt_channel: dict, match: str
):
    from sleep2vec2 import downstream_model

    calls = []
    ckpt = _checkpoint_with_channels(ckpt_channel)
    monkeypatch.setattr(downstream_model, "load_checkpoint", lambda path, device: ckpt)
    monkeypatch.setattr(downstream_model, "load_pretrain_init_weights", lambda *args, **kwargs: calls.append(kwargs))
    model = _downstream_model_with_init_from_targets(downstream_model)

    with pytest.raises(ValueError, match=match):
        model.load_pretrained_backbone("pretrain.ckpt")
    assert calls == []


def test_finetuning_module_rejects_init_from_without_initialization_checkpoint():
    from sleep2vec2 import sleep2vec_finetuning

    model_config = SimpleNamespace(channels=[ChannelConfig(name="zmax_eegl", input_dim=3840, init_from="eeg_original")])
    args = SimpleNamespace(pretrained_backbone_path=None, ckpt_path=None, init_checkpoint_path=None)

    # Raised before the backbone is built, so the stub model_config is never consumed.
    with pytest.raises(ValueError, match="needs --pretrained-backbone-path"):
        sleep2vec_finetuning.Sleep2vecFinetuning(args, model_config)


def test_finetune_rejects_init_from_without_weight_source_before_claiming_run_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from sleep2vec2 import finetune as finetune_mod

    calls = []
    monkeypatch.setattr(finetune_mod, "_preflight_finetune_run_directory", lambda *_: calls.append("preflight"))
    monkeypatch.setattr(finetune_mod, "persist_run_config_and_args", lambda *_: calls.append("persist"))
    monkeypatch.setattr(finetune_mod, "prepare_dataloader", lambda *_: calls.append("dataloader"))
    monkeypatch.chdir(tmp_path)
    args = argparse.Namespace(
        version="unit-test",
        epochs=1,
        test_after_fit=True,
        test_all_checkpoints_after_fit=False,
        pretrained_backbone_path=None,
        ckpt_path=None,
        init_checkpoint_path=None,
    )
    bundle = SimpleNamespace(
        model=SimpleNamespace(channels=[ChannelConfig(name="zmax_eegl", input_dim=3840, init_from="eeg_original")]),
        averaging=None,
        finetune=None,
    )

    with pytest.raises(ValueError, match="needs --pretrained-backbone-path"):
        finetune_mod.supervised(args, bundle)
    assert calls == []
