from __future__ import annotations

import importlib
import sys
import types

import pytest
import torch
import torch.nn as nn


class _DummyBackbone(nn.Module):
    def __init__(self, hidden_size: int = 8):
        super().__init__()
        self.transformer_hidden_size = hidden_size
        self.cls_embedding = None


def _model_config(package_name: str, head_name: str = "regression"):
    config = importlib.import_module(f"{package_name}.config")
    return config.ModelConfig(
        channels=[config.ChannelConfig(name="ppg", input_dim=1)],
        backbone=config.BackboneConfig(hidden_size=8, num_hidden_layers=1, num_attention_heads=2, vocab_size=1),
        projection=config.ProjectionConfig(name="simclr", enabled=False, hidden_dim=8, out_dim=4),
        cls=config.ClsConfig(downstream="tokens", embedding_type=None),
        head=config.HeadConfig(
            name=head_name,
            channel_agg=config.ChannelAggConfig(name="concat"),
            temporal_agg=config.TemporalAggConfig(name="mean"),
            hidden_dim=8,
            dropout=0.0,
        ),
    )


def _import_downstream_module(package_name: str, monkeypatch):
    peft_stub = types.SimpleNamespace(
        LoraConfig=object,
        TaskType=types.SimpleNamespace(FEATURE_EXTRACTION="FEATURE_EXTRACTION"),
        get_peft_model=lambda model, _: model,
    )
    pretrain_stub = types.SimpleNamespace(Sleep2vecPretrainModel=object)
    monkeypatch.setitem(sys.modules, "peft", peft_stub)
    monkeypatch.setitem(sys.modules, f"{package_name}.pretrain_model", pretrain_stub)
    monkeypatch.delitem(sys.modules, f"{package_name}.downstream_model", raising=False)
    return importlib.import_module(f"{package_name}.downstream_model")


def _downstream_model(
    package_name: str,
    monkeypatch,
    *,
    is_seq: bool = False,
    head_name: str = "regression",
    is_classification: bool = False,
):
    module = _import_downstream_module(package_name, monkeypatch)
    model_config = _model_config(package_name, head_name=head_name)
    return module.Sleep2vecDownstreamModel(
        target="risk",
        backbone=_DummyBackbone(),
        channel_names=["ppg"],
        output_dim=2,
        is_classification=is_classification,
        is_seq=is_seq,
        device="cpu",
        model_config=model_config,
        head_config=model_config.head,
        survival_covariates=["age", "sex"],
        survival_covariate_embedding_dim=3,
    )


@pytest.mark.parametrize("package_name", ["sleep2vec", "sleep2vec2", "sleep2expert"])
def test_downstream_builds_survival_covariate_features(package_name: str, monkeypatch):
    model = _downstream_model(package_name, monkeypatch)
    with torch.no_grad():
        model.survival_age_embedding.weight.fill_(2.0)
        model.survival_age_embedding.bias.fill_(0.5)
        model.survival_sex_embedding.weight.copy_(
            torch.tensor(
                [
                    [1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0],
                ]
            )
        )
    batch = {"metadata": {"age": torch.tensor([50.0, 60.0]), "sex": torch.tensor([0, 1])}}

    extra_features = model._build_survival_extra_features(batch, torch.zeros(2, 8))

    assert extra_features.shape == (2, 6)
    expected = torch.tensor(
        [
            [1.5, 1.5, 1.5, 1.0, 2.0, 3.0],
            [1.7, 1.7, 1.7, 4.0, 5.0, 6.0],
        ]
    )
    assert torch.allclose(extra_features, expected)


@pytest.mark.parametrize(
    ("metadata", "pattern"),
    [
        ({"age": torch.tensor([-1.0]), "sex": torch.tensor([0])}, "age"),
        ({"age": torch.tensor([50.0]), "sex": torch.tensor([-1])}, "sex"),
    ],
)
def test_downstream_survival_covariates_reject_missing_metadata(metadata: dict, pattern: str, monkeypatch):
    model = _downstream_model("sleep2vec", monkeypatch)

    with pytest.raises(ValueError, match=pattern):
        model._build_survival_extra_features({"metadata": metadata}, torch.zeros(1, 8))


@pytest.mark.parametrize("package_name", ["sleep2vec", "sleep2vec2", "sleep2expert"])
def test_downstream_survival_covariates_reject_sequence_tasks(package_name: str, monkeypatch):
    with pytest.raises(ValueError, match="non-sequence downstream tasks"):
        _downstream_model(package_name, monkeypatch, is_seq=True)


@pytest.mark.parametrize("package_name", ["sleep2vec", "sleep2vec2", "sleep2expert"])
def test_downstream_classification_head_accepts_covariates(package_name: str, monkeypatch):
    model = _downstream_model(package_name, monkeypatch, head_name="classification", is_classification=True)

    assert model.head.extra_feature_dim == 6


@pytest.mark.parametrize("package_name", ["sleep2vec", "sleep2vec2", "sleep2expert"])
def test_regression_head_concats_extra_features(package_name: str):
    module = importlib.import_module(f"{package_name}.downstreams.heads.regression")
    head = module.RegressionHead(
        target="risk",
        feature_dim=4,
        n_mods=1,
        out_dim=2,
        agg="concat",
        hidden_dim=5,
        dropout=0.0,
        extra_feature_dim=3,
    )

    output = head([torch.ones(2, 4)], extra_features=torch.ones(2, 3))

    assert output.shape == (2, 2)
    assert head.regressor[0].in_features == 7


@pytest.mark.parametrize("package_name", ["sleep2vec", "sleep2vec2", "sleep2expert"])
def test_classification_head_concats_extra_features(package_name: str):
    module = importlib.import_module(f"{package_name}.downstreams.heads.classification")
    head = module.ClassificationHead(
        feature_dim=4,
        n_mods=1,
        n_classes=2,
        agg="concat",
        hidden_dim=5,
        dropout=0.0,
        extra_feature_dim=3,
    )

    output = head([torch.ones(2, 4)], extra_features=torch.ones(2, 3))

    assert output.shape == (2, 2)
    assert head.mlp[1].in_features == 7
