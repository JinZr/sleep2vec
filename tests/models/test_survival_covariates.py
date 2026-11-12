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
    survival_covariate_fusion: str = "feature_concat",
    projection_dim: int | None = None,
):
    module = _import_downstream_module(package_name, monkeypatch)
    model_config = _model_config(package_name, head_name=head_name)
    if projection_dim is not None:
        model_config.head.channel_agg.kwargs["projection_dim"] = projection_dim
    kwargs = {}
    if package_name == "sleep2vec2":
        kwargs["survival_covariate_fusion"] = survival_covariate_fusion
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
        **kwargs,
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


@pytest.mark.parametrize(
    ("package_name", "fusion"),
    [
        ("sleep2vec", "feature_concat"),
        ("sleep2vec2", "feature_concat"),
        ("sleep2vec2", "token_concat"),
        ("sleep2expert", "feature_concat"),
    ],
)
def test_downstream_classification_head_rejects_relu_with_covariates(package_name: str, fusion: str, monkeypatch):
    module = _import_downstream_module(package_name, monkeypatch)
    config = _model_config(package_name, head_name="classification")
    config.head.act = "relu"
    kwargs = {"survival_covariate_fusion": fusion} if package_name == "sleep2vec2" else {}

    with pytest.raises(ValueError, match="do not support head.act 'relu'"):
        module.Sleep2vecDownstreamModel(
            target="disease",
            backbone=_DummyBackbone(),
            channel_names=["ppg"],
            output_dim=2,
            is_classification=True,
            is_seq=False,
            device="cpu",
            model_config=config,
            head_config=config.head,
            survival_covariates=["bmi"],
            survival_covariate_embedding_dim=3,
            covariate_normalization=_NORMALIZATION,
            **kwargs,
        )


def test_sleep2vec2_downstream_adds_survival_covariate_risk(monkeypatch):
    model = _downstream_model("sleep2vec2", monkeypatch, survival_covariate_fusion="risk")
    assert model.head.extra_feature_dim == 0
    assert model.survival_age_embedding is None
    assert model.survival_sex_embedding is None

    with torch.no_grad():
        model.survival_covariate_risk.weight.copy_(torch.tensor([[10.0, 1.0], [20.0, 2.0]]))
        model.survival_covariate_risk.bias.copy_(torch.tensor([0.5, -0.5]))

    model.backbone._tokenize_all = lambda tokens: {"ppg": tokens["ppg"]}

    def token_embeddings_to_hidden(_token_embeddings, batch):
        return torch.zeros(2, 1, 8), torch.ones(2, 1, dtype=torch.bool), None

    model.backbone._token_embeddings_to_hidden = token_embeddings_to_hidden

    class SignalHead(nn.Module):
        def forward(self, features):
            return torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=features[0].device)

    model.head = SignalHead()
    batch = {
        "tokens": {"ppg": torch.zeros(2, 1, 8)},
        "length": torch.tensor([1, 1]),
        "metadata": {"age": torch.tensor([50.0, 60.0]), "sex": torch.tensor([0, 1])},
    }

    output = model(batch)

    expected = torch.tensor([[6.5, 11.5], [10.5, 17.5]])
    assert torch.allclose(output, expected)


@pytest.mark.parametrize("projection_dim", [None, 3])
def test_sleep2vec2_downstream_concats_survival_covariates_to_tokens(monkeypatch, projection_dim: int | None):
    model = _downstream_model(
        "sleep2vec2",
        monkeypatch,
        head_name="classification",
        is_classification=True,
        survival_covariate_fusion="token_concat",
        projection_dim=projection_dim,
    )
    assert model.head.extra_feature_dim == 0
    assert model.head.feature_dim == 14

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

    signal_tokens = torch.zeros(2, 2, 8)
    batch = {"metadata": {"age": torch.tensor([50.0, 60.0]), "sex": torch.tensor([0, 1])}}

    token_hidden = model._append_survival_token_features(signal_tokens, batch)

    assert token_hidden.shape == (2, 2, 14)
    expected_covariates = torch.tensor(
        [
            [1.5, 1.5, 1.5, 1.0, 2.0, 3.0],
            [1.7, 1.7, 1.7, 4.0, 5.0, 6.0],
        ]
    )
    assert torch.allclose(token_hidden[:, 0, 8:], expected_covariates)
    assert torch.allclose(token_hidden[:, 1, 8:], expected_covariates)

    model.backbone._tokenize_all = lambda tokens: tokens
    model.backbone._token_embeddings_to_hidden = lambda tokens, _: (tokens, None, None)
    logits = model({**batch, "tokens": {"ppg": signal_tokens}, "length": torch.tensor([2, 2])})
    assert logits.shape == (2, 2)
    first_linear = next(layer for layer in model.head.mlp if isinstance(layer, nn.Linear))
    assert first_linear.in_features == (projection_dim or 14)
    if projection_dim is not None:
        projection = model.head.fusion.aggregator.projection
        assert projection.weight.shape == (projection_dim, 14)
        logits.square().mean().backward()
        assert projection.weight.grad is not None
        assert torch.count_nonzero(projection.weight.grad[:, 8:]) > 0


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


_ALL_COVARIATES_SHUFFLED = ["bmi_missing", "sex", "bmi", "age"]
_NORMALIZATION = {"age": {"mean": 50.0, "std": 25.0}, "bmi": {"mean": 25.0, "std": 5.0}}
# Canonical order is age, sex, bmi, bmi_missing whatever order the covariates were listed in.
_EXPECTED_VALUES = torch.tensor([[-2.0, 0.0, -1.0, 1.0], [1.0, 1.0, 1.0, 0.0]])


def _bmi_model(
    package_name: str, monkeypatch, *, covariates=_ALL_COVARIATES_SHUFFLED, normalization=_NORMALIZATION, **kwargs
):
    module = _import_downstream_module(package_name, monkeypatch)
    config = _model_config(package_name, head_name="classification")
    return module.Sleep2vecDownstreamModel(
        target="disease",
        backbone=_DummyBackbone(),
        channel_names=["ppg"],
        output_dim=2,
        is_classification=True,
        is_seq=False,
        device="cpu",
        model_config=config,
        head_config=config.head,
        survival_covariates=covariates,
        survival_covariate_embedding_dim=3,
        covariate_normalization=normalization,
        **kwargs,
    )


def _bmi_batch():
    return {
        "metadata": {
            "age": torch.tensor([0.0, 75.0]),
            "sex": torch.tensor([0, 1]),
            "bmi": torch.tensor([20.0, 30.0]),
            "bmi_missing": torch.tensor([1, 0]),
        }
    }


def _set_identity_embeddings(model) -> None:
    with torch.no_grad():
        for embedding in (model.survival_age_embedding, model.survival_bmi_embedding):
            embedding.weight.fill_(1.0)
            embedding.bias.zero_()
        for embedding in (model.survival_sex_embedding, model.survival_bmi_missing_embedding):
            embedding.weight[0].zero_()
            embedding.weight[1].fill_(1.0)


@pytest.mark.parametrize("package_name", ["sleep2vec", "sleep2vec2", "sleep2expert"])
def test_bmi_covariates_keep_checkpoint_parameter_names(package_name: str, monkeypatch):
    model = _bmi_model(package_name, monkeypatch)

    covariate_keys = {key for key in model.state_dict() if key.startswith("survival_")}

    assert covariate_keys == {
        "survival_age_embedding.weight",
        "survival_age_embedding.bias",
        "survival_sex_embedding.weight",
        "survival_bmi_embedding.weight",
        "survival_bmi_embedding.bias",
        "survival_bmi_missing_embedding.weight",
    }
    assert all(not value.any() for key, value in model.state_dict().items() if key in covariate_keys)


@pytest.mark.parametrize("package_name", ["sleep2vec", "sleep2vec2", "sleep2expert"])
def test_bmi_covariate_features_are_normalized_in_canonical_order(package_name: str, monkeypatch):
    model = _bmi_model(package_name, monkeypatch)
    _set_identity_embeddings(model)

    features = model._build_survival_extra_features(_bmi_batch(), torch.zeros(2, 8))

    assert model.head.extra_feature_dim == 12
    assert torch.equal(features, _EXPECTED_VALUES.repeat_interleave(3, dim=1))


@pytest.mark.parametrize(
    ("name", "value", "pattern"),
    [
        ("bmi", float("nan"), "'bmi' requires finite imputed values"),
        ("bmi_missing", -1, "'bmi_missing' must be 0 or 1"),
        ("age", -1.0, "'age' is missing"),
    ],
)
@pytest.mark.parametrize("package_name", ["sleep2vec", "sleep2vec2", "sleep2expert"])
def test_bmi_covariate_features_reject_invalid_metadata(package_name: str, name, value, pattern, monkeypatch):
    model = _bmi_model(package_name, monkeypatch)
    batch = _bmi_batch()
    batch["metadata"][name][0] = value

    with pytest.raises(ValueError, match=pattern):
        model._build_survival_extra_features(batch, torch.zeros(2, 8))


@pytest.mark.parametrize("fusion", ["feature_concat", "token_concat", "risk"])
def test_sleep2vec2_bmi_covariates_normalize_across_fusions(fusion: str, monkeypatch):
    model = _bmi_model("sleep2vec2", monkeypatch, survival_covariate_fusion=fusion)
    batch = _bmi_batch()

    assert torch.equal(model._build_survival_covariate_values(batch, torch.zeros(2, 8)), _EXPECTED_VALUES)
    if fusion != "risk":
        _set_identity_embeddings(model)
        features = model._build_survival_covariate_embeddings(batch, torch.zeros(2, 8))
        assert torch.equal(features, _EXPECTED_VALUES.repeat_interleave(3, dim=1))

    batch["metadata"]["bmi_missing"] = torch.tensor([0.5, 0.0])
    with pytest.raises(ValueError, match="bmi_missing"):
        model._build_survival_covariate_values(batch, torch.zeros(2, 8))


def test_sleep2vec2_risk_covariates_standardize_before_reduced_precision(monkeypatch):
    model = _bmi_model("sleep2vec2", monkeypatch, survival_covariate_fusion="risk")
    batch = _bmi_batch()
    batch["metadata"]["bmi"] = torch.tensor([25.05, 30.0])

    values = model._build_survival_covariate_values(batch, torch.zeros(2, 8, dtype=torch.bfloat16))

    assert values.dtype == torch.bfloat16
    assert values[0, 2].item() == pytest.approx(0.01, abs=1e-4)


@pytest.mark.parametrize("package_name", ["sleep2vec", "sleep2vec2", "sleep2expert"])
def test_all_eight_covariate_embeddings_share_feature_layout(package_name, monkeypatch):
    normalization = {**_NORMALIZATION, "height": {"mean": 170, "std": 10}, "weight": {"mean": 70, "std": 10}}
    model = _bmi_model(
        package_name,
        monkeypatch,
        covariates=["weight_missing", "weight", "height_missing", "height", *_ALL_COVARIATES_SHUFFLED],
        normalization=normalization,
    )
    batch = _bmi_batch()
    batch["metadata"].update(
        height=torch.tensor([160.0, 175.0]),
        height_missing=torch.tensor([1, 0]),
        weight=torch.tensor([65.0, 90.0]),
        weight_missing=torch.tensor([0, 1]),
    )
    assert model.head.extra_feature_dim == 24
    expected_names = ["age", "sex", "bmi", "bmi_missing", "height", "height_missing", "weight", "weight_missing"]
    with torch.no_grad():
        for name in expected_names:
            embedding = getattr(model, f"survival_{name}_embedding")
            assert not embedding.weight.any()
            if name in normalization:
                embedding.weight.fill_(1.0)
                embedding.bias.zero_()
            else:
                embedding.weight[0].zero_()
                embedding.weight[1].fill_(1.0)
    expected = torch.cat((_EXPECTED_VALUES, torch.tensor([[-1.0, 1.0, -0.5, 0.0], [0.5, 0.0, 2.0, 1.0]])), dim=1)
    features = model._build_survival_extra_features(batch, torch.zeros(2, 8))
    assert torch.equal(features, expected.repeat_interleave(3, dim=1))

    for name in ("height", "weight"):
        original = batch["metadata"][name]
        batch["metadata"][name] = torch.tensor([float("nan"), 175.0])
        with pytest.raises(ValueError, match=f"'{name}' requires finite imputed values"):
            model._build_survival_extra_features(batch, torch.zeros(2, 8))
        batch["metadata"][name] = original
    for name in ("height_missing", "weight_missing"):
        original = batch["metadata"][name]
        batch["metadata"][name] = torch.tensor([0.5, 0.0])
        with pytest.raises(ValueError, match=f"'{name}' must be 0 or 1"):
            model._build_survival_extra_features(batch, torch.zeros(2, 8))
        batch["metadata"][name] = original


@pytest.mark.parametrize("fusion", ["feature_concat", "token_concat", "risk"])
def test_sleep2vec2_height_weight_covariates_normalize_across_fusions(fusion, monkeypatch):
    normalization = {**_NORMALIZATION, "height": {"mean": 170, "std": 10}, "weight": {"mean": 70, "std": 10}}
    model = _bmi_model(
        "sleep2vec2",
        monkeypatch,
        covariates=["weight_missing", "height_missing", "weight", "height", *_ALL_COVARIATES_SHUFFLED],
        normalization=normalization,
        survival_covariate_fusion=fusion,
    )
    batch = _bmi_batch()
    batch["metadata"].update(
        height=torch.tensor([160.0, 175.0]),
        height_missing=torch.tensor([1, 0]),
        weight=torch.tensor([65.0, 90.0]),
        weight_missing=torch.tensor([0, 1]),
    )
    expected = torch.cat((_EXPECTED_VALUES, torch.tensor([[-1.0, 1.0, -0.5, 0.0], [0.5, 0.0, 2.0, 1.0]])), dim=1)

    assert torch.equal(model._build_survival_covariate_values(batch, torch.zeros(2, 8)), expected)
    if fusion == "risk":
        assert model.survival_covariate_risk.in_features == 8
    else:
        for name in ("height", "height_missing", "weight", "weight_missing"):
            assert not getattr(model, f"survival_{name}_embedding").weight.any()
        _set_identity_embeddings(model)
        with torch.no_grad():
            for name in ("height", "weight"):
                embedding = getattr(model, f"survival_{name}_embedding")
                embedding.weight.fill_(1.0)
                embedding.bias.zero_()
            for name in ("height_missing", "weight_missing"):
                embedding = getattr(model, f"survival_{name}_embedding")
                embedding.weight[0].zero_()
                embedding.weight[1].fill_(1.0)
        features = model._build_survival_covariate_embeddings(batch, torch.zeros(2, 8))
        assert torch.equal(features, expected.repeat_interleave(3, dim=1))

    for name in ("height", "weight"):
        original = batch["metadata"][name]
        batch["metadata"][name] = torch.tensor([float("nan"), 175.0])
        with pytest.raises(ValueError, match=f"'{name}' requires finite imputed values"):
            model._build_survival_covariate_values(batch, torch.zeros(2, 8))
        batch["metadata"][name] = original
    for name in ("height_missing", "weight_missing"):
        original = batch["metadata"][name]
        batch["metadata"][name] = torch.tensor([0.5, 0.0])
        with pytest.raises(ValueError, match=f"'{name}' must be 0 or 1"):
            model._build_survival_covariate_values(batch, torch.zeros(2, 8))
        batch["metadata"][name] = original
