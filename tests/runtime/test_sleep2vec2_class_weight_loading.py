import pytest
import torch


@pytest.fixture
def make_model():
    import pytorch_lightning as pl

    from sleep2vec2.sleep2vec_finetuning import Sleep2vecFinetuning

    def build(weights):
        model = Sleep2vecFinetuning.__new__(Sleep2vecFinetuning)
        pl.LightningModule.__init__(model)
        model.model = torch.nn.Linear(3, 5)
        weight = None if weights is None else torch.tensor(weights, dtype=torch.float32)
        model._classification_loss = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=-1)
        return model

    return build


@pytest.mark.parametrize("source_weights", [None, [1] * 5, [1, 2, 1, 2, 1], [3] * 3])
@pytest.mark.parametrize("target_weights", [None, [1] * 5, [1, 2, 1, 2, 1]])
def test_loading_uses_current_class_weights(make_model, source_weights, target_weights):
    source = make_model(source_weights)
    target = make_model(target_weights)
    state = source.state_dict()
    original = {key: value.clone() for key, value in state.items()}
    metadata = state._metadata.copy()

    result = target.load_state_dict(state)

    assert not result.missing_keys
    assert not result.unexpected_keys
    assert state.keys() == original.keys()
    assert state._metadata == metadata
    for key, value in original.items():
        assert torch.equal(state[key], value)
    for key, value in source.model.state_dict().items():
        assert torch.equal(target.model.state_dict()[key], value)
    if target_weights is None:
        assert target._classification_loss.weight is None
    else:
        assert torch.equal(target._classification_loss.weight, torch.tensor(target_weights, dtype=torch.float32))


@pytest.mark.parametrize("defect", ["missing", "unexpected", "shape"])
def test_model_state_errors_remain_strict(make_model, defect):
    state = make_model(None).state_dict()
    if defect == "missing":
        del state["model.weight"]
        message = "missing keys=.*model.weight"
    elif defect == "unexpected":
        state["model.extra"] = torch.ones(1)
        message = "unexpected keys=.*model.extra"
    else:
        state["model.weight"] = torch.ones(2, 3)
        message = "size mismatch for model.weight"

    with pytest.raises(RuntimeError, match=message):
        make_model([1, 2, 1, 2, 1]).load_state_dict(state)


def test_non_strict_loading_still_reports_model_errors(make_model):
    state = make_model(None).state_dict()
    del state["model.weight"]
    state["model.extra"] = torch.ones(1)

    result = make_model([1] * 5).load_state_dict(state, strict=False)

    assert result.missing_keys == ["model.weight"]
    assert result.unexpected_keys == ["model.extra"]


@pytest.mark.parametrize("source_has_layer_mix", [False, True])
def test_existing_layer_mix_exception_is_unchanged(make_model, source_has_layer_mix):
    source = make_model(None)
    target = make_model([1, 2, 1, 2, 1])
    model_with_layer_mix = source if source_has_layer_mix else target
    model_with_layer_mix.model.layer_mix = torch.nn.Linear(1, 1, bias=False)

    result = target.load_state_dict(source.state_dict())

    assert result.missing_keys == ([] if source_has_layer_mix else ["model.layer_mix.weight"])
    assert result.unexpected_keys == (["model.layer_mix.weight"] if source_has_layer_mix else [])
