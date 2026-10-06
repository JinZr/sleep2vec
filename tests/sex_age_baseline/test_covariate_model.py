"""The baseline model is the signal models' covariate pathway without a backbone."""

from types import SimpleNamespace

import pytest
import torch

from sex_age_baseline.model import SexAgeMLP


def _config(covariates, *, normalization=None, task="survival", output_dim=2):
    task_cfg = SimpleNamespace(
        covariates=list(covariates), covariate_embedding_dim=4, covariate_normalization=normalization or {}
    )
    return SimpleNamespace(
        finetune=SimpleNamespace(
            task=SimpleNamespace(type=task, output_dim=output_dim),
            survival=task_cfg if task == "survival" else None,
            multilabel=task_cfg if task != "survival" else None,
        ),
        model=SimpleNamespace(
            name="sex_age_mlp",
            head=SimpleNamespace(hidden_dim=8, dropout=0.0, act="gelu", kwargs={"num_layers": 2}),
        ),
    )


def _metadata(**columns):
    return {name: torch.tensor(values) for name, values in columns.items()}


def _capture_inputs(model, names):
    seen = {}
    for name in names:
        model.embeddings[name].register_forward_pre_hook(
            lambda module, args, name=name: seen.__setitem__(name, args[0].flatten().tolist())
        )
    return seen


def test_continuous_covariates_are_standardized_with_the_frozen_statistics():
    normalization = {"age": {"mean": 50.0, "std": 20.0}, "bmi": {"mean": 25.0, "std": 5.0}}
    # Listed out of order: the module always consumes covariates in the canonical order.
    model = SexAgeMLP(_config(["bmi_missing", "bmi", "sex", "age"], normalization=normalization))
    seen = _capture_inputs(model, ["age", "bmi"])

    output = model(_metadata(age=[0.0, 70.0], sex=[0, 1], bmi=[20.5, 25.0], bmi_missing=[0, 1]))

    assert list(model.embeddings) == ["age", "sex", "bmi", "bmi_missing"]
    assert seen["age"] == pytest.approx([-2.5, 1.0])
    assert seen["bmi"] == pytest.approx([-0.9, 0.0])
    assert output.shape == (2, 2)


def test_age_without_statistics_keeps_the_historical_scaling():
    model = SexAgeMLP(_config(["age", "sex"], task="multilabel_classification", output_dim=3))
    seen = _capture_inputs(model, ["age"])

    output = model(_metadata(age=[45.0, 80.0], sex=[1, 0]))

    assert seen["age"] == pytest.approx([0.45, 0.8])
    assert output.shape == (2, 3)


def test_covariate_embeddings_start_as_a_no_op():
    model = SexAgeMLP(_config(["age", "sex", "bmi", "bmi_missing"], normalization={"bmi": {"mean": 25.0, "std": 5.0}}))

    output = model(_metadata(age=[20.0, 90.0], sex=[0, 1], bmi=[18.0, 40.0], bmi_missing=[1, 0]))

    assert torch.equal(output[0], output[1])


@pytest.mark.parametrize(
    ("column", "values", "message"),
    [
        ("bmi", [float("nan"), 24.0], "finite imputed values"),
        ("bmi_missing", [2, 0], "must be 0 or 1"),
    ],
)
def test_invalid_batch_covariates_fail(column, values, message):
    model = SexAgeMLP(_config(["bmi", "bmi_missing"], normalization={"bmi": {"mean": 25.0, "std": 5.0}}))
    metadata = _metadata(bmi=[24.0, 26.0], bmi_missing=[0, 1])
    metadata[column] = torch.tensor(values)

    with pytest.raises(ValueError, match=message):
        model(metadata)
