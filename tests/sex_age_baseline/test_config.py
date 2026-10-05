from __future__ import annotations

from itertools import combinations
from pathlib import Path

import pytest
import yaml

from sex_age_baseline.config import load_config

BMI_NORMALIZATION = {"bmi": {"mean": 25.0, "std": 5.0}}


def _write_yaml(path: Path, payload: dict) -> Path:
    path.write_text(yaml.safe_dump(payload))
    return path


def _cox_payload(tmp_path: Path) -> dict:
    return {
        "model": {
            "name": "sex_age_mlp",
            "head": {
                "name": "classification",
                "hidden_dim": 8,
                "dropout": 0.1,
                "act": "elu",
                "kwargs": {"num_layers": 3},
            },
        },
        "data": {
            "backend": "npz",
            "finetune_data_index": str(tmp_path / "index.csv"),
            "finetune_preset_path": None,
            "kaldi_data_root": None,
            "kaldi_manifest": None,
            "train_dataset_names": [],
            "test_dataset_names": [],
        },
        "finetune": {
            "task": {
                "type": "survival",
                "output_dim": 2,
                "is_seq": False,
                "monitor": "val_c_index",
                "monitor_mod": "max",
            },
            "survival": {
                "key_column": "eid",
                "disease_columns_index": "disease_columns.txt",
                "event_time_index": "event_time.csv",
                "is_event_index": "is_event.csv",
                "has_label_index": "has_label.csv",
                "covariates": ["age", "sex"],
                "covariate_embedding_dim": 4,
            },
        },
    }


def _multilabel_payload(tmp_path: Path) -> dict:
    payload = _cox_payload(tmp_path)
    payload["finetune"] = {
        "task": {
            "type": "multilabel_classification",
            "output_dim": 2,
            "is_seq": False,
            "monitor": "val_macro_auroc",
            "monitor_mod": "max",
        },
        "multilabel": {
            "key_column": "eid",
            "disease_columns_index": "disease_columns.txt",
            "label_index": "disease_label.csv",
            "has_label_index": "has_label.csv",
            "covariates": ["age", "sex"],
            "covariate_embedding_dim": 4,
        },
        "loss": {"pos_weight": None},
    }
    return payload


@pytest.mark.parametrize("path", ["configs/sex_age_baseline/cox.yaml", "configs/sex_age_baseline/multilabel.yaml"])
def test_checked_in_configs_load(path: str):
    cfg = load_config(path)
    task_cfg = cfg.finetune.survival or cfg.finetune.multilabel

    assert task_cfg.covariates == ["age", "sex"]
    assert task_cfg.key_column == "eid"
    assert cfg.data.backend == "npz"
    # Like the signal templates, the data source is bound by the recipe or CLI rather than the repo config.
    assert cfg.data.finetune_data_index is None and cfg.data.finetune_preset_path is None


@pytest.mark.parametrize(
    "covariates", [list(c) for n in (1, 2, 4) for c in combinations(("age", "sex", "bmi", "bmi_missing"), n)]
)
@pytest.mark.parametrize("num_layers", [1, 2, 3])
@pytest.mark.parametrize("variant", ["sleep2vec", "sleep2vec2"])
def test_covariate_subsets_and_dense_head_match_the_shared_builders(tmp_path, covariates, num_layers, variant):
    import importlib

    import torch
    from torch import nn

    from sex_age_baseline.model import SexAgeMLP
    from sleep2vec.modules.covariates import embed_covariates

    ClassificationHead = importlib.import_module(f"{variant}.downstreams.heads.classification").ClassificationHead

    payload = _cox_payload(tmp_path)
    survival = payload["finetune"]["survival"]
    survival["covariates"] = list(reversed(covariates))
    if "bmi" in covariates:
        survival["covariate_normalization"] = BMI_NORMALIZATION
    payload["model"]["head"]["kwargs"]["num_layers"] = num_layers
    cfg = load_config(_write_yaml(tmp_path / "subset.yaml", payload))
    network = SexAgeMLP(cfg).eval()
    with torch.no_grad():
        for parameter in network.embeddings.parameters():
            parameter.normal_()
    metadata = {
        "age": torch.tensor([40.0, 60.0]),
        "sex": torch.tensor([0, 1]),
        "bmi": torch.tensor([20.0, 30.0]),
        "bmi_missing": torch.tensor([1, 0]),
    }
    embedded = embed_covariates(network.embeddings, metadata, network.covariate_normalization, torch.device("cpu"))
    reference = ClassificationHead(
        4 * len(covariates), 1, 2, agg="mean", hidden_dim=8, dropout=0.1, act=nn.ELU, num_layers=num_layers
    ).mlp.eval()
    reference.load_state_dict(network.head.state_dict())

    torch.testing.assert_close(network(metadata), reference(embedded))
    assert set(network.embeddings) == set(covariates)
    assert sum(isinstance(m, nn.Linear) for m in network.head) == num_layers


@pytest.mark.parametrize("act", ["elu", "gelu", "silu"])
@pytest.mark.parametrize("num_layers", [1, 2, 3])
def test_zero_initialized_embeddings_receive_gradients(tmp_path, act, num_layers):
    import torch

    from sex_age_baseline.model import SexAgeMLP

    payload = _cox_payload(tmp_path)
    survival = payload["finetune"]["survival"]
    survival.update(covariates=["age", "sex", "bmi", "bmi_missing"], covariate_normalization=BMI_NORMALIZATION)
    payload["model"]["head"].update(act=act, dropout=0.0)
    payload["model"]["head"]["kwargs"]["num_layers"] = num_layers
    network = SexAgeMLP(load_config(_write_yaml(tmp_path / "compatible.yaml", payload)))
    with torch.no_grad():
        for parameter in network.head.parameters():
            parameter.fill_(0.1)
    metadata = {
        "age": torch.tensor([40.0, 60.0]),
        "sex": torch.tensor([0, 1]),
        # Not symmetric around the mean: the identical zero-init rows would otherwise cancel the bmi gradient.
        "bmi": torch.tensor([20.0, 35.0]),
        "bmi_missing": torch.tensor([0, 1]),
    }

    network(metadata).sum().backward()

    assert all(
        parameter.grad is not None and torch.count_nonzero(parameter.grad)
        for name, parameter in network.embeddings.named_parameters()
        if name.endswith("weight")
    )


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda p: p["model"]["head"].update(act="relu"), "elu, gelu, or silu"),
        (lambda p: p["model"]["head"].update(activation="elu"), "unsupported fields"),
        (lambda p: p["model"]["head"]["kwargs"].update(num_layers=4), "num_layers"),
        # The removed per-covariate encoder blocks must not linger as inert model fields.
        (lambda p: p["model"].update(features=["age", "sex"]), "covariates belong in finetune"),
        (lambda p: p["model"].update(age={"transform": "divide", "scale": 100.0}), "covariates belong in finetune"),
        (lambda p: p["finetune"]["survival"].update(covariates=[]), "non-empty"),
        (lambda p: p["finetune"]["survival"].pop("covariates"), "non-empty"),
        (lambda p: p["finetune"]["survival"].update(covariates=["age", "age"]), "duplicates"),
        (lambda p: p["finetune"]["survival"].update(covariates=["height"]), "only supports"),
        (lambda p: p["finetune"]["survival"].update(covariates=["age", "bmi"]), "BMI requires"),
        (
            lambda p: p["finetune"]["survival"].update(covariate_normalization={"bmi": {"mean": 25.0, "std": 5.0}}),
            "selected age and bmi",
        ),
        (
            lambda p: p["finetune"]["survival"].update(covariate_normalization={"age": {"mean": 50.0}}),
            "requires mean and std",
        ),
        (
            lambda p: p["finetune"]["survival"].update(covariate_normalization={"age": {"mean": 50.0, "std": 0.0}}),
            "std must be positive",
        ),
    ],
)
def test_model_and_covariate_contract_rejects_invalid_fields(tmp_path, mutate, match):
    payload = _cox_payload(tmp_path)
    mutate(payload)
    with pytest.raises(ValueError, match=match):
        load_config(_write_yaml(tmp_path / "invalid.yaml", payload))


@pytest.mark.parametrize("payload_factory", [_cox_payload, _multilabel_payload])
def test_task_block_carries_the_signal_covariate_fields(tmp_path, payload_factory):
    payload = payload_factory(tmp_path)
    block = "survival" if "survival" in payload["finetune"] else "multilabel"
    payload["finetune"][block].update(
        covariates=["age", "sex", "bmi", "bmi_missing"],
        covariate_normalization={"age": {"mean": 50, "std": 20}, **BMI_NORMALIZATION},
    )

    cfg = load_config(_write_yaml(tmp_path / "bmi.yaml", payload))
    task_cfg = getattr(cfg.finetune, block)

    assert task_cfg.covariates == ["age", "sex", "bmi", "bmi_missing"]
    assert task_cfg.covariate_embedding_dim == 4
    assert task_cfg.covariate_normalization == {"age": {"mean": 50.0, "std": 20.0}, **BMI_NORMALIZATION}


def test_model_contract_freezes_the_covariate_pathway(tmp_path):
    from sex_age_baseline.runtime import _model_contract

    payload = _multilabel_payload(tmp_path)
    payload["finetune"]["multilabel"].update(
        covariates=["bmi", "age", "bmi_missing"], covariate_normalization=BMI_NORMALIZATION
    )
    cfg = load_config(_write_yaml(tmp_path / "contract.yaml", payload))

    assert _model_contract(cfg) == {
        "covariates": ["age", "bmi", "bmi_missing"],
        "covariate_embedding_dim": 4,
        "covariate_normalization": BMI_NORMALIZATION,
        "head": payload["model"]["head"],
    }


@pytest.mark.parametrize("loss", [{"pos_weight": 2.0}, {"pos_weigth": 2.0}])
def test_survival_config_rejects_unsupported_loss_fields(tmp_path: Path, loss: dict):
    payload = _cox_payload(tmp_path)
    payload["finetune"]["loss"] = loss
    config = _write_yaml(tmp_path / "cox-with-loss.yaml", payload)

    with pytest.raises(ValueError, match="Survival finetune.loss supports only eps"):
        load_config(config)


@pytest.mark.parametrize("loss,expected", [(None, 1e-9), ({}, 1e-9), ({"eps": 1e-7}, 1e-7)])
def test_survival_eps_default_and_override(tmp_path, loss, expected):
    payload = _cox_payload(tmp_path)
    if loss is not None:
        payload["finetune"]["loss"] = loss
    cfg = load_config(_write_yaml(tmp_path / "eps.yaml", payload))
    assert cfg.finetune.loss.eps == expected


@pytest.mark.parametrize("eps", [0, -1, float("nan"), float("inf"), True])
def test_survival_eps_invalid(tmp_path, eps):
    payload = _cox_payload(tmp_path)
    payload["finetune"]["loss"] = {"eps": eps}
    with pytest.raises(ValueError):
        load_config(_write_yaml(tmp_path / "eps.yaml", payload))


@pytest.mark.parametrize("field,value", [("tuning", {"preset": "head_only"}), ("lr", 0.001), ("epochs", 8)])
def test_finetune_rejects_unconsumed_training_fields(tmp_path, field, value):
    payload = _cox_payload(tmp_path)
    payload["finetune"][field] = value
    with pytest.raises(ValueError, match="finetune contains unsupported fields"):
        load_config(_write_yaml(tmp_path / "unconsumed.yaml", payload))


@pytest.mark.parametrize(
    "field,value",
    [("outputs", {"prediction_csv": True, "per_disease_metrics_csv": True}), ("runtime", {"epochs": 8})],
)
def test_config_rejects_top_level_fields_outside_model_data_finetune(tmp_path, field, value):
    # Prediction export is the CLI --export-predictions flag; a YAML outputs block must not linger as a switch.
    payload = _cox_payload(tmp_path)
    payload[field] = value
    with pytest.raises(ValueError, match="unsupported top-level fields"):
        load_config(_write_yaml(tmp_path / "top-level.yaml", payload))


def test_model_rejects_runtime_learning_rate(tmp_path):
    payload = _cox_payload(tmp_path)
    payload["model"]["lr"] = 0.001
    with pytest.raises(ValueError, match="unsupported fields"):
        load_config(_write_yaml(tmp_path / "model-lr.yaml", payload))


@pytest.mark.parametrize(
    ("payload_factory", "inactive_block", "message"),
    [
        (_cox_payload, "multilabel", "finetune.multilabel is only supported"),
        (_multilabel_payload, "survival", "finetune.survival is only supported"),
    ],
)
def test_config_rejects_inactive_task_label_blocks(
    tmp_path: Path,
    payload_factory,
    inactive_block: str,
    message: str,
):
    payload = payload_factory(tmp_path)
    payload["finetune"][inactive_block] = {}
    config = _write_yaml(tmp_path / "mixed-task-labels.yaml", payload)

    with pytest.raises(ValueError, match=message):
        load_config(config)


@pytest.mark.parametrize("field", ["class_weights", "pos_weigth"])
def test_multilabel_loss_rejects_unsupported_fields(tmp_path: Path, field: str):
    payload = _multilabel_payload(tmp_path)
    payload["finetune"]["loss"] = {field: [1.0, 2.0]}
    config = _write_yaml(tmp_path / "bad-loss.yaml", payload)

    with pytest.raises(ValueError, match="finetune.loss has unsupported fields"):
        load_config(config)


@pytest.mark.parametrize("pos_weight", [0.0, -1.0, [1.0, 0.0]])
def test_multilabel_loss_rejects_non_positive_pos_weight(tmp_path: Path, pos_weight):
    payload = _multilabel_payload(tmp_path)
    payload["finetune"]["loss"] = {"pos_weight": pos_weight}
    config = _write_yaml(tmp_path / "bad-pos-weight.yaml", payload)

    with pytest.raises(ValueError, match="pos_weight must contain only positive numbers"):
        load_config(config)


def test_multilabel_loss_rejects_pos_weight_length_mismatch(tmp_path: Path):
    payload = _multilabel_payload(tmp_path)
    payload["finetune"]["loss"] = {"pos_weight": [1.0]}
    config = _write_yaml(tmp_path / "bad-pos-weight-length.yaml", payload)

    with pytest.raises(ValueError, match="pos_weight length must match"):
        load_config(config)


@pytest.mark.parametrize(
    ("pos_weight", "expected"),
    [
        (2, 2.0),
        ([1, 2.5], [1.0, 2.5]),
    ],
)
def test_multilabel_loss_accepts_valid_pos_weight(tmp_path: Path, pos_weight, expected):
    payload = _multilabel_payload(tmp_path)
    payload["finetune"]["loss"] = {"pos_weight": pos_weight}
    config = _write_yaml(tmp_path / "good-pos-weight.yaml", payload)

    cfg = load_config(config)

    assert cfg.finetune.loss.pos_weight == expected


@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: payload["finetune"]["task"].update({"type": "regression"}),
        lambda payload: payload["finetune"]["task"].update({"is_seq": True}),
    ],
)
def test_invalid_semantics_fail(tmp_path: Path, mutate):
    payload = _cox_payload(tmp_path)
    mutate(payload)
    config = _write_yaml(tmp_path / "bad.yaml", payload)

    with pytest.raises(ValueError):
        load_config(config)


@pytest.mark.parametrize(
    "data",
    [
        {"finetune_data_index": None},
        {"finetune_data_index": None, "finetune_preset_path": "preset.pkl"},
        {"backend": "kaldi", "finetune_data_index": None, "kaldi_data_root": "/k", "kaldi_manifest": "/k/m.json"},
        {"train_dataset_names": ["shhs"], "test_dataset_names": None},
    ],
)
def test_data_block_uses_the_signal_spellings_and_leaves_the_source_to_the_loader(tmp_path: Path, data: dict):
    payload = _cox_payload(tmp_path)
    payload["data"].update(data)

    cfg = load_config(_write_yaml(tmp_path / "data.yaml", payload))

    for key, value in data.items():
        assert getattr(cfg.data, key) == value


@pytest.mark.parametrize(
    ("data", "match"),
    [
        ({"backend": "bad"}, "data.backend"),
        # The private participant-index semantics are gone: the split column is fixed and rows always collapse by key.
        ({"split_column": "split"}, "unsupported fields"),
        ({"key_column": "eid"}, "unsupported fields"),
        ({"deduplicate_by_key": True}, "unsupported fields"),
        ({"train_dataset_names": "shhs"}, "train_dataset_names"),
    ],
)
def test_data_block_rejects_invalid_and_removed_fields(tmp_path: Path, data: dict, match: str):
    payload = _cox_payload(tmp_path)
    payload["data"].update(data)

    with pytest.raises(ValueError, match=match):
        load_config(_write_yaml(tmp_path / "bad_data.yaml", payload))
