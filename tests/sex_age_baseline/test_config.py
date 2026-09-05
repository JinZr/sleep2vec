from __future__ import annotations

from itertools import combinations
from pathlib import Path

import pytest
import yaml

from sex_age_baseline.config import load_config


def _write_yaml(path: Path, payload: dict) -> Path:
    path.write_text(yaml.safe_dump(payload))
    return path


def _cox_payload(tmp_path: Path) -> dict:
    sidecars = _write_survival_sidecars(tmp_path)
    return {
        "model": {
            "name": "sex_age_mlp",
            "features": ["age", "sex"],
            "age": {"transform": "divide", "scale": 100.0, "embedding_dim": 4, "initialization": "default"},
            "sex": {"encoding": "binary", "embedding_dim": 4, "initialization": "default"},
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
            "split_column": "split",
            "key_column": "eid",
            "deduplicate_by_key": True,
        },
        "finetune": {
            "task": {
                "type": "survival",
                "output_dim": 2,
                "is_seq": False,
                "monitor": "val_c_index",
                "monitor_mod": "max",
            },
            "survival": {"key_column": "eid", **sidecars},
        },
        "outputs": {"prediction_csv": True, "per_disease_metrics_csv": True},
    }


def _multilabel_payload(tmp_path: Path) -> dict:
    sidecars = _write_multilabel_sidecars(tmp_path)
    payload = _cox_payload(tmp_path)
    payload["finetune"] = {
        "task": {
            "type": "multilabel_classification",
            "output_dim": 2,
            "is_seq": False,
            "monitor": "val_macro_auroc",
            "monitor_mod": "max",
        },
        "multilabel": {"key_column": "eid", **sidecars},
        "loss": {"pos_weight": None},
    }
    return payload


def _write_survival_sidecars(tmp_path: Path) -> dict[str, str]:
    disease_columns = tmp_path / "disease_columns.txt"
    event_time = tmp_path / "event_time.csv"
    is_event = tmp_path / "is_event.csv"
    has_label = tmp_path / "has_label.csv"
    disease_columns.write_text("d1\nd2\n")
    header = "eid,d1,d2\n"
    event_time.write_text(header + "001,5,6\n002,3,4\n")
    is_event.write_text(header + "001,1,0\n002,0,1\n")
    has_label.write_text(header + "001,1,1\n002,1,1\n")
    return {
        "disease_columns_index": str(disease_columns),
        "event_time_index": str(event_time),
        "is_event_index": str(is_event),
        "has_label_index": str(has_label),
    }


def _write_multilabel_sidecars(tmp_path: Path) -> dict[str, str]:
    disease_columns = tmp_path / "disease_columns.txt"
    label_index = tmp_path / "disease_label.csv"
    has_label = tmp_path / "has_label.csv"
    disease_columns.write_text("d1\nd2\n")
    header = "eid,d1,d2\n"
    label_index.write_text(header + "001,1,0\n002,0,1\n")
    has_label.write_text(header + "001,1,1\n002,1,1\n")
    return {
        "disease_columns_index": str(disease_columns),
        "label_index": str(label_index),
        "has_label_index": str(has_label),
    }


@pytest.mark.parametrize("path", ["configs/sex_age_baseline/cox.yaml", "configs/sex_age_baseline/multilabel.yaml"])
def test_checked_in_configs_load(path: str):
    cfg = load_config(path)

    assert cfg.model.features == ["age", "sex"]
    assert cfg.data.key_column == "eid"
    assert cfg.data.backend == "npz"


@pytest.mark.parametrize("features", [list(c) for n in (1, 2, 3) for c in combinations(("age", "sex", "bmi"), n)])
@pytest.mark.parametrize("num_layers", [1, 2, 3])
@pytest.mark.parametrize("variant", ["sleep2vec", "sleep2vec2"])
def test_feature_subsets_and_dense_head_match_shared_builder(tmp_path, features, num_layers, variant):
    import importlib

    import torch
    from torch import nn

    from sex_age_baseline.config import validate_model_config
    from sex_age_baseline.model import SexAgeMLP

    ClassificationHead = importlib.import_module(f"{variant}.downstreams.heads.classification").ClassificationHead

    payload = _cox_payload(tmp_path)
    model = payload["model"]
    model["bmi"] = dict(model["age"], scale=40.0)
    model["features"] = list(reversed(features))
    for feature in set(("age", "sex", "bmi")) - set(features):
        del model[feature]
    model["head"]["kwargs"]["num_layers"] = num_layers
    cfg = load_config(_write_yaml(tmp_path / "subset.yaml", payload))
    network = SexAgeMLP(cfg).eval()
    values = {"age": torch.tensor([40.0, 60.0]), "bmi": torch.tensor([20.0, 30.0]), "sex": torch.tensor([0, 1])}
    encoded = []
    for feature in cfg.model.features:
        value = (
            values[feature]
            if feature == "sex"
            else (values[feature] / getattr(cfg.model, feature).scale).reshape(-1, 1)
        )
        encoded.append(network.encoders[feature](value))
    reference = ClassificationHead(
        validate_model_config(cfg), 1, 2, agg="mean", hidden_dim=8, dropout=0.1, act=nn.ELU, num_layers=num_layers
    ).mlp.eval()
    reference.load_state_dict(network.head.state_dict())
    torch.testing.assert_close(network(values), reference(torch.cat(encoded, dim=-1)))
    assert set(network.encoders) == set(features)
    assert sum(isinstance(m, nn.Linear) for m in network.head) == num_layers


def test_zero_encoding_and_window_mode(tmp_path):
    import torch

    from sex_age_baseline.model import SexAgeMLP

    payload = _cox_payload(tmp_path)
    payload["data"]["deduplicate_by_key"] = False
    for feature in payload["model"]["features"]:
        payload["model"][feature]["initialization"] = "zeros"
    cfg = load_config(_write_yaml(tmp_path / "zero.yaml", payload))
    assert not cfg.data.deduplicate_by_key
    assert all(torch.count_nonzero(p) == 0 for p in SexAgeMLP(cfg).encoders.parameters())


@pytest.mark.parametrize("zero_feature", ["age", "sex", "bmi"])
@pytest.mark.parametrize("num_layers", [1, 2, 3])
@pytest.mark.parametrize("mixed", [False, True])
def test_relu_rejects_any_selected_zero_encoder(tmp_path, zero_feature, num_layers, mixed):
    payload = _cox_payload(tmp_path)
    model = payload["model"]
    model["bmi"] = dict(model["age"], scale=40.0)
    model["features"] = ["age", "sex", "bmi"] if mixed else [zero_feature]
    for feature in set(("age", "sex", "bmi")) - set(model["features"]):
        del model[feature]
    model[zero_feature]["initialization"] = "zeros"
    model["head"]["act"] = "relu"
    model["head"]["kwargs"]["num_layers"] = num_layers

    with pytest.raises(ValueError, match=f"blocks gradients.*{zero_feature}"):
        load_config(_write_yaml(tmp_path / "relu_zero.yaml", payload))


@pytest.mark.parametrize(
    "act,initialization", [("relu", "default"), ("elu", "zeros"), ("gelu", "zeros"), ("silu", "zeros")]
)
@pytest.mark.parametrize("num_layers", [1, 2, 3])
def test_compatible_initialization_activation_encoders_receive_gradients(tmp_path, act, initialization, num_layers):
    import torch

    from sex_age_baseline.model import SexAgeMLP

    payload = _cox_payload(tmp_path)
    model = payload["model"]
    model["features"] = ["age", "sex", "bmi"]
    model["bmi"] = dict(model["age"], scale=40.0)
    for feature in model["features"]:
        model[feature]["initialization"] = initialization
    model["head"].update(act=act, dropout=0.0)
    model["head"]["kwargs"]["num_layers"] = num_layers
    network = SexAgeMLP(load_config(_write_yaml(tmp_path / "compatible.yaml", payload)))
    with torch.no_grad():
        for parameter in network.head.parameters():
            parameter.fill_(0.1)
        if initialization == "default":
            for parameter in network.encoders.parameters():
                parameter.fill_(0.1)
    network(
        {"age": torch.tensor([40.0, 60.0]), "sex": torch.tensor([0, 1]), "bmi": torch.tensor([20.0, 30.0])}
    ).sum().backward()
    assert all(
        parameter.grad is not None and torch.count_nonzero(parameter.grad)
        for parameter in network.encoders.parameters()
    )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda m: m["age"].pop("initialization"),
        lambda m: m["age"].update(scale=float("nan")),
        lambda m: m["age"].update(initialization="other"),
        lambda m: m["head"].update(activation="elu"),
        lambda m: m["head"]["kwargs"].update(num_layers=4),
        lambda m: m.update(features=[]),
        lambda m: m.update(features=["age", "age"]),
        lambda m: m.update(features=["height"]),
    ],
)
def test_explicit_model_contract_rejects_invalid_fields(tmp_path, mutate):
    payload = _cox_payload(tmp_path)
    mutate(payload["model"])
    with pytest.raises(ValueError):
        load_config(_write_yaml(tmp_path / "invalid.yaml", payload))


def test_validates_sidecar_output_dim(tmp_path: Path):
    payload = _cox_payload(tmp_path)
    config = _write_yaml(tmp_path / "cox.yaml", payload)

    cfg = load_config(config, validate_sidecars=True)

    assert cfg.finetune.task.output_dim == 2


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


def test_model_rejects_runtime_learning_rate(tmp_path):
    payload = _cox_payload(tmp_path)
    payload["model"]["lr"] = 0.001
    with pytest.raises(ValueError, match="unsupported or inactive"):
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
        lambda payload: payload["model"].update({"features": ["age"]}),
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


def test_bad_output_dim_sidecar_mismatch_fails(tmp_path: Path):
    payload = _cox_payload(tmp_path)
    payload["finetune"]["task"]["output_dim"] = 3
    config = _write_yaml(tmp_path / "bad_dim.yaml", payload)

    with pytest.raises(ValueError, match="output_dim"):
        load_config(config, validate_sidecars=True)


def test_multilabel_config_validates_sidecars(tmp_path: Path):
    payload = _multilabel_payload(tmp_path)
    config = _write_yaml(tmp_path / "multilabel.yaml", payload)

    cfg = load_config(config, validate_sidecars=True)

    assert cfg.finetune.multilabel.label_index.endswith("disease_label.csv")


def test_npz_preset_config_loads(tmp_path: Path):
    payload = _cox_payload(tmp_path)
    payload["data"]["finetune_data_index"] = None
    payload["data"]["finetune_preset_path"] = str(tmp_path / "preset.pkl")
    config = _write_yaml(tmp_path / "preset.yaml", payload)

    cfg = load_config(config)

    assert cfg.data.backend == "npz"
    assert cfg.data.finetune_preset_path.endswith("preset.pkl")


def test_kaldi_config_loads(tmp_path: Path):
    payload = _cox_payload(tmp_path)
    payload["data"].update(
        {
            "backend": "kaldi",
            "finetune_data_index": None,
            "finetune_preset_path": None,
            "kaldi_data_root": str(tmp_path / "kaldi"),
            "kaldi_manifest": str(tmp_path / "kaldi" / "manifest.json"),
        }
    )
    config = _write_yaml(tmp_path / "kaldi.yaml", payload)

    cfg = load_config(config)

    assert cfg.data.backend == "kaldi"
    assert cfg.data.kaldi_manifest.endswith("manifest.json")


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda payload: payload["data"].update({"finetune_preset_path": "preset.pkl"}), "exactly one"),
        (lambda payload: payload["data"].update({"finetune_data_index": None}), "exactly one"),
        (lambda payload: payload["data"].update({"backend": "bad"}), "data.backend"),
        (
            lambda payload: payload["data"].update(
                {"backend": "kaldi", "finetune_data_index": None, "kaldi_data_root": None}
            ),
            "kaldi_data_root",
        ),
        (
            lambda payload: payload["data"].update(
                {
                    "backend": "kaldi",
                    "finetune_data_index": None,
                    "finetune_preset_path": "preset.pkl",
                    "kaldi_data_root": "/kaldi",
                    "kaldi_manifest": "/kaldi/manifest.json",
                }
            ),
            "must not set",
        ),
    ],
)
def test_backend_input_validation_fails(tmp_path: Path, mutate, match: str):
    payload = _cox_payload(tmp_path)
    mutate(payload)
    config = _write_yaml(tmp_path / "bad_backend.yaml", payload)

    with pytest.raises(ValueError, match=match):
        load_config(config)
