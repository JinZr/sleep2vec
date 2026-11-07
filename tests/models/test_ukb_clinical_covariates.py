from __future__ import annotations

from argparse import Namespace
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
import yaml

from data.psg_pretrain_dataset import PSGPretrainDataset
from sex_age_baseline.config import covariate_task_config, load_config
from sex_age_baseline.data import load_split_dataset, make_dataloader
from sex_age_baseline.model import SexAgeMLP
from sex_age_baseline.runtime import _batch_loss, load_checkpoint, save_checkpoint
from sleep2vec2.common import apply_finetune_config
from sleep2vec2.config import load_finetune_config
from sleep2vec2.sleep2vec_finetuning import Sleep2vecFinetuning
from sleep2vec2.utils import _build_finetune_loader

CONFIGS = Path(__file__).resolve().parents[2] / "configs"
EXAMPLES = {
    "baseline": CONFIGS / "sex_age_baseline" / "ukb_clinical_cox.yaml",
    "joint": CONFIGS / "sleep2vec2" / "ukb_joint_clinical_cox.yaml",
}
OUTPUT_DIM = 624
CONTINUOUS = {
    "age": 60.0,
    "bmi": 25.0,
    "sbp": 125.0,
    "dbp": 80.0,
    "total_cholesterol": 5.0,
    "hdl_cholesterol": 1.5,
    "ldl_cholesterol": 3.0,
    "hba1c": 35.0,
}
CATEGORY_COUNTS = {
    "smoking_status": 4,
    "diabetes_history": 3,
    "antihypertensive_use": 3,
    "lipid_lowering_use": 3,
    "parental_cvd_history": 3,
    "ethnicity": 5,
    "alcohol_frequency": 7,
}
ORDER = (
    "age",
    "sex",
    "bmi",
    "bmi_missing",
    *list(CONTINUOUS)[2:],
    *CATEGORY_COUNTS,
    *(f"{name}_missing" for name in CONTINUOUS if name != "bmi"),
)


def _write_yaml(path, payload):
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    return path


@pytest.mark.parametrize(("arm", "loader"), [("baseline", load_config), ("joint", load_finetune_config)])
def test_checked_in_ukb_cox_examples_parse(arm, loader):
    cfg = loader(EXAMPLES[arm])
    task = cfg.finetune.survival

    assert cfg.finetune.task.output_dim == OUTPUT_DIM
    assert set(task.covariates) == set(ORDER)
    assert task.covariate_embedding_dim == 16
    if arm == "baseline":
        assert cfg.model.head.hidden_dim == 256
        assert cfg.model.head.kwargs["num_layers"] == 3
    else:
        assert cfg.finetune.layer_mix.enabled
        assert cfg.model.head.temporal_agg.name == "mean"
        assert task.covariate_fusion == "feature_concat"


@pytest.fixture
def clinical_payloads():
    payloads = {arm: yaml.safe_load(path.read_text()) for arm, path in EXAMPLES.items()}
    for payload in payloads.values():
        task = payload["finetune"]["survival"]
        fold = task["fold_preprocessing"]
        # Use the artifact's ID ordering even when the unknown ID is zero.
        smoking = fold["categorical"]["smoking_status"]
        smoking.remove("unknown")
        smoking.insert(0, "unknown")
        fold["continuous"] = {
            name: {"median": center, "mean": center - 0.5, "std": 2.0} for name, center in CONTINUOUS.items()
        }
        task["covariates"] = list(reversed(ORDER))
        payload["model"]["head"]["dropout"] = 0.0

    joint = payloads["joint"]
    joint.pop("model_averaging", None)
    joint["data"].update(max_tokens=2, data_channel_names=["ppg"])
    joint["model"]["backbone"] = {
        "name": "roformer",
        "hidden_size": 16,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "vocab_size": 1,
        "config_overrides": {"intermediate_size": 32, "max_position_embeddings": 8},
    }
    joint["model"]["channels"] = [{"name": "ppg", "input_dim": 4, "tokenizer": {"name": "linear", "out_dim": 16}}]
    joint["finetune"]["layer_mix"] = {"enabled": True, "shared_across_modalities": False, "layer_indices": [1, 2]}
    return payloads


@pytest.fixture
def clinical_data(tmp_path, clinical_payloads):
    def create(task_type):
        block = "survival" if task_type == "survival" else "multilabel"
        eids = ["1001", "1002", "1003"]
        fold = clinical_payloads["baseline"]["finetune"]["survival"]["fold_preprocessing"]
        rows = []
        for i, eid in enumerate(eids):
            signal = tmp_path / f"{eid}.npz"
            np.savez(signal, ppg=np.arange(8, dtype=np.float32) + i)
            row = {"eid": eid, "split": "train", "path": str(signal), "duration": 60, "sex": i % 2}
            for j, (name, center) in enumerate(CONTINUOUS.items()):
                missing = i == 1 + j % 2
                row[name] = center if missing else center + i + 0.125
                row[f"{name}_missing"] = int(missing)
            for name, categories in fold["categorical"].items():
                unknown = categories.index("unknown")
                row[name] = (next(j for j in range(len(categories)) if j != unknown), unknown, len(categories) - 1)[i]
            rows.append(row)
        duplicate_path = tmp_path / "1002-night2.npz"
        np.savez(duplicate_path, ppg=np.arange(8, dtype=np.float32) + 4)
        rows.append({**rows[1], "path": str(duplicate_path)})
        frame = pd.DataFrame(rows)
        index = tmp_path / "index.csv"
        frame.to_csv(index, index=False)

        columns = [f"disease_{i:03d}" for i in range(OUTPUT_DIM)]
        column_index = tmp_path / "columns.txt"
        column_index.write_text("\n".join(columns) + "\n")
        fields = {"has_label_index": [1, 1, 1]}
        if task_type == "survival":
            fields.update(event_time_index=[10, 20, 30], is_event_index=[1, 1, 0])
        else:
            fields["label_index"] = [1, 0, 1]
        sidecars = {"disease_columns_index": str(column_index)}
        for name, values in fields.items():
            target = tmp_path / f"{name}.csv"
            pd.DataFrame(
                [[eid, *([value] * OUTPUT_DIM)] for eid, value in zip(eids, values)], columns=["eid", *columns]
            ).to_csv(target, index=False)
            sidecars[name] = str(target)

        for arm, payload in clinical_payloads.items():
            payload["data"].update(finetune_data_index=str(index), finetune_preset_path=None)
            finetune = payload["finetune"]
            if block == "multilabel":
                finetune[block] = finetune.pop("survival")
                for name in ("event_time_index", "is_event_index"):
                    finetune[block].pop(name)
                finetune["task"].update(type=task_type, monitor="val_macro_auroc", monitor_mod="max")
                if arm == "joint":
                    payload["model"]["head"].update(name="classification", act="elu")
                else:
                    finetune["loss"] = {"pos_weight": None}
            finetune[block].update(sidecars)
        return block, rows, index

    return create


def _load_arms(tmp_path, payloads):
    cfg = load_config(_write_yaml(tmp_path / "baseline.yaml", payloads["baseline"]))
    args = Namespace(
        config=_write_yaml(tmp_path / "joint.yaml", payloads["joint"]),
        label_name="clinical",
        batch_size=4,
        num_workers=0,
        device="cpu",
        pretrained_backbone_path=None,
    )
    bundle, _ = apply_finetune_config(args)
    return cfg, args, bundle


def _generate_preset(path, args):
    PSGPretrainDataset(
        channel_names=["ppg"],
        channel_input_dims={"ppg": 4},
        index=str(args.finetune_data_index),
        split=["train"],
        max_tokens=2,
        mask_rate=0.0,
        randomly_select_channels=False,
        min_channels=1,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        survival_label_config=args.survival,
        survival_output_dim=args.output_dim if args.is_survival else None,
        multilabel_label_config=args.multilabel,
        multilabel_output_dim=args.output_dim if args.is_multilabel else None,
        save_preset_path=str(path),
        load_preset_path=None,
    )


def _load_joint_data(args):
    return _build_finetune_loader(args, split=["train"], sources=[], shuffle=False, is_train_set=True)


def _embedding_inputs(embeddings):
    seen = {}
    handles = [
        embedding.register_forward_pre_hook(
            lambda module, inputs, name=name: seen.__setitem__(name, inputs[0].detach().reshape(-1).cpu())
        )
        for name, embedding in embeddings.items()
    ]
    return seen, handles


@pytest.mark.parametrize("task_type", ["survival", "multilabel_classification"])
def test_clinical_arms_preserve_prepared_values_and_reload_predictions(
    tmp_path, clinical_payloads, clinical_data, task_type
):
    block, rows, _ = clinical_data(task_type)
    cfg, args, bundle = _load_arms(tmp_path, clinical_payloads)
    preset = tmp_path / "train.pkl"
    _generate_preset(preset, args)
    fold = clinical_payloads["baseline"]["finetune"][block]["fold_preprocessing"]
    assert fold == clinical_payloads["joint"]["finetune"][block]["fold_preprocessing"]
    assert {name: len(labels) for name, labels in fold["categorical"].items()} == CATEGORY_COUNTS
    assert rows[1]["smoking_status"] == 0
    assert rows[1]["diabetes_history"] != fold["categorical"]["diabetes_history"].index("no")

    for preset_path in (None, str(preset)):
        args.finetune_preset_path = preset_path
        baseline_cfg = replace(cfg, data=replace(cfg.data, finetune_preset_path=preset_path))
        baseline_data = load_split_dataset(baseline_cfg, "train", sources=None)
        baseline_batch = next(iter(make_dataloader(baseline_data, batch_size=4, num_workers=0, shuffle=False)))
        joint_loader = _load_joint_data(args)
        joint_batch = next(iter(joint_loader))
        joint_meta = joint_batch["metadata"]
        baseline_meta = baseline_batch["metadata"]
        assert baseline_batch["key"] == [row["eid"] for row in rows[:3]]
        assert sorted(joint_meta["eid"]) == sorted(row["eid"] for row in rows)
        by_path = {row["path"]: row for row in rows}
        for sample in joint_loader.dataset.data:
            for name in ORDER:
                assert sample.metadata[name] == by_path[sample.metadata["path"]][name]
        baseline_indices = [baseline_batch["key"].index(eid) for eid in joint_meta["eid"]]
        for name in ORDER:
            expected = torch.tensor(
                [row[name] for row in rows[:3]], dtype=torch.float32 if name in CONTINUOUS else torch.long
            )
            torch.testing.assert_close(baseline_meta[name], expected)
            torch.testing.assert_close(joint_meta[name], expected[baseline_indices])

    baseline = SexAgeMLP(cfg)
    joint = Sleep2vecFinetuning(args, bundle.model, finetune_config=bundle.finetune)
    assert tuple(baseline.embeddings) == ORDER
    assert joint.model.head.extra_feature_dim == len(ORDER) * 16
    assert joint.model.layer_mix is not None
    joint_embeddings = {name: getattr(joint.model, f"survival_{name}_embedding") for name in ORDER}
    baseline_seen, baseline_hooks = _embedding_inputs(baseline.embeddings)
    joint_seen, joint_hooks = _embedding_inputs(joint_embeddings)

    for model, embeddings, logits, labels, batch_size in (
        (baseline, baseline.embeddings, baseline(baseline_meta), baseline_batch, 3),
        (joint, joint_embeddings, joint.model(joint_batch), joint_meta, 4),
    ):
        assert logits.shape == (batch_size, OUTPUT_DIM)
        assert all(not embedding.weight.any() for embedding in embeddings.values())
        for name, count in {**CATEGORY_COUNTS, "sex": 2}.items():
            assert isinstance(embeddings[name], nn.Embedding)
            assert embeddings[name].weight.shape == (count, 16)
        loss = _batch_loss(logits, labels, cfg)
        assert loss.isfinite()
        loss.backward()
        for name, embedding in embeddings.items():
            gradient = embedding.weight.grad
            assert gradient is not None and gradient.isfinite().all() and gradient.any(), name
            if name in CATEGORY_COUNTS:
                unknown_id = fold["categorical"][name].index("unknown")
                assert gradient[unknown_id].any(), name
        torch.optim.SGD(model.parameters(), lr=0.1).step()

    for hook in baseline_hooks + joint_hooks:
        hook.remove()
    assert tuple(baseline_seen) == tuple(joint_seen) == ORDER
    for name in ORDER:
        expected = baseline_meta[name]
        if name in CONTINUOUS:
            stats = fold["continuous"][name]
            expected = (expected - stats["mean"]) / stats["std"]
        torch.testing.assert_close(baseline_seen[name], expected)
        torch.testing.assert_close(joint_seen[name], expected[baseline_indices])

    baseline_path = tmp_path / "clinical.ckpt"
    save_checkpoint(baseline_path, baseline, cfg, epoch=0, global_step=1, metrics={})
    restored_baseline = SexAgeMLP(cfg)
    saved = load_checkpoint(restored_baseline, baseline_path, device=torch.device("cpu"), cfg=cfg)
    assert saved["config"]["finetune"][block]["fold_preprocessing"] == fold

    joint_path = tmp_path / "joint-clinical.ckpt"
    checkpoint = {"state_dict": joint.state_dict()}
    joint.on_save_checkpoint(checkpoint)
    torch.save(checkpoint, joint_path)
    saved = torch.load(joint_path, map_location="cpu", weights_only=False)
    assert saved["finetune_config"][block]["fold_preprocessing"] == fold
    restored_joint = Sleep2vecFinetuning(args, bundle.model, finetune_config=bundle.finetune)
    restored_joint.on_load_checkpoint(saved)
    restored_joint.load_state_dict(saved["state_dict"])

    with torch.no_grad():
        torch.testing.assert_close(restored_baseline.eval()(baseline_meta), baseline.eval()(baseline_meta))
        torch.testing.assert_close(restored_joint.eval().model(joint_batch), joint.eval().model(joint_batch))

    changed_baseline_cfg = deepcopy(cfg)
    covariate_task_config(changed_baseline_cfg).fold_preprocessing["continuous"]["age"]["mean"] += 1
    with pytest.raises(ValueError, match="model contract"):
        load_checkpoint(restored_baseline, baseline_path, device=torch.device("cpu"), cfg=changed_baseline_cfg)
    saved["finetune_config"][block]["fold_preprocessing"]["continuous"]["age"]["mean"] += 1
    with pytest.raises(ValueError, match="fold_preprocessing"):
        restored_joint.on_load_checkpoint(saved)


def test_legacy_age_sex_bmi_checkpoint_loads_without_fold_preprocessing(tmp_path, clinical_payloads, clinical_data):
    clinical_data("survival")
    for payload in clinical_payloads.values():
        task = payload["finetune"]["survival"]
        task.pop("fold_preprocessing")
        task.update(
            covariates=["age", "sex", "bmi", "bmi_missing"],
            covariate_normalization={"bmi": {"mean": 25.0, "std": 2.0}},
        )
    cfg, args, bundle = _load_arms(tmp_path, clinical_payloads)
    batch = next(iter(_load_joint_data(args)))

    baseline = SexAgeMLP(cfg).eval()
    baseline_path = tmp_path / "old-clinical.ckpt"
    save_checkpoint(baseline_path, baseline, cfg, epoch=0, global_step=0, metrics={})
    checkpoint = torch.load(baseline_path, weights_only=False)
    checkpoint["config"]["finetune"]["survival"].pop("fold_preprocessing", None)
    checkpoint["model_contract"].pop("fold_preprocessing", None)
    torch.save(checkpoint, baseline_path)
    restored_baseline = SexAgeMLP(cfg).eval()
    load_checkpoint(restored_baseline, baseline_path, device=torch.device("cpu"), cfg=cfg)

    joint = Sleep2vecFinetuning(args, bundle.model, finetune_config=bundle.finetune).eval()
    joint_path = tmp_path / "old-joint.ckpt"
    checkpoint = {"state_dict": joint.state_dict()}
    joint.on_save_checkpoint(checkpoint)
    checkpoint["finetune_config"]["survival"].pop("fold_preprocessing", None)
    checkpoint["finetune_config_yaml"] = yaml.safe_dump(checkpoint["finetune_config"])
    torch.save(checkpoint, joint_path)
    init_args = Namespace(**vars(args), init_checkpoint_path=str(joint_path))
    restored_joint = Sleep2vecFinetuning(init_args, bundle.model, finetune_config=bundle.finetune).eval()
    restored_joint.on_load_checkpoint(checkpoint)

    with torch.no_grad():
        torch.testing.assert_close(restored_baseline(batch["metadata"]), baseline(batch["metadata"]))
        torch.testing.assert_close(restored_joint.model(batch), joint.model(batch))


@pytest.mark.parametrize("use_preset", [False, True], ids=["index", "preset"])
@pytest.mark.parametrize(
    ("name", "value", "row_idx"),
    [
        ("parental_cvd_history", None, 0),
        ("sbp", float("nan"), 0),
        ("ethnicity", 5, 0),
        ("age_missing", 0.5, 0),
        ("smoking_status", 1, 3),
    ],
)
def test_invalid_prepared_covariates_fail_in_both_arms(
    tmp_path, clinical_payloads, clinical_data, use_preset, name, value, row_idx
):
    _, rows, index = clinical_data("survival")
    cfg, args, _ = _load_arms(tmp_path, clinical_payloads)
    if use_preset:
        preset = tmp_path / "train.pkl"
        _generate_preset(preset, args)
        samples = pickle.loads(preset.read_bytes())
        sample = next(sample for sample in samples if sample.metadata["path"] == rows[row_idx]["path"])
        if value is None:
            sample.metadata.pop(name)
        else:
            sample.metadata[name] = value
        preset.write_bytes(pickle.dumps(samples))
        args.finetune_preset_path = str(preset)
        cfg = replace(cfg, data=replace(cfg.data, finetune_preset_path=str(preset)))
    else:
        if value is None:
            for row in rows:
                row.pop(name)
        else:
            rows[row_idx][name] = value
        pd.DataFrame(rows).to_csv(index, index=False)

    with pytest.raises(ValueError, match=name):
        load_split_dataset(cfg, "train", sources=None)
    with pytest.raises(ValueError, match=name):
        next(iter(_load_joint_data(args)))


@pytest.mark.parametrize(("arm", "loader"), [("baseline", load_config), ("joint", load_finetune_config)])
@pytest.mark.parametrize("invalid", ["no_fold", "no_unknown", "wrong_count", "zero_std"])
def test_clinical_config_rejects_incomplete_frozen_preprocessing(tmp_path, clinical_payloads, arm, loader, invalid):
    payload = clinical_payloads[arm]
    task = payload["finetune"]["survival"]
    fold = task["fold_preprocessing"]
    if invalid == "no_fold":
        task.pop("fold_preprocessing")
        message = "fold_preprocessing"
    elif invalid == "no_unknown":
        labels = fold["categorical"]["smoking_status"]
        labels[labels.index("unknown")] = "missing"
        message = "smoking_status"
    elif invalid == "wrong_count":
        fold["categorical"]["alcohol_frequency"].pop(0)
        message = "alcohol_frequency"
    else:
        fold["continuous"]["age"]["std"] = 0.0
        message = "age"
    with pytest.raises(ValueError, match=message):
        loader(_write_yaml(tmp_path / "invalid.yaml", payload))


@pytest.mark.parametrize("fusion", ["risk", "token_concat"])
def test_clinical_config_rejects_other_fusion_paths(tmp_path, clinical_payloads, fusion):
    payload = clinical_payloads["joint"]
    payload["finetune"]["survival"]["covariate_fusion"] = fusion

    with pytest.raises(ValueError, match="feature_concat"):
        load_finetune_config(_write_yaml(tmp_path / "invalid.yaml", payload))
