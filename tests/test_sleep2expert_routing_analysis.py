from __future__ import annotations

from argparse import Namespace
import csv
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pytorch_lightning")

from sleep2expert.backbones.roformer.moe import MoERoutingOutput
from sleep2expert.config import BackboneConfig, ChannelConfig, ModelConfig, MoeConfig, TokenizerConfig
import sleep2expert.routing_analysis as routing_analysis
from sleep2expert.routing_analysis import ROUTING_CSV_COLUMNS, build_routing_rows, run_routing_analysis


def _routing_aux(
    router_probs: torch.Tensor,
    *,
    layer_idx: int = 4,
    modality_name: str = "eeg",
    top_k: int = 2,
) -> MoERoutingOutput:
    topk_probs, topk_indices = torch.topk(router_probs, k=top_k, dim=-1)
    topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(router_probs.dtype).eps)
    expert_mask = torch.zeros_like(router_probs, dtype=torch.bool)
    expert_mask.scatter_(-1, topk_indices, True)
    load = expert_mask.float().sum(dim=(0, 1))
    importance = router_probs.sum(dim=(0, 1))
    entropy = -(router_probs * router_probs.clamp_min(torch.finfo(router_probs.dtype).eps).log()).sum(dim=-1)
    return MoERoutingOutput(
        router_logits=router_probs.clamp_min(torch.finfo(router_probs.dtype).eps).log(),
        router_probs=router_probs,
        topk_indices=topk_indices,
        topk_probs=topk_probs,
        expert_mask=expert_mask,
        load=load,
        importance=importance,
        z_loss=torch.tensor(0.0),
        entropy=entropy.mean(),
        modality_name=modality_name,
        layer_idx=layer_idx,
    )


def _expert_groups() -> dict[int, str]:
    moe_cfg = MoeConfig(
        enabled=True,
        layer_indices=[4],
        num_experts=3,
        top_k=2,
        expert_groups={"shared": [0, 1], "cardiac": [1, 2]},
    )
    return routing_analysis._build_expert_group_lookup(moe_cfg)


def _model_config(moe_cfg: MoeConfig) -> ModelConfig:
    return ModelConfig(
        channels=[ChannelConfig(name="eeg", input_dim=1, tokenizer=TokenizerConfig(name="linear", out_dim=1))],
        backbone=BackboneConfig(name="roformer", moe=moe_cfg),
    )


def test_build_routing_rows_groups_sequence_labels_and_excludes_cls_padding():
    router_probs = torch.tensor(
        [
            [
                [0.05, 0.10, 0.85],
                [0.70, 0.20, 0.10],
                [0.05, 0.80, 0.15],
                [0.90, 0.05, 0.05],
            ]
        ],
        dtype=torch.float32,
    )
    aux = _routing_aux(router_probs)
    batch = {
        "id": ["s1"],
        "length": torch.tensor([2]),
        "token_start": torch.tensor([30]),
        "tokens": {"stage5": torch.tensor([[0, 1, -1]])},
        "metadata": {"source": ["site-a"], "path": ["/tmp/s1.npz"]},
    }
    records = [{"modality": "eeg", "aux": (aux,), "attention_mask": torch.tensor([[1, 1, 1, 0]])}]
    args = Namespace(label_name="stage5", label_source_name="stage5", is_seq=True, is_multilabel=False)

    rows = build_routing_rows(records, batch, args, _expert_groups())

    observed = {(row["label_value_if_available"], row["expert_id"], row["usage_count"]) for row in rows}
    assert observed == {
        (0, 0, 1),
        (0, 1, 1),
        (1, 1, 1),
        (1, 2, 1),
    }
    assert all(row["sample_id"] == "s1" for row in rows)
    assert all(row["token_start"] == 30 for row in rows)
    assert {row["expert_group"] for row in rows} == {"shared", "cardiac|shared", "cardiac"}
    assert 2 not in {row["label_value_if_available"] for row in rows}
    assert -1 not in {row["label_value_if_available"] for row in rows}
    rows_by_label_expert = {(row["label_value_if_available"], row["expert_id"]): row for row in rows}
    assert rows_by_label_expert[(0, 0)]["mean_router_prob"] == pytest.approx(0.70)
    assert all(row["analysis_tag"] == "" for row in rows)
    assert all(row["split"] == "" for row in rows)


def test_build_routing_rows_groups_scalar_label_once_per_sample():
    router_probs = torch.tensor([[[0.90, 0.10], [0.80, 0.20]]], dtype=torch.float32)
    aux = _routing_aux(router_probs, top_k=1)
    batch = {
        "id": ["s2"],
        "length": torch.tensor([2]),
        "tokens": {"eeg": torch.zeros(1, 2, 1)},
        "metadata": {"age": torch.tensor([63.0]), "source": ["site-b"], "path": ["/tmp/s2.npz"]},
    }
    records = [{"modality": "eeg", "aux": (aux,), "attention_mask": torch.tensor([[1, 1]])}]
    args = Namespace(label_name="age", is_seq=False, is_multilabel=False)

    rows = build_routing_rows(records, batch, args, {0: "shared", 1: "cardiac"})

    assert rows == [
        {
            "sample_id": "s2",
            "source": "site-b",
            "path": "/tmp/s2.npz",
            "token_start": "",
            "modality": "eeg",
            "layer_idx": 4,
            "expert_id": 0,
            "expert_group": "shared",
            "usage_count": 2,
            "mean_router_prob": pytest.approx(0.85),
            "router_entropy": pytest.approx(float((-(router_probs * router_probs.log()).sum(dim=-1)).mean().item())),
            "label_name": "age",
            "label_value_if_available": 63.0,
            "analysis_tag": "",
            "split": "",
        }
    ]


def test_run_routing_analysis_writes_fixed_columns(monkeypatch, tmp_path):
    output = tmp_path / "routing.csv"
    router_probs = torch.tensor([[[0.90, 0.10], [0.20, 0.80]]], dtype=torch.float32)
    aux = _routing_aux(router_probs, top_k=1)
    batch = {
        "id": ["s3"],
        "length": torch.tensor([2]),
        "tokens": {"eeg": torch.zeros(1, 2, 1)},
        "metadata": {"age": torch.tensor([50.0]), "source": ["site-c"], "path": ["/tmp/s3.npz"]},
    }
    records = [{"modality": "eeg", "aux": (aux,), "attention_mask": torch.tensor([[1, 1]])}]
    moe_cfg = MoeConfig(enabled=True, layer_indices=[4], num_experts=2, top_k=1, expert_groups={"shared": [0, 1]})
    model_cfg = _model_config(moe_cfg)

    def fake_apply(args):
        args.is_seq = False
        args.is_multilabel = False
        return SimpleNamespace(finetune=None, averaging=None), model_cfg

    class DummyDownstream:
        def __init__(self, backbone, records_to_write):
            self.backbone = backbone
            self.records_to_write = records_to_write

        def __call__(self, batch):
            self.backbone.last_moe_aux = self.records_to_write
            return torch.zeros(1, 1)

    class DummyModule(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.backbone = SimpleNamespace(last_moe_aux=None)
            self.model = DummyDownstream(self.backbone, [])
            self.eval_backbone = SimpleNamespace(last_moe_aux=None)
            self.eval_model = DummyDownstream(self.eval_backbone, records)

        def load_state_dict(self, state_dict, strict=True):
            return [], []

        def _get_eval_model(self):
            return self.eval_model

    monkeypatch.setattr(routing_analysis, "apply_finetune_config", fake_apply)
    monkeypatch.setattr(routing_analysis, "_build_inference_loader", lambda args: [batch])
    monkeypatch.setattr(routing_analysis, "Sleep2vecFinetuning", DummyModule)
    monkeypatch.setattr(routing_analysis, "_load_analysis_weights", lambda module, args: None)

    rows = run_routing_analysis(
        Namespace(
            config=tmp_path / "config.yaml",
            ckpt_path=str(tmp_path / "model.ckpt"),
            label_name="age",
            output=output,
            batch_size=1,
            num_workers=0,
            device="cpu",
            eval_split="test",
            analysis_tag="",
            pretrained_only=False,
            override_dataset_names=None,
            avg_ckpts=1,
            avg_ckpt_dir=None,
            seed=1,
            lr=1e-6,
            weight_decay=0.0,
            pretrained_backbone_path=None,
        )
    )

    assert len(rows) == 2
    with output.open(newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        assert reader.fieldnames == ROUTING_CSV_COLUMNS
        written_rows = list(reader)

    assert len(written_rows) == 2
    assert {row["expert_id"] for row in written_rows} == {"0", "1"}
    assert all(row["label_name"] == "age" for row in written_rows)
    assert all(row["label_value_if_available"] == "50.0" for row in written_rows)
    assert all(row["analysis_tag"] == "" for row in written_rows)
    assert all(row["split"] == "test" for row in written_rows)


def test_run_routing_analysis_writes_analysis_tag_and_split_columns(monkeypatch, tmp_path):
    output = tmp_path / "routing.csv"
    router_probs = torch.tensor([[[0.90, 0.10], [0.20, 0.80]]], dtype=torch.float32)
    aux = _routing_aux(router_probs, top_k=1)
    batch = {
        "id": ["s4"],
        "length": torch.tensor([2]),
        "tokens": {"eeg": torch.zeros(1, 2, 1)},
        "metadata": {"age": torch.tensor([51.0]), "source": ["site-d"], "path": ["/tmp/s4.npz"]},
    }
    records = [{"modality": "eeg", "aux": (aux,), "attention_mask": torch.tensor([[1, 1]])}]
    moe_cfg = MoeConfig(enabled=True, layer_indices=[4], num_experts=2, top_k=1, expert_groups={"shared": [0, 1]})
    model_cfg = _model_config(moe_cfg)

    def fake_apply(args):
        args.is_seq = False
        args.is_multilabel = False
        return SimpleNamespace(finetune=None, averaging=None), model_cfg

    class DummyDownstream:
        def __init__(self, backbone):
            self.backbone = backbone

        def __call__(self, batch):
            self.backbone.last_moe_aux = records
            return torch.zeros(1, 1)

    class DummyModule(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.eval_backbone = SimpleNamespace(last_moe_aux=None)
            self.eval_model = DummyDownstream(self.eval_backbone)

        def load_state_dict(self, state_dict, strict=True):
            return [], []

        def _get_eval_model(self):
            return self.eval_model

    monkeypatch.setattr(routing_analysis, "apply_finetune_config", fake_apply)
    monkeypatch.setattr(routing_analysis, "_build_inference_loader", lambda args: [batch])
    monkeypatch.setattr(routing_analysis, "Sleep2vecFinetuning", DummyModule)
    monkeypatch.setattr(routing_analysis, "_load_analysis_weights", lambda module, args: None)

    rows = run_routing_analysis(
        Namespace(
            config=tmp_path / "config.yaml",
            ckpt_path=str(tmp_path / "model.ckpt"),
            label_name="age",
            output=output,
            batch_size=1,
            num_workers=0,
            device="cpu",
            eval_split="val",
            analysis_tag="post_finetune",
            pretrained_only=False,
            override_dataset_names=None,
            avg_ckpts=1,
            avg_ckpt_dir=None,
            seed=1,
            lr=1e-6,
            weight_decay=0.0,
            pretrained_backbone_path=None,
        )
    )

    assert {row["analysis_tag"] for row in rows} == {"post_finetune"}
    assert {row["split"] for row in rows} == {"val"}
    with output.open(newline="") as file_obj:
        written_rows = list(csv.DictReader(file_obj))
    assert {row["analysis_tag"] for row in written_rows} == {"post_finetune"}
    assert {row["split"] for row in written_rows} == {"val"}


def test_pretrained_only_skips_downstream_checkpoint_loading(monkeypatch, tmp_path):
    output = tmp_path / "routing.csv"
    router_probs = torch.tensor([[[0.90, 0.10], [0.20, 0.80]]], dtype=torch.float32)
    aux = _routing_aux(router_probs, top_k=1)
    batch = {
        "id": ["s5"],
        "length": torch.tensor([2]),
        "tokens": {"eeg": torch.zeros(1, 2, 1)},
        "metadata": {"age": torch.tensor([52.0]), "source": ["site-e"], "path": ["/tmp/s5.npz"]},
    }
    records = [{"modality": "eeg", "aux": (aux,), "attention_mask": torch.tensor([[1, 1]])}]
    moe_cfg = MoeConfig(enabled=True, layer_indices=[4], num_experts=2, top_k=1, expert_groups={"shared": [0, 1]})
    model_cfg = _model_config(moe_cfg)

    def fake_apply(args):
        args.is_seq = False
        args.is_multilabel = False
        return SimpleNamespace(finetune=None, averaging=None), model_cfg

    class DummyDownstream:
        def __init__(self, backbone):
            self.backbone = backbone

        def __call__(self, batch):
            self.backbone.last_moe_aux = records
            return torch.zeros(1, 1)

    class DummyModule(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.eval_backbone = SimpleNamespace(last_moe_aux=None)
            self.eval_model = DummyDownstream(self.eval_backbone)

        def _get_eval_model(self):
            return self.eval_model

    def fail_load(*args, **kwargs):
        raise AssertionError("_load_analysis_weights should not run in pretrained-only mode")

    monkeypatch.setattr(routing_analysis, "apply_finetune_config", fake_apply)
    monkeypatch.setattr(routing_analysis, "_build_inference_loader", lambda args: [batch])
    monkeypatch.setattr(routing_analysis, "Sleep2vecFinetuning", DummyModule)
    monkeypatch.setattr(routing_analysis, "_load_analysis_weights", fail_load)
    pretrained_path = tmp_path / "pretrain.ckpt"
    pretrained_path.touch()

    rows = run_routing_analysis(
        Namespace(
            config=tmp_path / "config.yaml",
            ckpt_path=None,
            label_name="age",
            output=output,
            batch_size=1,
            num_workers=0,
            device="cpu",
            eval_split="test",
            analysis_tag="pre_finetune",
            pretrained_only=True,
            override_dataset_names=None,
            avg_ckpts=1,
            avg_ckpt_dir=None,
            seed=1,
            lr=1e-6,
            weight_decay=0.0,
            pretrained_backbone_path=str(pretrained_path),
        )
    )

    assert len(rows) == 2
    assert {row["analysis_tag"] for row in rows} == {"pre_finetune"}


def test_pretrained_only_requires_pretrained_backbone_path(tmp_path):
    with pytest.raises(ValueError, match="--pretrained-only requires --pretrained-backbone-path"):
        run_routing_analysis(
            Namespace(
                config=tmp_path / "config.yaml",
                ckpt_path=None,
                label_name="age",
                output=tmp_path / "routing.csv",
                batch_size=1,
                num_workers=0,
                device="cpu",
                eval_split="test",
                analysis_tag="pre_finetune",
                pretrained_only=True,
                override_dataset_names=None,
                avg_ckpts=1,
                avg_ckpt_dir=None,
                seed=1,
                lr=1e-6,
                weight_decay=0.0,
                pretrained_backbone_path=None,
            )
        )


def test_pretrained_only_rejects_ckpt_path(tmp_path):
    pretrained_path = tmp_path / "pretrain.ckpt"
    pretrained_path.touch()

    with pytest.raises(ValueError, match="cannot be combined with --ckpt-path"):
        run_routing_analysis(
            Namespace(
                config=tmp_path / "config.yaml",
                ckpt_path=str(tmp_path / "finetuned.ckpt"),
                label_name="age",
                output=tmp_path / "routing.csv",
                batch_size=1,
                num_workers=0,
                device="cpu",
                eval_split="test",
                analysis_tag="pre_finetune",
                pretrained_only=True,
                override_dataset_names=None,
                avg_ckpts=1,
                avg_ckpt_dir=None,
                seed=1,
                lr=1e-6,
                weight_decay=0.0,
                pretrained_backbone_path=str(pretrained_path),
            )
        )


def test_pretrained_only_requires_existing_pretrained_backbone_path(tmp_path):
    with pytest.raises(FileNotFoundError, match="Pretrained backbone checkpoint not found"):
        run_routing_analysis(
            Namespace(
                config=tmp_path / "config.yaml",
                ckpt_path=None,
                label_name="age",
                output=tmp_path / "routing.csv",
                batch_size=1,
                num_workers=0,
                device="cpu",
                eval_split="test",
                analysis_tag="pre_finetune",
                pretrained_only=True,
                override_dataset_names=None,
                avg_ckpts=1,
                avg_ckpt_dir=None,
                seed=1,
                lr=1e-6,
                weight_decay=0.0,
                pretrained_backbone_path=str(tmp_path / "missing.ckpt"),
            )
        )


def test_parse_args_accepts_split_alias(monkeypatch, tmp_path):
    monkeypatch.setattr(
        routing_analysis.sys,
        "argv",
        [
            "routing_analysis.py",
            "--config",
            str(tmp_path / "config.yaml"),
            "--ckpt-path",
            str(tmp_path / "model.ckpt"),
            "--label-name",
            "age",
            "--output",
            str(tmp_path / "routing.csv"),
            "--split",
            "val",
        ],
    )

    args = routing_analysis.parse_args()

    assert args.eval_split == "val"


def test_parse_args_accepts_analysis_tag(monkeypatch, tmp_path):
    monkeypatch.setattr(
        routing_analysis.sys,
        "argv",
        [
            "routing_analysis.py",
            "--config",
            str(tmp_path / "config.yaml"),
            "--ckpt-path",
            str(tmp_path / "model.ckpt"),
            "--label-name",
            "age",
            "--output",
            str(tmp_path / "routing.csv"),
            "--analysis-tag",
            "post_finetune",
        ],
    )

    args = routing_analysis.parse_args()

    assert args.analysis_tag == "post_finetune"


def test_parse_args_accepts_pretrained_only_without_ckpt(monkeypatch, tmp_path):
    monkeypatch.setattr(
        routing_analysis.sys,
        "argv",
        [
            "routing_analysis.py",
            "--config",
            str(tmp_path / "config.yaml"),
            "--label-name",
            "age",
            "--output",
            str(tmp_path / "routing.csv"),
            "--pretrained-only",
            "--pretrained-backbone-path",
            str(tmp_path / "pretrain.ckpt"),
        ],
    )

    args = routing_analysis.parse_args()

    assert args.pretrained_only is True
    assert args.ckpt_path is None
