from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pytest
import yaml

torch = pytest.importorskip("torch")

from sleep2expert.export_subnetwork import export_subnetwork, parse_args


def _write_config(
    path: Path,
    *,
    finetune: bool,
    num_hidden_layers: int = 1,
    moe_layer_indices: list[int] | None = None,
    router_type: str = "learned",
) -> None:
    payload = {
        "model": {
            "backbone": {
                "name": "roformer",
                "hidden_size": 4,
                "num_hidden_layers": num_hidden_layers,
                "num_attention_heads": 1,
                "vocab_size": 1,
                "moe": {
                    "enabled": True,
                    "layer_indices": moe_layer_indices or [1],
                    "num_experts": 6,
                    "top_k": 2,
                    "expert_hidden_size": 8,
                    "router_type": router_type,
                    "use_modality_group_mask": True,
                    "expert_groups": {
                        "shared": [0, 5],
                        "cardiac": [2, 4],
                        "respiratory": [1, 3],
                    },
                    "modality_to_groups": {
                        "heartbeat": ["shared", "cardiac"],
                        "breath": ["shared", "respiratory"],
                    },
                },
            },
            "projection": {"name": "simclr", "enabled": True, "hidden_dim": 4, "out_dim": 2},
            "cls": {"embedding_type": None, "downstream": "tokens"},
            "channels": [
                {"name": "heartbeat", "input_dim": 2, "tokenizer": {"name": "linear", "out_dim": 4}},
                {"name": "breath", "input_dim": 2, "tokenizer": {"name": "linear", "out_dim": 4}},
            ],
        },
        "data": {"backend": "npz", "max_tokens": 4},
    }
    if finetune:
        payload["model"]["head"] = {
            "name": "classification",
            "channel_agg": {"name": "gated_scalar", "kwargs": {}},
            "temporal_agg": {"name": "mean", "kwargs": {}},
        }
        payload["finetune"] = {"freeze_tokenizer": True}
    else:
        payload["loss"] = {"name": "weighted_info_nce", "temperature": 0.2, "params": {}}
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def _write_checkpoint(path: Path) -> None:
    state = {
        "model.encoder.encoder.layer.0.moe_ffn.router.router.weight": torch.arange(12, dtype=torch.float32).view(6, 2),
        "model.encoder.encoder.layer.0.moe_ffn.router.router.bias": torch.arange(6, dtype=torch.float32),
        "model.encoder.encoder.layer.0.moe_ffn.layer_norm.weight": torch.ones(4),
        "model.head.weight": torch.full((1,), 99.0),
    }
    for expert_id in range(6):
        state[f"model.encoder.encoder.layer.0.moe_ffn.experts.{expert_id}.dense_in.weight"] = torch.full(
            (1,), float(expert_id)
        )
    torch.save(
        {
            "state_dict": state,
            "model_config": {"stale": True},
            "model_config_yaml": "stale: true\n",
            "optimizer_states": [{"state": "stale"}],
            "lr_schedulers": [{"state": "stale"}],
            "epoch": 3,
        },
        path,
    )


@pytest.mark.parametrize("finetune", [False, True])
def test_sleep2expert_export_subnetwork_rewrites_config_and_checkpoint(tmp_path: Path, finetune: bool):
    config_path = tmp_path / "config.yaml"
    ckpt_path = tmp_path / "source.ckpt"
    output_dir = tmp_path / "exported"
    _write_config(config_path, finetune=finetune)
    _write_checkpoint(ckpt_path)

    manifest = export_subnetwork(
        argparse.Namespace(
            config=config_path,
            ckpt_path=ckpt_path,
            route_expert_groups=["shared", "cardiac"],
            output_dir=output_dir,
        )
    )

    assert (output_dir / "config.yaml").exists()
    assert (output_dir / "model.ckpt").exists()
    assert (output_dir / "manifest.json").exists()
    assert (output_dir / "expert_id_map.csv").exists()
    assert manifest["config_type"] == ("finetune" if finetune else "pretrain")
    assert manifest["old_to_new"] == {"0": 0, "2": 1, "4": 2, "5": 3}
    assert manifest["dropped"]["expert_state_keys"] == 2
    assert manifest["dropped"]["removed_resume_keys"] == ["optimizer_states", "lr_schedulers"]

    exported_config = yaml.safe_load((output_dir / "config.yaml").read_text())
    moe = exported_config["model"]["backbone"]["moe"]
    assert moe["num_experts"] == 4
    assert moe["top_k"] == 2
    assert moe["expert_groups"] == {"shared": [0, 3], "cardiac": [1, 2]}
    assert moe["modality_to_groups"] == {"heartbeat": ["shared", "cardiac"], "breath": ["shared"]}

    exported = torch.load(output_dir / "model.ckpt", map_location="cpu", weights_only=False)
    assert "optimizer_states" not in exported
    assert "lr_schedulers" not in exported
    assert exported["epoch"] == 3
    assert exported["model_config"] == exported_config["model"]
    assert yaml.safe_load(exported["model_config_yaml"]) == exported_config["model"]

    state = exported["state_dict"]
    assert torch.equal(
        state["model.encoder.encoder.layer.0.moe_ffn.router.router.weight"],
        torch.arange(12, dtype=torch.float32).view(6, 2).index_select(0, torch.tensor([0, 2, 4, 5])),
    )
    assert torch.equal(
        state["model.encoder.encoder.layer.0.moe_ffn.router.router.bias"],
        torch.tensor([0.0, 2.0, 4.0, 5.0]),
    )
    assert torch.equal(state["model.encoder.encoder.layer.0.moe_ffn.experts.0.dense_in.weight"], torch.tensor([0.0]))
    assert torch.equal(state["model.encoder.encoder.layer.0.moe_ffn.experts.1.dense_in.weight"], torch.tensor([2.0]))
    assert torch.equal(state["model.encoder.encoder.layer.0.moe_ffn.experts.2.dense_in.weight"], torch.tensor([4.0]))
    assert torch.equal(state["model.encoder.encoder.layer.0.moe_ffn.experts.3.dense_in.weight"], torch.tensor([5.0]))
    assert "model.encoder.encoder.layer.0.moe_ffn.experts.4.dense_in.weight" not in state
    assert torch.equal(state["model.head.weight"], torch.tensor([99.0]))

    with (output_dir / "expert_id_map.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert rows == [
        {"old_expert_id": "0", "new_expert_id": "0"},
        {"old_expert_id": "2", "new_expert_id": "1"},
        {"old_expert_id": "4", "new_expert_id": "2"},
        {"old_expert_id": "5", "new_expert_id": "3"},
    ]


def test_sleep2expert_export_subnetwork_rejects_unknown_group(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    ckpt_path = tmp_path / "source.ckpt"
    _write_config(config_path, finetune=False)
    _write_checkpoint(ckpt_path)

    with pytest.raises(ValueError, match="Unknown route expert group"):
        export_subnetwork(
            argparse.Namespace(
                config=config_path,
                ckpt_path=ckpt_path,
                route_expert_groups=["unknown"],
                output_dir=tmp_path / "exported",
            )
        )


def test_sleep2expert_export_subnetwork_rejects_channel_with_too_few_experts(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    ckpt_path = tmp_path / "source.ckpt"
    _write_config(config_path, finetune=False)
    _write_checkpoint(ckpt_path)

    with pytest.raises(ValueError, match="breath.*0 experts"):
        export_subnetwork(
            argparse.Namespace(
                config=config_path,
                ckpt_path=ckpt_path,
                route_expert_groups=["cardiac"],
                output_dir=tmp_path / "exported",
            )
        )


def test_sleep2expert_export_subnetwork_rejects_missing_selected_expert_weights(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    ckpt_path = tmp_path / "source.ckpt"
    output_dir = tmp_path / "exported"
    _write_config(config_path, finetune=False)
    _write_checkpoint(ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    checkpoint["state_dict"].pop("model.encoder.encoder.layer.0.moe_ffn.experts.4.dense_in.weight")
    torch.save(checkpoint, ckpt_path)

    with pytest.raises(ValueError, match="missing selected MoE expert weights.*old expert ID\\(s\\) \\[4\\]"):
        export_subnetwork(
            argparse.Namespace(
                config=config_path,
                ckpt_path=ckpt_path,
                route_expert_groups=["shared", "cardiac"],
                output_dir=output_dir,
            )
        )

    assert not output_dir.exists()


def test_sleep2expert_export_subnetwork_rejects_missing_learned_router_weights(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    ckpt_path = tmp_path / "source.ckpt"
    output_dir = tmp_path / "exported"
    _write_config(config_path, finetune=False)
    _write_checkpoint(ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    checkpoint["state_dict"].pop("model.encoder.encoder.layer.0.moe_ffn.router.router.bias")
    torch.save(checkpoint, ckpt_path)

    with pytest.raises(ValueError, match="missing learned MoE router weights.*router\\.router.*bias"):
        export_subnetwork(
            argparse.Namespace(
                config=config_path,
                ckpt_path=ckpt_path,
                route_expert_groups=["shared", "cardiac"],
                output_dir=output_dir,
            )
        )

    assert not output_dir.exists()


def test_sleep2expert_export_subnetwork_rejects_learned_router_weights_for_hard_router(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    ckpt_path = tmp_path / "source.ckpt"
    output_dir = tmp_path / "exported"
    _write_config(config_path, finetune=False, router_type="hard_group")
    _write_checkpoint(ckpt_path)

    with pytest.raises(ValueError, match="contains learned MoE router weights.*router_type='hard_group'"):
        export_subnetwork(
            argparse.Namespace(
                config=config_path,
                ckpt_path=ckpt_path,
                route_expert_groups=["shared", "cardiac"],
                output_dir=output_dir,
            )
        )

    assert not output_dir.exists()


def test_sleep2expert_export_subnetwork_rejects_missing_expected_moe_layer(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    ckpt_path = tmp_path / "source.ckpt"
    output_dir = tmp_path / "exported"
    _write_config(config_path, finetune=False, num_hidden_layers=2, moe_layer_indices=[1, 2])
    _write_checkpoint(ckpt_path)

    with pytest.raises(ValueError, match="missing expected MoE layer expert weights.*layer\\.1\\.moe_ffn\\.experts"):
        export_subnetwork(
            argparse.Namespace(
                config=config_path,
                ckpt_path=ckpt_path,
                route_expert_groups=["shared", "cardiac"],
                output_dir=output_dir,
            )
        )

    assert not output_dir.exists()


def test_sleep2expert_export_subnetwork_rejects_missing_expected_moe_layer_router(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    ckpt_path = tmp_path / "source.ckpt"
    output_dir = tmp_path / "exported"
    _write_config(config_path, finetune=False, num_hidden_layers=2, moe_layer_indices=[1, 2])
    _write_checkpoint(ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    for expert_id in range(6):
        checkpoint["state_dict"][f"model.encoder.encoder.layer.1.moe_ffn.experts.{expert_id}.dense_in.weight"] = (
            torch.full((1,), float(expert_id))
        )
    torch.save(checkpoint, ckpt_path)

    with pytest.raises(ValueError, match="missing learned MoE router weights.*layer\\.1\\.moe_ffn\\.router\\.router"):
        export_subnetwork(
            argparse.Namespace(
                config=config_path,
                ckpt_path=ckpt_path,
                route_expert_groups=["shared", "cardiac"],
                output_dir=output_dir,
            )
        )

    assert not output_dir.exists()


def test_sleep2expert_export_subnetwork_rejects_unexpected_moe_layer(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    ckpt_path = tmp_path / "source.ckpt"
    output_dir = tmp_path / "exported"
    _write_config(config_path, finetune=False, num_hidden_layers=2, moe_layer_indices=[1])
    _write_checkpoint(ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    checkpoint["state_dict"]["model.encoder.encoder.layer.1.moe_ffn.router.router.weight"] = torch.arange(
        12, dtype=torch.float32
    ).view(6, 2)
    checkpoint["state_dict"]["model.encoder.encoder.layer.1.moe_ffn.router.router.bias"] = torch.arange(
        6, dtype=torch.float32
    )
    for expert_id in range(6):
        checkpoint["state_dict"][f"model.encoder.encoder.layer.1.moe_ffn.experts.{expert_id}.dense_in.weight"] = (
            torch.full((1,), float(expert_id))
        )
    torch.save(checkpoint, ckpt_path)

    with pytest.raises(ValueError, match="unexpected MoE layer weights.*layer\\.1\\.moe_ffn"):
        export_subnetwork(
            argparse.Namespace(
                config=config_path,
                ckpt_path=ckpt_path,
                route_expert_groups=["shared", "cardiac"],
                output_dir=output_dir,
            )
        )

    assert not output_dir.exists()


def test_sleep2expert_export_subnetwork_rejects_nonempty_output_dir(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    ckpt_path = tmp_path / "source.ckpt"
    output_dir = tmp_path / "exported"
    output_dir.mkdir()
    (output_dir / "old.txt").write_text("stale")
    _write_config(config_path, finetune=False)
    _write_checkpoint(ckpt_path)

    with pytest.raises(FileExistsError, match="empty or absent"):
        export_subnetwork(
            argparse.Namespace(
                config=config_path,
                ckpt_path=ckpt_path,
                route_expert_groups=["shared", "cardiac"],
                output_dir=output_dir,
            )
        )


def test_sleep2expert_export_subnetwork_checks_output_dir_before_loading_checkpoint(monkeypatch, tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    ckpt_path = tmp_path / "source.ckpt"
    output_dir = tmp_path / "exported"
    output_dir.mkdir()
    (output_dir / "old.txt").write_text("stale")
    _write_config(config_path, finetune=False)
    _write_checkpoint(ckpt_path)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("checkpoint should not be loaded when output dir is invalid")

    monkeypatch.setattr("sleep2expert.export_subnetwork.load_checkpoint", fail_if_called)

    with pytest.raises(FileExistsError, match="empty or absent"):
        export_subnetwork(
            argparse.Namespace(
                config=config_path,
                ckpt_path=ckpt_path,
                route_expert_groups=["shared", "cardiac"],
                output_dir=output_dir,
            )
        )


def test_sleep2expert_export_subnetwork_parse_args_accepts_route_expert_groups(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    ckpt_path = tmp_path / "source.ckpt"
    _write_config(config_path, finetune=False)
    _write_checkpoint(ckpt_path)

    args = parse_args(
        [
            "--config",
            str(config_path),
            "--ckpt-path",
            str(ckpt_path),
            "--route-expert-groups",
            "shared",
            "cardiac",
            "--output-dir",
            str(tmp_path / "exported"),
        ]
    )

    assert args.route_expert_groups == ["shared", "cardiac"]
    assert args.output_dir == tmp_path / "exported"
