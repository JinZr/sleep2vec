from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from sleep2expert.checkpoints import load_pretrain_init_weights


class _TinyExpert(torch.nn.Module):
    def __init__(self, hidden_size: int = 2, intermediate_size: int = 3):
        super().__init__()
        self.dense_in = torch.nn.Linear(hidden_size, intermediate_size)
        self.dense_out = torch.nn.Linear(intermediate_size, hidden_size)


class _TinyMoeLayer(torch.nn.Module):
    def __init__(self, hidden_size: int = 2, intermediate_size: int = 3, num_experts: int = 2):
        super().__init__()
        self.moe_ffn = torch.nn.Module()
        self.moe_ffn.experts = torch.nn.ModuleList(
            [_TinyExpert(hidden_size=hidden_size, intermediate_size=intermediate_size) for _ in range(num_experts)]
        )
        self.moe_ffn.layer_norm = torch.nn.LayerNorm(hidden_size)


class _TinyMoeModule(torch.nn.Module):
    def __init__(self, hidden_size: int = 2, intermediate_size: int = 3, num_experts: int = 2):
        super().__init__()
        self.encoder = torch.nn.Module()
        self.encoder.encoder = torch.nn.Module()
        self.encoder.encoder.layer = torch.nn.ModuleList(
            [_TinyMoeLayer(hidden_size=hidden_size, intermediate_size=intermediate_size, num_experts=num_experts)]
        )


class _TinyStandaloneTarget(_TinyMoeModule):
    def __init__(self):
        super().__init__()
        self.encoder.encoder.layer[0].attention = torch.nn.Module()
        self.encoder.encoder.layer[0].attention.self_attention = torch.nn.Module()
        self.encoder.encoder.layer[0].attention.self_attention.query = torch.nn.Linear(2, 2)


def _dense_ffn_state(prefix: str = "encoder.encoder.layer.0.") -> dict[str, torch.Tensor]:
    return {
        f"{prefix}intermediate.dense.weight": torch.arange(6, dtype=torch.float32).view(3, 2),
        f"{prefix}intermediate.dense.bias": torch.arange(3, dtype=torch.float32),
        f"{prefix}output.dense.weight": torch.arange(6, dtype=torch.float32).view(2, 3),
        f"{prefix}output.dense.bias": torch.arange(2, dtype=torch.float32),
        f"{prefix}output.layer_norm.weight": torch.tensor([1.5, 2.5]),
        f"{prefix}output.layer_norm.bias": torch.tensor([0.5, -0.5]),
    }


def _save_pretrain_ckpt(path: Path, state: dict[str, torch.Tensor]) -> None:
    torch.save({"state_dict": {f"model.{key}": value for key, value in state.items()}}, path)


def test_sleep2expert_dense_ffn_weights_clone_into_compatible_moe_experts(tmp_path: Path):
    module = _TinyMoeModule()
    ckpt_path = tmp_path / "dense.ckpt"
    dense_state = _dense_ffn_state()
    _save_pretrain_ckpt(ckpt_path, dense_state)

    result = load_pretrain_init_weights(module, ckpt_path, device="cpu", strict=False)

    assert result.used_prefix == "model."
    layer = module.encoder.encoder.layer[0]
    for expert in layer.moe_ffn.experts:
        assert torch.equal(expert.dense_in.weight, dense_state["encoder.encoder.layer.0.intermediate.dense.weight"])
        assert torch.equal(expert.dense_in.bias, dense_state["encoder.encoder.layer.0.intermediate.dense.bias"])
        assert torch.equal(expert.dense_out.weight, dense_state["encoder.encoder.layer.0.output.dense.weight"])
        assert torch.equal(expert.dense_out.bias, dense_state["encoder.encoder.layer.0.output.dense.bias"])
    assert torch.equal(layer.moe_ffn.layer_norm.weight, dense_state["encoder.encoder.layer.0.output.layer_norm.weight"])
    assert torch.equal(layer.moe_ffn.layer_norm.bias, dense_state["encoder.encoder.layer.0.output.layer_norm.bias"])


def test_sleep2expert_full_moe_checkpoint_loads_without_dense_sources(tmp_path: Path):
    module = _TinyMoeModule()
    ckpt_path = tmp_path / "moe.ckpt"
    moe_state = {
        key: torch.full_like(value, float(idx + 1))
        for idx, (key, value) in enumerate(module.state_dict().items())
        if "moe_ffn.experts." in key or "moe_ffn.layer_norm." in key
    }
    assert all("intermediate.dense" not in key and "output.dense" not in key for key in moe_state)
    _save_pretrain_ckpt(ckpt_path, moe_state)

    result = load_pretrain_init_weights(module, ckpt_path, device="cpu", strict=False)

    assert result.missing_keys == []
    layer = module.encoder.encoder.layer[0]
    assert torch.equal(
        layer.moe_ffn.experts[0].dense_in.weight,
        moe_state["encoder.encoder.layer.0.moe_ffn.experts.0.dense_in.weight"],
    )
    assert torch.equal(layer.moe_ffn.layer_norm.weight, moe_state["encoder.encoder.layer.0.moe_ffn.layer_norm.weight"])


def test_sleep2expert_dense_ffn_weights_fail_when_shapes_differ(tmp_path: Path):
    module = _TinyMoeModule(intermediate_size=2)
    ckpt_path = tmp_path / "dense.ckpt"
    _save_pretrain_ckpt(ckpt_path, _dense_ffn_state())

    with pytest.raises(ValueError, match="Cannot initialize MoE layer") as exc:
        load_pretrain_init_weights(module, ckpt_path, device="cpu", strict=False)

    message = str(exc.value)
    assert "encoder.encoder.layer.0.moe_ffn.experts.0.dense_in.weight" in message
    assert "encoder.encoder.layer.0.intermediate.dense.weight" in message


def test_sleep2expert_checkpoint_loader_still_rejects_legacy_hf_roformer_keys(tmp_path: Path):
    module = _TinyStandaloneTarget()
    ckpt_path = tmp_path / "legacy.ckpt"
    torch.save(
        {
            "state_dict": {
                "model.encoder.encoder.layer.0.attention.self.query.weight": torch.ones(2, 2),
            }
        },
        ckpt_path,
    )

    with pytest.raises(ValueError, match="does not support loading legacy sleep2vec/HF RoFormer"):
        load_pretrain_init_weights(module, ckpt_path, device="cpu", strict=False)
