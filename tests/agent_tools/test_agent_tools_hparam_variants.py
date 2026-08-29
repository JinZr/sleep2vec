from __future__ import annotations

import pytest

from agent_tools.models import REPO_ROOT, SUPPORTED_VARIANTS
from agent_tools.plan_hparam import render_hparam_preflight_card, validate_final_eval_config_bytes

_SIGNAL_CHANNELS = (
    ("heartbeat", 120),
    ("breath", 120),
    ("eeg_original", 3840),
    ("ecg_original", 3840),
    ("eog_original", 3840),
    ("emg_original", 3840),
    ("spo2", 120),
    ("resp_original", 120),
    ("resp_nasal_original", 120),
)


def _snapshot(module: str) -> dict:
    return {
        "target": "local",
        "host": "",
        "python": "/target/python",
        "python_version": "3.10.0",
        "python_command": "python",
        "runtime_commit": "a" * 40,
        "runtime_hostname": "runtime-host",
        "module": module,
        "module_origin": f"/target/repo/{module.replace('.', '/')}.py",
        "validated_argv_sha256": "b" * 64,
    }


@pytest.mark.parametrize(
    ("variant", "config_path"),
    [
        ("sleep2vec2", "configs/sleep2vec2/sleep2vec_dense_finetune_cls.yaml"),
        ("sleep2expert", "configs/sleep2expert/moe/sleep2expert_phase_moe_finetune_cls.yaml"),
        ("sex_age_baseline", "configs/sex_age_baseline/cox.yaml"),
    ],
)
def test_final_eval_config_bytes_use_variant_loader(variant: str, config_path: str):
    validate_final_eval_config_bytes({"variant": variant}, (REPO_ROOT / config_path).read_bytes())

    with pytest.raises(ValueError):
        validate_final_eval_config_bytes({"variant": variant}, b"{}\n")


_PREFLIGHT_VARIANT_CASES = [
    (
        "sleep2vec",
        "configs/sleep2vec_dense_finetune_cls.yaml",
        "sleep2vec.finetune",
        "sleep2vec.config.load_finetune_config",
        "roformer (hidden_size=768, layers=12)",
        "sundial",
    ),
    (
        "sleep2vec2",
        "configs/sleep2vec2/sleep2vec_dense_finetune_cls.yaml",
        "sleep2vec2.finetune",
        "sleep2vec2.config.load_finetune_config",
        "roformer (hidden_size=768, layers=12)",
        "sundial2",
    ),
    (
        "sleep2expert",
        "configs/sleep2expert/moe/sleep2expert_phase_moe_finetune_cls.yaml",
        "sleep2expert.finetune",
        "sleep2expert.config.load_finetune_config",
        "roformer (hidden_size=768, layers=12)",
        "sundial",
    ),
    (
        "sex_age_baseline",
        "configs/sex_age_baseline/cox.yaml",
        "sex_age_baseline.finetune",
        "sex_age_baseline.config.load_config(validate_sidecars=True)",
        "sex_age_mlp (features=age, sex)",
        None,
    ),
]


def test_hparam_preflight_route_cases_cover_supported_variants():
    assert {case[0] for case in _PREFLIGHT_VARIANT_CASES} == set(SUPPORTED_VARIANTS)


@pytest.mark.parametrize(
    ("variant", "config_path", "module", "loader", "architecture", "tokenizer"),
    _PREFLIGHT_VARIANT_CASES,
)
def test_hparam_preflight_card_uses_variant_config_provenance(
    variant: str,
    config_path: str,
    module: str,
    loader: str,
    architecture: str,
    tokenizer: str | None,
):
    card = render_hparam_preflight_card(
        {"variant": variant, "inputs": {"config": config_path}},
        _snapshot(module),
        [({"run_id": "run-000"}, (REPO_ROOT / config_path).read_bytes())],
    )
    channels = "none"
    if tokenizer is not None:
        channels = ", ".join(
            f"{name} (input_dim={input_dim}, tokenizer={tokenizer}, out_dim=768)"
            for name, input_dim in _SIGNAL_CHANNELS
        )

    assert "- Scheduler: `direct`" in card
    assert "- Control transport: `local`" in card
    assert "- Validated preflight host: `runtime-host`" in card
    assert "- Runtime Python: `/target/python` (version `3.10.0`; frozen command: `python`)" in card
    assert f"| {variant} | {module} | {loader} | {architecture} | {channels} | run-000 |" in card


@pytest.mark.parametrize(
    ("direct_controller", "controller_label"),
    [(False, "bound cluster"), (True, "direct controller")],
)
def test_hparam_preflight_card_projects_slurm_topology(direct_controller: bool, controller_label: str):
    config_path = "configs/sleep2vec_dense_finetune_cls.yaml"
    card = render_hparam_preflight_card(
        {
            "variant": "sleep2vec",
            "inputs": {"config": config_path},
            "execution": {
                "gpus_per_run": 8,
                "scheduler": {
                    "type": "slurm",
                    "partition": "gpu",
                    "cpus_per_task": 6,
                    "memory": "64G",
                    "walltime": "12:00:00",
                    "direct_controller": direct_controller,
                },
            },
        },
        _snapshot("sleep2vec.finetune"),
        [({"run_id": "run-000"}, (REPO_ROOT / config_path).read_bytes())],
    )

    assert "- Scheduler: `slurm`" in card
    assert (
        "- Planned allocation topology: nodes/run=1, tasks/run=8, GPUs/run=8, "
        "one Slurm task per Lightning rank" in card
    )
    assert "- Planned task resources: CPUs/task=6, total CPUs=48, memory=64G/allocation, walltime=12:00:00" in card
    assert f"- Scheduler controller: {controller_label}" in card
