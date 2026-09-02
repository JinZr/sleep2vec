from __future__ import annotations

from importlib import import_module
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from agent_tools import configs, plan_context
from agent_tools.decisions import DecisionStatus, evaluate_consultation_gates
from agent_tools.models import REPO_ROOT, SUPPORTED_VARIANTS
from agent_tools.plan_hparam import (
    final_test_checkpoint_issues,
    hparam_yaml_override_issues,
    render_hparam_preflight_card,
    validate_finetune_config_bytes,
    validate_hparam_run_configs,
)
from agent_tools.recipes import load_consultation_policy

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
        "expected_runtime_commit": "a" * 40,
        "runtime_commit": "a" * 40,
        "runtime_hostname": "runtime-host",
        "module": module,
        "module_origin": f"/target/repo/{module.replace('.', '/')}.py",
        "validated_argv_sha256": "b" * 64,
    }


@pytest.mark.parametrize(
    ("variant", "config_path"),
    [
        ("sleep2vec", "configs/sleep2vec_dense_finetune_cls.yaml"),
        ("sleep2vec2", "configs/sleep2vec2/sleep2vec_dense_finetune_cls.yaml"),
        ("sleep2expert", "configs/sleep2expert/moe/sleep2expert_phase_moe_finetune_cls.yaml"),
        ("sex_age_baseline", "configs/sex_age_baseline/cox.yaml"),
    ],
)
def test_finetune_config_bytes_use_variant_loader(monkeypatch, variant: str, config_path: str):
    config_module = import_module(f"{variant}.config")
    load = config_module.load_finetune_config
    validate = config_module.validate_model_config
    config_bytes = (REPO_ROOT / config_path).read_bytes()
    calls = []

    def tracked_load(path):
        calls.append(("load", Path(path).read_bytes()))
        return load(path)

    def tracked_validate(model):
        calls.append(("validate", model))
        return validate(model)

    monkeypatch.setattr(config_module, "load_finetune_config", tracked_load)
    monkeypatch.setattr(config_module, "validate_model_config", tracked_validate)
    validate_finetune_config_bytes({"variant": variant}, config_bytes)

    assert [call[0] for call in calls] == ["load", "validate"]
    assert calls[0][1] == config_bytes

    with pytest.raises(ValueError):
        validate_finetune_config_bytes({"variant": variant}, b"{}\n")


@pytest.mark.parametrize(
    ("variant", "label", "config_path"),
    [
        ("sleep2vec", "age", "configs/ppg_age_finetune_large.yaml"),
        ("sleep2vec2", "age", "configs/sleep2vec2/ppg_age_finetune_large.yaml"),
    ],
)
def test_finetune_balanced_validation_stays_torch_free(variant: str, label: str, config_path: str):
    script = """
import builtins
from pathlib import Path
import sys

original_import = builtins.__import__

def import_without_torch(name, *args, **kwargs):
    if name == "torch" or name.startswith("torch."):
        raise ModuleNotFoundError("No module named 'torch'")
    return original_import(name, *args, **kwargs)

builtins.__import__ = import_without_torch
from agent_tools.plan_hparam import validate_finetune_config_bytes

validate_finetune_config_bytes(
    {
        "variant": sys.argv[1],
        "inputs": {"label_name": sys.argv[2]},
        "search": {"profile": "finetune_balanced"},
    },
    Path(sys.argv[3]).read_bytes(),
)
"""

    result = subprocess.run(
        [sys.executable, "-c", script, variant, label, str(REPO_ROOT / config_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("variant", "config_path"),
    [
        ("sleep2vec", "configs/sleep2vec_dense_finetune_cls.yaml"),
        ("sleep2vec2", "configs/sleep2vec2/sleep2vec_dense_finetune_cls.yaml"),
        ("sleep2expert", "configs/sleep2expert/moe/sleep2expert_phase_moe_finetune_cls.yaml"),
    ],
)
def test_finetune_candidates_validate_joint_overrides_with_real_variant_loader(variant: str, config_path: str):
    payload = yaml.safe_load((REPO_ROOT / config_path).read_bytes())
    payload["model"]["cls"].update({"downstream": "tokens", "embedding_type": "bert"})
    config_bytes = yaml.safe_dump(payload).encode()
    recipe = {"variant": variant, "inputs": {"config": config_path}}
    downstream = {"yaml:/model/cls/downstream": "cls"}
    embedding = {"yaml:/model/cls/embedding_type": "none"}

    for point in ({}, downstream, embedding):
        recipe["search"] = {"configurations": [point]}
        assert hparam_yaml_override_issues(recipe, config_bytes=config_bytes) == []

    recipe["search"] = {"configurations": [{**downstream, **embedding}]}
    issues = hparam_yaml_override_issues(recipe, config_bytes=config_bytes)

    assert len(issues) == 1
    assert issues[0].status == DecisionStatus.FAIL
    assert issues[0].field == "hparam_search_space"
    assert "model.cls.embedding_type must be set" in issues[0].message
    assert issues[0].evidence["preflight_before_workspace"] is True


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
        "sex_age_baseline.config.load_finetune_config (load_config with default validate_sidecars=False)",
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
    monkeypatch,
    variant: str,
    config_path: str,
    module: str,
    loader: str,
    architecture: str,
    tokenizer: str | None,
):
    config_bytes = (REPO_ROOT / config_path).read_bytes()
    summarize = configs.config_summary
    calls = []

    def tracked_summary(path, **kwargs):
        calls.append((path, kwargs))
        return summarize(path, **kwargs)

    monkeypatch.setattr(configs, "config_summary", tracked_summary)
    card = render_hparam_preflight_card(
        {"variant": variant, "inputs": {"config": config_path}},
        _snapshot(module),
        [({"run_id": "run-000"}, config_bytes)],
    )
    channels = "none"
    if tokenizer is not None:
        channels = ", ".join(
            f"{name} (input_dim={input_dim}, tokenizer={tokenizer}, out_dim=768)"
            for name, input_dim in _SIGNAL_CHANNELS
        )

    assert "- Scheduler: `direct`" in card
    assert "- Control transport: `local`" in card
    assert "- Target CLI preflight host: `runtime-host`" in card
    assert "- Target CLI Python: `/target/python` (version `3.10.0`; frozen command: `python`)" in card
    assert f"| {variant} | {module} | {loader} | {architecture} | {channels} | run-000 |" in card
    assert calls == [
        (config_path, {"variant": variant, "validate_survival_local_paths": False, "config_bytes": config_bytes})
    ]
    assert "- Total planned runs: 1" in card
    assert "- Target CLI argv checks: 1" in card
    assert "- Planner-local final-config checks: 1 runs; 1 unique exact YAML byte sequences" in card
    assert f"`{variant}.config.load_finetune_config` + `{variant}.config.validate_model_config`" in card
    assert "target CLI preflight proves argument parsing only" in card
    assert "Model construction, forward/backward, checkpoint compatibility, and GPU execution: not performed" in card
    assert "do not perform full-sidecar reads or complete candidate-specific dataset validation" in card
    assert "Validated run count" not in card


def test_hparam_preflight_card_memoizes_exact_bytes_only_within_one_card(monkeypatch):
    config_path = "configs/sleep2vec_dense_finetune_cls.yaml"
    first = (REPO_ROOT / config_path).read_bytes()
    same_model = first + b"\n# Distinct frozen bytes with the same model.\n"
    payload = yaml.safe_load(first)
    payload["model"]["backbone"]["num_hidden_layers"] = 6
    changed_model = yaml.safe_dump(payload).encode()
    recipe = {"variant": "sleep2vec", "inputs": {"config": config_path}}
    run_configs = [
        ({"run_id": f"run-{index:03d}"}, content)
        for index, content in enumerate([first, first, same_model, changed_model])
    ]
    summarize = configs.config_summary
    calls = []

    def tracked_summary(path, **kwargs):
        calls.append(kwargs["config_bytes"])
        return summarize(path, **kwargs)

    monkeypatch.setattr(configs, "config_summary", tracked_summary)
    card = render_hparam_preflight_card(recipe, _snapshot("sleep2vec.finetune"), run_configs)

    assert calls == [first, same_model, changed_model]
    assert "roformer (hidden_size=768, layers=12)" in card
    assert "roformer (hidden_size=768, layers=6)" in card
    assert "| run-000, run-001, run-002 |" in card
    assert "| run-003 |" in card
    assert "- Target CLI argv checks: 4" in card
    assert "- Planner-local final-config checks: 4 runs; 3 unique exact YAML byte sequences" in card
    assert render_hparam_preflight_card(recipe, _snapshot("sleep2vec.finetune"), run_configs) == card
    assert calls == [first, same_model, changed_model] * 2


def test_hparam_preflight_card_ignores_source_config_mutation(tmp_path: Path):
    config = tmp_path / "config.yaml"
    config_bytes = (REPO_ROOT / "configs/sleep2vec_dense_finetune_cls.yaml").read_bytes()
    config.write_bytes(config_bytes)
    recipe = {"variant": "sleep2vec", "inputs": {"config": str(config)}}
    run_configs = [({"run_id": "run-000"}, config_bytes)]
    card = render_hparam_preflight_card(recipe, _snapshot("sleep2vec.finetune"), run_configs)

    config.write_text("model: {}\n")

    assert render_hparam_preflight_card(recipe, _snapshot("sleep2vec.finetune"), run_configs) == card


def test_hparam_preflight_card_requires_architecture_provenance():
    with pytest.raises(ValueError, match="Generated hparam config lacks architecture provenance: run-000"):
        render_hparam_preflight_card(
            {"variant": "sleep2vec", "inputs": {"config": "config.yaml"}},
            _snapshot("sleep2vec.finetune"),
            [({"run_id": "run-000"}, b"model: {}\n")],
        )


def test_hparam_preflight_card_keeps_sex_age_structural_loader(monkeypatch):
    import sex_age_baseline.config as baseline_config

    config_path = "configs/sex_age_baseline/cox.yaml"
    config_bytes = (REPO_ROOT / config_path).read_bytes()
    load_config = baseline_config.load_config
    calls = []

    def tracked_load(path, *, validate_sidecars=False):
        calls.append((Path(path).read_bytes(), validate_sidecars))
        return load_config(path, validate_sidecars=validate_sidecars)

    monkeypatch.setattr(baseline_config, "load_config", tracked_load)
    validate_finetune_config_bytes({"variant": "sex_age_baseline"}, config_bytes)
    assert calls == [(config_bytes, False)]
    calls.clear()
    card = render_hparam_preflight_card(
        {"variant": "sex_age_baseline", "inputs": {"config": config_path}},
        _snapshot("sex_age_baseline.finetune"),
        [({"run_id": "run-000"}, config_bytes)],
    )

    assert calls == [(config_bytes, False)]
    assert "sex_age_mlp (features=age, sex)" in card


@pytest.mark.parametrize(("variant", "config_path", "module"), [case[:3] for case in _PREFLIGHT_VARIANT_CASES])
@pytest.mark.parametrize("sidecar_kind", ["survival", "multilabel"])
def test_hparam_card_skips_sidecar_tables_without_weakening_validation(
    tmp_path: Path, monkeypatch, variant: str, config_path: str, module: str, sidecar_kind: str
):
    from data import multilabel, survival

    index = tmp_path / "index.csv"
    index.write_text("eid,split,age,sex\n001,train,50,0\n")
    disease_columns = tmp_path / "disease_columns.txt"
    disease_columns.write_text("d1\nd2\n")
    bad_table = tmp_path / "bad.csv"
    bad_table.write_text("eid,wrong_column\n001,1\n")
    payload = yaml.safe_load((REPO_ROOT / config_path).read_bytes())
    payload["data"]["finetune_data_index"] = str(index)
    payload["finetune"].pop("survival", None)
    payload["finetune"]["task"] = {
        "type": "survival" if sidecar_kind == "survival" else "multilabel_classification",
        "output_dim": 2,
        "is_seq": False,
        "monitor": "val_loss",
        "monitor_mod": "min",
    }
    table_fields = (
        ("event_time_index", "is_event_index", "has_label_index")
        if sidecar_kind == "survival"
        else ("label_index", "has_label_index")
    )
    payload["finetune"][sidecar_kind] = {
        "key_column": "eid",
        "disease_columns_index": str(disease_columns),
        **{field: str(bad_table) for field in table_fields},
    }
    config = tmp_path / "config.yaml"
    config_bytes = yaml.safe_dump(payload).encode()
    config.write_bytes(config_bytes)
    recipe = {"task": "finetune", "variant": variant, "inputs": {"config": str(config)}}
    calls = []
    load_survival = survival.load_survival_label_table
    load_multilabel = multilabel.load_multilabel_label_table

    def tracked_survival(*args, **kwargs):
        calls.append("survival")
        return load_survival(*args, **kwargs)

    def tracked_multilabel(*args, **kwargs):
        calls.append("multilabel")
        return load_multilabel(*args, **kwargs)

    monkeypatch.setattr(survival, "load_survival_label_table", tracked_survival)
    monkeypatch.setattr(multilabel, "load_multilabel_label_table", tracked_multilabel)

    run_configs = [({"run_id": "run-000"}, config_bytes), ({"run_id": "run-001"}, config_bytes)]
    validate_hparam_run_configs(recipe, run_configs)
    card = render_hparam_preflight_card(recipe, _snapshot(module), run_configs)

    assert "| run-000, run-001 |" in card
    assert calls == []

    summary = plan_context.load_config_summary_for_recipe(recipe, config_bytes=config_bytes)
    report = evaluate_consultation_gates(
        "finetune", recipe, summary, {}, load_consultation_policy(), require_experiment=False
    )

    assert calls == [sidecar_kind]
    assert any("columns must exactly match" in message for message in summary["finetune"][sidecar_kind]["issues"])
    assert report.exit_code != 0
    assert any(issue.field == f"{sidecar_kind}_sidecars" for issue in report.issues)

    checkpoint = tmp_path / "selected.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    recipe["inputs"].update({"ckpt_path": str(checkpoint), "final_eval_config_path": str(config)})
    issues = final_test_checkpoint_issues(recipe, summary, unlock_final_test=True)

    assert calls == [sidecar_kind, sidecar_kind]
    assert any(
        issue.field == f"{sidecar_kind}_sidecars" and issue.status == DecisionStatus.NEEDS_USER_INPUT
        for issue in issues
    )


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
