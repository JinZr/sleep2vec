from __future__ import annotations

import copy
from importlib import import_module
from pathlib import Path

import pytest
import yaml

from agent_tools import plan_hparam
from agent_tools.adaptive_proposals import validate_proposal
from agent_tools.decisions import DecisionStatus
from agent_tools.models import REPO_ROOT
from tests.agent_tools.test_agent_tools_adaptive_proposals import _proposal, _snapshot

_VARIANT_CONFIGS = [
    ("sleep2vec", "configs/sleep2vec_dense_finetune_cls.yaml"),
    ("sleep2vec2", "configs/sleep2vec2/sleep2vec_dense_finetune_cls.yaml"),
    ("sleep2expert", "configs/sleep2expert/moe/sleep2expert_phase_moe_finetune_cls.yaml"),
]


@pytest.mark.parametrize(("variant", "config_path"), _VARIANT_CONFIGS)
@pytest.mark.parametrize("proposal_form", ["parameters", "configurations"])
def test_agent_proposal_blocks_replace_source_groups_and_preserve_choices(
    monkeypatch, variant: str, config_path: str, proposal_form: str
):
    source = yaml.safe_load((REPO_ROOT / config_path).read_bytes())
    source["finetune"]["tuning"] = {"preset": "full", "groups": {"encoder": {"train": True}}}
    source["finetune"]["layer_mix"] = {
        "enabled": True,
        "shared_across_modalities": True,
        "layer_indices": [1, 2],
    }
    choices = {
        "yaml:/finetune/tuning": [{"preset": "head_only"}, source["finetune"]["tuning"]],
        "yaml:/finetune/layer_mix": [
            {"enabled": False, "shared_across_modalities": False, "layer_indices": None},
            source["finetune"]["layer_mix"],
        ],
    }
    source_before = copy.deepcopy(source)
    choices_before = copy.deepcopy(choices)
    snapshot = _snapshot(parameters=choices)
    proposal = _proposal(snapshot, parameters={key: [values[0]] for key, values in choices.items()})
    if proposal_form == "configurations":
        proposal["configurations"] = [{key: values[0] for key, values in proposal.pop("parameters").items()}]
    validated = validate_proposal(proposal, snapshot)
    recipe = {
        "variant": variant,
        "inputs": {"config": config_path},
        "search": {proposal_form: validated[proposal_form]},
    }
    config_module = import_module(f"{variant}.config")
    load = config_module.load_finetune_config
    observed = []

    def capture_load(path):
        bundle = load(path)
        observed.append((yaml.safe_load(Path(path).read_bytes()), bundle))
        return bundle

    monkeypatch.setattr(config_module, "load_finetune_config", capture_load)

    assert plan_hparam.hparam_yaml_override_issues(recipe, config_bytes=yaml.safe_dump(source).encode()) == []

    assert len(observed) == 1
    candidate, bundle = observed[0]
    assert candidate["finetune"]["tuning"] == {"preset": "head_only"}
    assert candidate["finetune"]["layer_mix"] == choices_before["yaml:/finetune/layer_mix"][0]
    assert bundle.finetune.tuning.groups["encoder"].train is False
    assert bundle.finetune.layer_mix.enabled is False
    assert source == source_before
    assert choices == choices_before


@pytest.mark.parametrize(("variant", "config_path"), _VARIANT_CONFIGS)
def test_agent_proposal_list_choice_replaces_layer_indices(monkeypatch, variant: str, config_path: str):
    source = yaml.safe_load((REPO_ROOT / config_path).read_bytes())
    source["finetune"]["layer_mix"].update({"enabled": True, "layer_indices": [1, 2]})
    choices = {"yaml:/finetune/layer_mix/layer_indices": [[2, 4], [1, 3, 5]]}
    choices_before = copy.deepcopy(choices)
    snapshot = _snapshot(parameters=choices)
    proposal = _proposal(snapshot, parameters={key: [values[0]] for key, values in choices.items()})
    validated = validate_proposal(proposal, snapshot)
    recipe = {
        "variant": variant,
        "inputs": {"config": config_path},
        "search": {"parameters": validated["parameters"]},
    }
    config_module = import_module(f"{variant}.config")
    load = config_module.load_finetune_config
    observed = []

    def capture_load(path):
        bundle = load(path)
        observed.append(bundle.finetune.layer_mix.layer_indices)
        return bundle

    monkeypatch.setattr(config_module, "load_finetune_config", capture_load)

    assert plan_hparam.hparam_yaml_override_issues(recipe, config_bytes=yaml.safe_dump(source).encode()) == []

    assert observed == [[2, 4]]
    assert source["finetune"]["layer_mix"]["layer_indices"] == [1, 2]
    assert choices == choices_before


@pytest.mark.parametrize(("variant", "config_path"), _VARIANT_CONFIGS)
def test_authorized_layer_mix_choice_still_requires_valid_depth(variant: str, config_path: str):
    source = yaml.safe_load((REPO_ROOT / config_path).read_bytes())
    invalid = {
        "enabled": True,
        "shared_across_modalities": False,
        "layer_indices": [source["model"]["backbone"]["num_hidden_layers"] + 1],
    }
    parameters = {"yaml:/finetune/layer_mix": [invalid]}
    snapshot = _snapshot(parameters=parameters)
    validated = validate_proposal(_proposal(snapshot, parameters=parameters), snapshot)
    recipe = {
        "variant": variant,
        "inputs": {"config": config_path},
        "search": {"parameters": validated["parameters"]},
    }

    issues = plan_hparam.hparam_yaml_override_issues(recipe, config_bytes=yaml.safe_dump(source).encode())

    assert len(issues) == 1
    assert issues[0].status == DecisionStatus.FAIL
    assert issues[0].field == "hparam_search_space"
    assert "layer_mix.layer_indices must be <= num_hidden_layers" in issues[0].message
    assert issues[0].evidence["preflight_before_workspace"] is True


@pytest.mark.parametrize(("variant", "config_path"), _VARIANT_CONFIGS)
def test_agent_proposal_tuning_block_uses_variant_preset_contract(variant: str, config_path: str):
    parameters = {"yaml:/finetune/tuning": [{"preset": "moe_conservative"}]}
    snapshot = _snapshot(parameters=parameters)
    validated = validate_proposal(_proposal(snapshot, parameters=parameters), snapshot)
    recipe = {
        "variant": variant,
        "inputs": {"config": config_path},
        "search": {"parameters": validated["parameters"]},
    }

    issues = plan_hparam.hparam_yaml_override_issues(recipe, config_bytes=(REPO_ROOT / config_path).read_bytes())

    if variant == "sleep2expert":
        assert issues == []
    else:
        assert len(issues) == 1
        assert issues[0].status == DecisionStatus.FAIL
        assert issues[0].field == "hparam_search_space"
        assert "finetune.tuning.preset must be one of" in issues[0].message
        assert issues[0].evidence["preflight_before_workspace"] is True
