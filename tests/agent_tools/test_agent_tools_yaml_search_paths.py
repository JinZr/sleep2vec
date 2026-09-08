from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from agent_tools import adaptive_hparam, managed_scheduler, plan_hparam
from agent_tools.adaptive_proposals import validate_proposal
from tests.agent_tools.adaptive_hparam_test_support import _agent_recipe
from tests.agent_tools.test_agent_tools_adaptive_proposals import _proposal, _snapshot
from tests.agent_tools.test_agent_tools_hparam_preflight import _workspace_files


@pytest.mark.parametrize("alias", ["00", "01", "-0", "+1", " 1", "１"])
@pytest.mark.parametrize("intermediate", [False, True])
def test_list_index_aliases_fail_without_mutation(alias: str, intermediate: bool):
    config = {"items": [{"value": 10}, {"value": 20}]}
    before = copy.deepcopy(config)
    pointer = f"/items/{alias}" + ("/value" if intermediate else "")

    with pytest.raises(ValueError, match="list index must use canonical integer spelling"):
        plan_hparam.set_json_pointer(config, pointer, 30 if intermediate else {"value": 30})

    assert config == before


@pytest.mark.parametrize("index", ["0", "1"])
def test_canonical_list_indices_support_assignment_and_traversal(index: str):
    config = {"items": [{"value": 10}, {"value": 20}]}
    untouched = copy.deepcopy(config["items"][1 - int(index)])

    plan_hparam.set_json_pointer(config, f"/items/{index}", {"value": 30})
    assert config["items"][int(index)] == {"value": 30}
    plan_hparam.set_json_pointer(config, f"/items/{index}/value", 40)

    assert config["items"][int(index)] == {"value": 40}
    assert config["items"][1 - int(index)] == untouched


def test_numeric_dictionary_keys_remain_distinct_literals():
    config = {"1": {"value": 10}, "01": {"value": 20}, "0": 30, "-0": 40}

    plan_hparam.set_json_pointer(config, "/01/value", 21)
    plan_hparam.set_json_pointer(config, "/1/value", 11)
    plan_hparam.set_json_pointer(config, "/-0", 41)

    assert config == {"1": {"value": 11}, "01": {"value": 21}, "0": 30, "-0": 41}


@pytest.mark.parametrize("pointer", ["/a~2/value", "/a~/value", "/a~~0/value"])
def test_json_pointer_rejects_invalid_raw_tilde_escapes(pointer: str):
    config = {"a~2": {"value": 10}, "a~": {"value": 20}, "a~~": {"value": 30}}
    before = copy.deepcopy(config)

    with pytest.raises(ValueError, match="JSON Pointer escapes must use ~0 or ~1"):
        plan_hparam.set_json_pointer(config, pointer, 40)

    assert config == before


def test_json_pointer_escapes_preserve_slashes_and_literal_tilde_sequences():
    config = {"a/b": {"~2": [{"value": 10}], "~02": 20}}

    plan_hparam.set_json_pointer(config, "/a~1b/~02/0/value", 11)
    plan_hparam.set_json_pointer(config, "/a~1b/~002", 21)

    assert config == {"a/b": {"~2": [{"value": 11}], "~02": 21}}


@pytest.mark.parametrize("intermediate", [False, True])
def test_proposal_preflight_rejects_list_alias_before_acceptance(tmp_path: Path, monkeypatch, intermediate: bool):
    recipe_path = _agent_recipe(tmp_path / "source")
    recipe = yaml.safe_load(recipe_path.read_bytes())
    base = yaml.safe_load(Path(recipe["base_recipe"]).read_bytes())
    source_config = yaml.safe_load(Path(base["inputs"]["config"]).read_bytes())
    channel = source_config["model"]["channels"][0]
    alias_key = "yaml:/model/channels/00" + ("/input_dim" if intermediate else "")
    alias_value = 16 if intermediate else {**channel, "input_dim": 16}
    parameters = {
        "runtime.lr": [1e-6],
        "yaml:/model/channels/0": [channel],
        alias_key: [alias_value],
    }
    snapshot = _snapshot(parameters=parameters)
    validated = validate_proposal(_proposal(snapshot, parameters=parameters), snapshot)
    validated_before = copy.deepcopy(validated)
    recipe["search"]["parameters"] = parameters
    candidate = adaptive_hparam._agent_suggestion_payload(recipe, {"recipe_path": str(recipe_path)}, 1, validated)
    candidate_before = copy.deepcopy(candidate)
    workspace = Path(base["experiment"]["root"])
    before = _workspace_files(workspace)
    next_dir = workspace / "adaptive" / "rounds" / "round_001"

    def reject_target_inspection(*_args, **_kwargs):
        pytest.fail("Invalid pointer must fail before execution-target inspection")

    monkeypatch.setattr(managed_scheduler, "inspect_execution_target", reject_target_inspection)

    with pytest.raises(RuntimeError, match="Agent proposal failed preflight.*canonical integer spelling"):
        adaptive_hparam._preflight_candidate(yaml.safe_dump(candidate).encode(), next_dir, "Agent proposal")

    assert validated == validated_before
    assert candidate == candidate_before
    assert _workspace_files(workspace) == before
    assert not next_dir.exists()
    assert not (workspace / "adaptive" / "proposals").exists()
