from __future__ import annotations

import copy

import pytest

from agent_tools import plan_hparam


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
