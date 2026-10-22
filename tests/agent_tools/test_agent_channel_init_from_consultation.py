import copy
import hashlib
from pathlib import Path
import sys

from agent_tool_test_helpers import config_payload, run_cli, write_finetune_recipe, write_yaml
import pytest
from test_agent_plan_blocks_on_ambiguity import _hparam_recipe
import yaml

from agent_tools import plan_contract, plan_hparam
from agent_tools.configs import config_summary
from agent_tools.decision_models import ResolvedDecision
from agent_tools.decision_paths import finetune_init_from_source_issue
from agent_tools.decisions import DecisionStatus, evaluate_consultation_gates
from agent_tools.recipes import load_consultation_policy


def _add_init_from(config_path: Path) -> None:
    payload = yaml.safe_load(config_path.read_text())
    payload["model"]["channels"][0]["init_from"] = "eeg_original"
    write_yaml(config_path, payload)


def _scratch_recipe(tmp_path: Path, *, init_from: bool = True) -> dict:
    recipe_path = write_finetune_recipe(tmp_path, variant="sleep2vec2")
    if init_from:
        _add_init_from(tmp_path / "config.yaml")
    return yaml.safe_load(recipe_path.read_text())


def _gate(tmp_path: Path, recipe: dict):
    summary = config_summary(tmp_path / "config.yaml", variant="sleep2vec2")
    return evaluate_consultation_gates("finetune", recipe, summary, {}, load_consultation_policy())


def _init_from_messages(report) -> list[str]:
    return [issue.message for issue in report.issues if "set init_from" in issue.message]


def test_scratch_finetune_with_init_from_fails_consultation(tmp_path):
    report = _gate(tmp_path, _scratch_recipe(tmp_path))

    assert report.status is DecisionStatus.FAIL
    matches = [
        issue
        for issue in report.issues
        if issue.field == "pretrained_backbone_path" and issue.status is DecisionStatus.FAIL
    ]
    assert len(matches) == 1
    issue = matches[0]
    assert "['ppg'] set init_from" in issue.message
    assert issue.evidence["init_from_channels"] == ["ppg"]
    assert issue.evidence["preflight_before_workspace"] is True


def _set_backbone(recipe: dict, tmp_path: Path) -> None:
    backbone = tmp_path / "backbone.ckpt"
    backbone.write_bytes(b"")
    recipe["inputs"]["pretrained_backbone_path"] = str(backbone)
    recipe["decisions"]["pretrained_backbone_path"]["value"] = str(backbone)


def _set_ckpt(recipe: dict, tmp_path: Path) -> None:
    ckpt = tmp_path / "resume.ckpt"
    ckpt.write_bytes(b"")
    recipe["inputs"]["ckpt_path"] = str(ckpt)


def _set_init_checkpoint(recipe: dict, tmp_path: Path) -> None:
    init = tmp_path / "init.ckpt"
    init.write_bytes(b"")
    recipe["inputs"]["init_checkpoint_path"] = str(init)


@pytest.mark.parametrize("edit", [_set_backbone, _set_ckpt, _set_init_checkpoint])
def test_init_from_accepts_each_weight_source(tmp_path, edit):
    recipe = _scratch_recipe(tmp_path)
    edit(recipe, tmp_path)

    assert _init_from_messages(_gate(tmp_path, recipe)) == []


def test_unresolved_backbone_is_asked_not_failed(tmp_path):
    recipe = _scratch_recipe(tmp_path)
    del recipe["decisions"]["pretrained_backbone_path"]

    report = _gate(tmp_path, recipe)

    assert any(
        issue.field == "pretrained_backbone_path" and issue.status is DecisionStatus.NEEDS_USER_INPUT
        for issue in report.issues
    )
    assert _init_from_messages(report) == []


def test_config_without_init_from_is_unchanged(tmp_path):
    recipe = _scratch_recipe(tmp_path, init_from=False)

    assert _init_from_messages(_gate(tmp_path, recipe)) == []


def test_helper_treats_ask_user_as_pending_source():
    summary = {"model": {"channels": [{"name": "ppg", "init_from": "eeg_original"}]}}
    ckpt = ResolvedDecision("ckpt_path", None, "missing", "none", {})
    ask = ResolvedDecision("pretrained_backbone_path", "ASK_USER", "explicit_recipe", "high", {"value": "ASK_USER"})

    assert (
        finetune_init_from_source_issue({"inputs": {}}, summary, {"pretrained_backbone_path": ask, "ckpt_path": ckpt})
        is None
    )

    scratch = ResolvedDecision("pretrained_backbone_path", None, "explicit_recipe", "high", {"value": None})
    issue = finetune_init_from_source_issue(
        {"inputs": {}}, summary, {"pretrained_backbone_path": scratch, "ckpt_path": ckpt}
    )
    assert issue is not None
    assert issue.status is DecisionStatus.FAIL
    assert issue.evidence["sources"] == {
        "pretrained_backbone_path": None,
        "ckpt_path": None,
        "init_checkpoint_path": None,
    }


def test_helper_counts_ckpt_path_only_for_direct_finetune():
    summary = {"model": {"channels": [{"name": "ppg", "init_from": "eeg_original"}]}}
    scratch = ResolvedDecision("pretrained_backbone_path", None, "explicit_recipe", "high", {"value": None})
    ckpt = ResolvedDecision("ckpt_path", "x", "explicit_recipe", "high", {"value": "x"})
    decisions = {"pretrained_backbone_path": scratch, "ckpt_path": ckpt}

    assert finetune_init_from_source_issue({"inputs": {"ckpt_path": "x"}}, summary, decisions) is None

    issue = finetune_init_from_source_issue(
        {"inputs": {"ckpt_path": "x"}, "search": {"parameters": {"runtime.lr": [1e-6]}}}, summary, decisions
    )
    assert issue is not None
    assert issue.evidence["sources"]["ckpt_path"] is None
    assert "final-evaluation-only" in issue.message


def test_hparam_doctor_blocks_scratch_init_from(tmp_path):
    recipe = _hparam_recipe(tmp_path, variant="sleep2vec2")
    _add_init_from(tmp_path / "config.yaml")

    result = run_cli("doctor", "--recipe", str(recipe))

    assert result.returncode == 1
    assert "base_finetune.pretrained_backbone_path" in result.stdout
    assert "set init_from" in result.stdout


def test_hparam_doctor_ignores_final_evaluation_ckpt_path_as_init_from_source(tmp_path):
    ckpt = tmp_path / "final.ckpt"
    ckpt.write_bytes(b"")
    recipe = _hparam_recipe(tmp_path, variant="sleep2vec2", ckpt_path=ckpt)
    _add_init_from(tmp_path / "config.yaml")

    result = run_cli("doctor", "--recipe", str(recipe))

    assert result.returncode == 1
    assert "base_finetune.pretrained_backbone_path" in result.stdout
    assert "set init_from" in result.stdout


def _hparam_recipe_with_init_from_override(tmp_path: Path) -> Path:
    base_channel = config_payload(tmp_path / "index.csv")["model"]["channels"][0]
    parameters = {"yaml:/model/channels/0": [base_channel, {**base_channel, "init_from": "eeg_original"}]}
    return _hparam_recipe(tmp_path, variant="sleep2vec2", parameters=parameters, max_runs=2)


def test_hparam_override_adding_init_from_without_source_fails_doctor(tmp_path):
    recipe = _hparam_recipe_with_init_from_override(tmp_path)

    result = run_cli("doctor", "--recipe", str(recipe))

    assert result.returncode == 1
    assert "hparam_search_space" in result.stdout
    assert "sets init_from on model.channels" in result.stdout


def test_hparam_override_adding_init_from_with_backbone_passes_override_check(tmp_path):
    recipe_path = _hparam_recipe_with_init_from_override(tmp_path)
    base_path = Path(yaml.safe_load(recipe_path.read_text())["base_recipe"])
    base = yaml.safe_load(base_path.read_text())
    _set_backbone(base, tmp_path)
    write_yaml(base_path, base)

    result = run_cli("doctor", "--recipe", str(recipe_path))

    assert "sets init_from on model.channels" not in result.stdout


def test_compile_hparam_run_contracts_rejects_init_from_candidate_without_source(tmp_path):
    source = tmp_path / "source.yaml"
    payload = config_payload(tmp_path / "index.csv")
    captured = yaml.safe_dump(payload, sort_keys=False).encode()
    source.write_bytes(captured)
    channel = {**payload["model"]["channels"][0], "init_from": "eeg_original"}
    recipe = {
        "variant": "sleep2vec2",
        "inputs": {"config": str(source), "label_name": "ahi", "ckpt_path": str(tmp_path / "final.ckpt")},
        "experiment": {"id": "unit-candidates"},
        "step": {"id": "tune"},
        "execution": {"python": sys.executable, "runtime_commit": "a" * 40, "workdir": str(tmp_path)},
        "evaluation_policy": {"test_after_fit": False, "selection_split": "val"},
        "search": {"configurations": [{"yaml:/model/channels/0": channel}]},
    }
    plan_contract.bind_plan_context(recipe)
    plan_contract.bind_frozen_input_snapshot(recipe, "inputs.config", source, hashlib.sha256(captured).hexdigest())

    with pytest.raises(ValueError, match="init_from"):
        plan_hparam.compile_hparam_run_contracts(recipe, tmp_path / "plan", 0, source_config_bytes=captured)


def test_hparam_pending_backbone_with_base_init_from_is_asked_not_failed(tmp_path):
    recipe_path = _hparam_recipe(tmp_path, variant="sleep2vec2")
    _add_init_from(tmp_path / "config.yaml")
    base_path = Path(yaml.safe_load(recipe_path.read_text())["base_recipe"])
    base = yaml.safe_load(base_path.read_text())
    del base["decisions"]["pretrained_backbone_path"]
    write_yaml(base_path, base)
    output_dir = tmp_path / "doctor"

    result = run_cli("doctor", "--recipe", str(recipe_path), "--output-dir", str(output_dir))

    assert result.returncode == 2
    assert "base_finetune.pretrained_backbone_path" in result.stdout
    assert "hparam_search_space" not in result.stdout
    assert (output_dir / "decisions.yaml").exists()


def test_trial_helper_checks_only_override_introduced_init_from():
    base = {
        "model": {
            "channels": [
                {"name": "ppg", "init_from": "eeg_original"},
                {"name": "eeg", "input_dim": 8},
            ]
        }
    }
    combo = {"yaml:/model/channels/1": {"name": "eeg", "input_dim": 8, "init_from": "eeg_original"}}

    plan_hparam.check_trial_init_from_sources({}, base, copy.deepcopy(base), {})

    run_config = copy.deepcopy(base)
    plan_hparam.apply_search_overrides(run_config, combo)
    with pytest.raises(ValueError, match=r"\['eeg'\]"):
        plan_hparam.check_trial_init_from_sources({}, base, run_config, combo)
