from pathlib import Path

from agent_tool_test_helpers import run_cli, write_finetune_recipe, write_yaml
import pytest
from test_agent_plan_blocks_on_ambiguity import _hparam_recipe
import yaml

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


def test_hparam_doctor_blocks_scratch_init_from(tmp_path):
    recipe = _hparam_recipe(tmp_path, variant="sleep2vec2")
    _add_init_from(tmp_path / "config.yaml")

    result = run_cli("doctor", "--recipe", str(recipe))

    assert result.returncode == 1
    assert "base_finetune.pretrained_backbone_path" in result.stdout
    assert "set init_from" in result.stdout
