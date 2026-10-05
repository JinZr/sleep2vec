from __future__ import annotations

import ast
from pathlib import Path
import sys

import pytest

import sex_age_baseline.finetune as baseline_finetune
import sex_age_baseline.infer as baseline_infer

REPO_ROOT = Path(__file__).resolve().parents[2]
# Parser semantics that must match; help text may name the baseline model instead of sleep2vec.
COMPARED_KEYWORDS = ("default", "choices", "nargs", "type", "action", "dest", "required")
# The shared-data contract change removes this exemption by giving the baseline dataset-name overrides.
INFER_EXEMPT_OPTIONS = {"--override-dataset-names"}


def _parser_options(relative_path: str) -> dict[str, dict[str, str]]:
    options = {}
    for node in ast.walk(ast.parse((REPO_ROOT / relative_path).read_text())):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
            option = next(arg.value for arg in node.args if isinstance(arg, ast.Constant))
            options[option] = {
                keyword.arg: ast.unparse(keyword.value) for keyword in node.keywords if keyword.arg in COMPARED_KEYWORDS
            }
    return options


@pytest.mark.parametrize(
    ("reference", "baseline", "exempt"),
    [
        ("sleep2vec/finetune.py", "sex_age_baseline/finetune.py", set()),
        ("sleep2vec/infer.py", "sex_age_baseline/infer.py", INFER_EXEMPT_OPTIONS),
    ],
)
def test_baseline_cli_matches_sleep2vec_options(reference: str, baseline: str, exempt: set[str]):
    expected = _parser_options(reference)
    actual = _parser_options(baseline)

    assert exempt <= set(expected)
    assert set(actual) == set(expected) - exempt
    assert actual == {option: expected[option] for option in actual}


def _parse(module, monkeypatch: pytest.MonkeyPatch, argv: list[str]):
    monkeypatch.setattr(sys, "argv", [module.__name__, *argv])
    return module.parse_args()


def test_finetune_accepts_sleep2vec_runtime_flags(monkeypatch: pytest.MonkeyPatch):
    args = _parse(
        baseline_finetune,
        monkeypatch,
        [
            "--config",
            "cox.yaml",
            "--label-name",
            "cox",
            "--results-csv-path",
            "results.csv",
            "--lr-scheduler",
            "wsd",
            "--lr-decay-ratio",
            "0.2",
            "--device",
            "cuda:1",
            "--devices",
            "1",
            "--version-prefix",
            "baseline",
            "--version-tag",
            "seed0",
            "--export-predictions",
        ],
    )

    assert (args.lr_scheduler, args.lr_decay_ratio, args.device, args.devices) == ("wsd", 0.2, "cuda:1", [1])
    assert (args.version_prefix, args.version_tag, args.export_predictions) == ("baseline", "seed0", True)
    # Like sleep2vec finetuning, training seeding is fixed rather than a CLI option.
    assert not hasattr(args, "seed")


@pytest.mark.parametrize(
    "unsupported",
    [
        ["--pretrained-backbone-path", "backbone.ckpt"],
        ["--print-diagnostics"],
        ["--diagnostics-steps", "3"],
    ],
)
def test_finetune_rejects_sleep2vec_only_capabilities(monkeypatch: pytest.MonkeyPatch, unsupported: list[str]):
    argv = ["--config", "cox.yaml", "--label-name", "cox", "--results-csv-path", "results.csv", *unsupported]

    with pytest.raises(ValueError, match=f"does not support {unsupported[0]}"):
        _parse(baseline_finetune, monkeypatch, argv)


def test_infer_accepts_sleep2vec_averaging_and_wandb_flags(monkeypatch: pytest.MonkeyPatch):
    args = _parse(
        baseline_infer,
        monkeypatch,
        [
            "--config",
            "cox.yaml",
            "--ckpt-path",
            "last",
            "--label-name",
            "cox",
            "--avg-ckpts",
            "3",
            "--avg-ckpt-dir",
            "checkpoints",
            "--wandb",
            "--wandb-name",
            "eval",
            "--no-wandb-artifact",
        ],
    )

    assert (args.ckpt_path, args.avg_ckpts, args.avg_ckpt_dir) == ("last", 3, Path("checkpoints"))
    assert (args.wandb, args.wandb_name, args.wandb_artifact, args.seed) == (True, "eval", False, 4523)


def test_infer_rejects_pretrained_backbone(monkeypatch: pytest.MonkeyPatch):
    argv = ["--config", "cox.yaml", "--ckpt-path", "model.ckpt", "--label-name", "cox"]

    with pytest.raises(ValueError, match="does not support --pretrained-backbone-path"):
        _parse(baseline_infer, monkeypatch, [*argv, "--pretrained-backbone-path", "backbone.ckpt"])
