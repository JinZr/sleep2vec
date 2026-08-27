from __future__ import annotations

import pytest

from agent_tools import adaptive_hparam, checkpoint_test_results, hparam_selection


def test_checkpoint_test_results_preserve_manifest_order_and_numeric_scores():
    expected = checkpoint_test_results.expected_epoch_checkpoints(
        "/checkpoints",
        ["last.ckpt", "epoch=2-step=20.ckpt", "epoch=1-step=10.ckpt"],
        step_id="tune",
        run_id="run-001",
    )

    rows = checkpoint_test_results.validate_checkpoint_test_results(
        [
            {
                "checkpoint_path": "/checkpoints/epoch=2-step=20.ckpt",
                "epoch": 2,
                "metrics": {"test_metric": "0.75"},
            },
            {
                "checkpoint_path": "/checkpoints/epoch=1-step=10.ckpt",
                "epoch": 1,
                "metrics": {"test_metric": 0.5},
            },
        ],
        "test_metric",
        expected,
        step_id="tune",
        run_id="run-001",
    )

    assert list(expected.items()) == [
        ("/checkpoints/epoch=1-step=10.ckpt", 1),
        ("/checkpoints/epoch=2-step=20.ckpt", 2),
    ]
    assert rows == [
        {"checkpoint_path": "/checkpoints/epoch=2-step=20.ckpt", "epoch": 2, "score": 0.75},
        {"checkpoint_path": "/checkpoints/epoch=1-step=10.ckpt", "epoch": 1, "score": 0.5},
    ]


def test_checkpoint_test_results_keep_selection_error_text():
    expected = checkpoint_test_results.expected_epoch_checkpoints(
        "/checkpoints",
        ["epoch=1-step=10.ckpt", "epoch=2-step=20.ckpt"],
        step_id="tune",
        run_id="run-001",
    )

    with pytest.raises(
        ValueError,
        match=(
            r"^checkpoint_test_results is incomplete for tune / run-001: "
            r"/checkpoints/epoch=2-step=20\.ckpt$"
        ),
    ):
        checkpoint_test_results.validate_checkpoint_test_results(
            [
                {
                    "checkpoint_path": "/checkpoints/epoch=1-step=10.ckpt",
                    "epoch": 1,
                    "metrics": {"test_metric": 0.5},
                }
            ],
            "test_metric",
            expected,
            step_id="tune",
            run_id="run-001",
        )


def test_hparam_selection_keeps_evidence_and_hash_order(monkeypatch):
    calls = []
    original_expected = checkpoint_test_results.expected_epoch_checkpoints
    original_validate = checkpoint_test_results.validate_checkpoint_test_results

    def expected(*args, **kwargs):
        calls.append("expected")
        return original_expected(*args, **kwargs)

    def validate(*args, **kwargs):
        calls.append("validate")
        return original_validate(*args, **kwargs)

    def validate_evidence(_runs, _rows):
        calls.append("evidence")

    def checkpoint_sha(_run, _path):
        calls.append("sha256")
        return "a" * 64

    monkeypatch.setattr(checkpoint_test_results, "expected_epoch_checkpoints", expected)
    monkeypatch.setattr(checkpoint_test_results, "validate_checkpoint_test_results", validate)
    monkeypatch.setattr(hparam_selection.tracking, "validate_checkpoint_evidence_rows", validate_evidence)
    monkeypatch.setattr(hparam_selection.evidence, "checkpoint_file_sha256", checkpoint_sha)

    rows = hparam_selection._checkpoint_test_result_rows(
        {
            "step_id": "tune",
            "run_id": "run-001",
            "run_name": "lr-1e-4",
            "version": "unit",
            "checkpoint_dir": "/checkpoints",
        },
        "test_metric",
        "/run/run_manifest.json",
        {
            "test_all_checkpoints_after_fit": True,
            "checkpoint_test_results": [
                {
                    "checkpoint_path": "/checkpoints/epoch=1-step=10.ckpt",
                    "epoch": 1,
                    "metrics": {"test_metric": 0.5},
                }
            ],
        },
        ["epoch=1-step=10.ckpt"],
    )

    assert calls == ["expected", "evidence", "validate", "evidence", "sha256"]
    assert rows[0]["checkpoint_sha256"] == "a" * 64


def test_adaptive_checkpoint_objective_keeps_invalid_none_and_epoch_tie_break():
    manifest = {
        "test_all_checkpoints_after_fit": True,
        "checkpoint_test_results": [
            {
                "checkpoint_path": "/checkpoints/epoch=2-step=20.ckpt",
                "epoch": 2,
                "metrics": {"test_metric": 0.5},
            },
            {
                "checkpoint_path": "/checkpoints/epoch=1-step=10.ckpt",
                "epoch": 1,
                "metrics": {"test_metric": 0.5},
            },
        ],
    }
    objective = {"metric": "test_metric", "mode": "max"}
    checkpoint_names = ["epoch=1-step=10.ckpt", "epoch=2-step=20.ckpt"]

    selected = adaptive_hparam._test_checkpoint_objective(
        manifest,
        objective,
        "/checkpoints",
        checkpoint_names,
    )
    manifest["checkpoint_test_results"].pop()
    invalid = adaptive_hparam._test_checkpoint_objective(
        manifest,
        objective,
        "/checkpoints",
        checkpoint_names,
    )

    assert selected == {
        "checkpoint_path": "/checkpoints/epoch=1-step=10.ckpt",
        "epoch": 1,
        "score": 0.5,
    }
    assert invalid is None
