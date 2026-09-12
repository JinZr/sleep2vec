from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from agent_tools import adaptive_evidence, adaptive_hparam
from agent_tools.experiment_workspace import merge_run_manifest
from tests.agent_tools import adaptive_hparam_test_support as test_support
from tests.agent_tools.adaptive_hparam_test_support import (
    _adaptive_recipe,
    _mark_round_terminal,
    _read_table,
    _write_fake_manifest,
)

_stub_execution_snapshot_preflight = test_support._stub_execution_snapshot_preflight


@pytest.mark.parametrize(
    ("metrics", "expected"),
    [({}, None), ({"val_ahi_pearson": 0.42}, 0.42), ({"val_ahi_pearson": None}, None)],
)
def test_manifest_metrics_preserves_monitor_score_without_synthesizing_a_named_metric(metrics, expected):
    row = adaptive_evidence.manifest_metrics(
        {"monitor": "val_ahi_pearson", "best_model_score": 0.73, "metrics": metrics}
    )

    assert ("val_ahi_pearson" in row) == ("val_ahi_pearson" in metrics)
    assert row.get("val_ahi_pearson") == expected
    assert row["best_model_score"] == 0.73
    assert row["monitor"] == "val_ahi_pearson"


@pytest.mark.parametrize(
    "monitor_evidence",
    ["missing_metric", "explicit_metric", "different_monitor", "explicit_null"],
)
def test_val_digest_uses_only_matching_nonempty_monitor_evidence(tmp_path: Path, monitor_evidence: str):
    recipe_path = _adaptive_recipe(tmp_path, test_feedback=False)
    recipe = yaml.safe_load(recipe_path.read_text())
    recipe["adaptive"]["objective_metric"] = "val_ahi_pearson"
    recipe["evaluation_policy"]["test_after_fit"] = False
    recipe["decisions"]["test_after_fit"] = {"value": False, "source": "explicit_recipe"}
    recipe_path.write_text(yaml.safe_dump(recipe))
    workflow = adaptive_hparam.init_adaptive_workflow(recipe_path, tmp_path / "workflow")
    _write_fake_manifest(workflow)
    _mark_round_terminal(workflow, tmp_path)
    run = json.loads((workflow / "adaptive" / "rounds" / "round_000" / "plan.json").read_text())["runs"][0]
    manifest_path = Path(run["runtime_dir"]) / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["status"] = "skipped_test"
    manifest["metrics"] = {}
    manifest["best_model_score"] = 0.73
    if monitor_evidence == "different_monitor":
        manifest["monitor"] = "val_loss"
    elif monitor_evidence == "explicit_null":
        manifest["metrics"]["val_ahi_pearson"] = None
    elif monitor_evidence == "explicit_metric":
        manifest["metrics"]["val_ahi_pearson"] = 0.42
    manifest_path.write_text(json.dumps(manifest))

    if monitor_evidence in {"different_monitor", "explicit_null"}:
        with pytest.raises(ValueError, match="lacks finite val_ahi_pearson objective evidence"):
            adaptive_hparam.digest_hparam_run(workflow)
        assert not (workflow / "adaptive" / "digests" / "round_000.csv").exists()
        return

    digest_path = adaptive_hparam.digest_hparam_run(workflow)

    row = _read_table(digest_path)[0]
    assert row["val_ahi_pearson"] == ("0.42" if monitor_evidence == "explicit_metric" else "0.73")
    assert "test_auroc" not in row
    assert row["best_model_score"] == "0.73"
    assert row["monitor"] == "val_ahi_pearson"
    assert row["checkpoint_path"] == str(Path(run["checkpoint_dir"]) / "epoch=3.ckpt")


@pytest.mark.parametrize(
    ("statuses", "named_metrics", "winner"),
    [
        (("failed", "finished"), False, 1),
        (("stopped", "completed"), False, 1),
        (("failed", "stopped"), False, None),
        (("running", "running"), True, 0),
    ],
)
def test_monitor_fallback_respects_canonical_state_without_blocking_live_metrics(
    tmp_path: Path, monkeypatch, statuses: tuple[str, str], named_metrics: bool, winner: int | None
):
    recipe_path = _adaptive_recipe(tmp_path, test_feedback=False)
    recipe = yaml.safe_load(recipe_path.read_text())
    recipe["search"]["max_runs"] = 2
    recipe["search"]["parameters"]["runtime.lr"] = [1e-6, 3e-6]
    recipe["adaptive"].update({"objective_metric": "val_ahi_pearson", "round_size": 3, "max_runs_total": 5})
    recipe["evaluation_policy"]["test_after_fit"] = False
    recipe["decisions"]["test_after_fit"] = {"value": False, "source": "explicit_recipe"}
    recipe_path.write_text(yaml.safe_dump(recipe))
    workflow = adaptive_hparam.init_adaptive_workflow(recipe_path, tmp_path / "workflow")
    _write_fake_manifest(workflow)
    runs = json.loads((workflow / "adaptive" / "rounds" / "round_000" / "plan.json").read_text())["runs"]
    for run, score in zip(runs, [0.95, 0.73]):
        checkpoint_dir = Path(run["checkpoint_dir"])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / "epoch=3.ckpt"
        checkpoint_path.write_text("checkpoint")
        manifest = {
            "version": run["version"],
            "status": "skipped_test",
            "monitor": "val_ahi_pearson",
            "monitor_mode": "max",
            "best_model_score": score,
            "best_model_path": str(checkpoint_path),
            "metrics": {"val_ahi_pearson": score} if named_metrics else {},
        }
        (Path(run["runtime_dir"]) / "run_manifest.json").write_text(json.dumps(manifest))
    merge_run_manifest(
        tmp_path,
        [
            {
                "step_id": run["step_id"],
                "run_id": run["run_id"],
                "status": status,
                "stop_reason": "Stopped for diagnosis." if status == "stopped" else "",
            }
            for run, status in zip(runs, statuses)
        ],
    )
    monkeypatch.setattr(adaptive_hparam, "monitor_hparam_runs", lambda _run_dir: None)

    digest_path = adaptive_hparam.digest_hparam_run(workflow)

    rows = _read_table(digest_path)
    assert [row["status"] for row in rows] == list(statuses)
    assert [row["best_model_score"] for row in rows] == ["0.95", "0.73"]
    incumbent_path = workflow / "adaptive" / "incumbents.tsv"
    markdown = digest_path.with_suffix(".md").read_text()
    if winner is None:
        assert all("val_ahi_pearson" not in row for row in rows)
        assert not incumbent_path.exists()
        assert not any(line.startswith("- ") for line in markdown.splitlines())
        with pytest.raises(ValueError, match="No digest rows with finite val_ahi_pearson"):
            adaptive_hparam.suggest_next_round(workflow, digest_path=digest_path)
        assert not (workflow / "adaptive" / "suggestions" / "round_001.yaml").exists()
        return

    assert [row["val_ahi_pearson"] for row in rows] == ["0.95" if named_metrics else "", "0.73"]
    incumbent = _read_table(incumbent_path)[-1]
    assert incumbent["run_id"] == runs[winner]["run_id"]
    assert incumbent["objective_score"] == ["0.95", "0.73"][winner]
    assert f"- {runs[winner]['run_id']}:" in markdown
    if not named_metrics:
        assert f"- {runs[0]['run_id']}:" not in markdown

    suggestion_path = adaptive_hparam.suggest_next_round(workflow, digest_path=digest_path)

    suggestion = yaml.safe_load(suggestion_path.read_text())
    expected_lr = [5e-7, 1e-6, 1.5e-6] if winner == 0 else [1.5e-6, 3e-6, 4.5e-6]
    assert suggestion["search"]["parameters"]["runtime.lr"] == expected_lr
