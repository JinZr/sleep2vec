from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from agent_tools import adaptive_hparam
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
    [({}, 0.73), ({"val_ahi_pearson": 0.42}, 0.42), ({"val_ahi_pearson": None}, None)],
)
def test_manifest_monitor_metric_uses_canonical_lookup_precedence(metrics, expected):
    row = adaptive_hparam._manifest_metrics(
        {"monitor": "val_ahi_pearson", "best_model_score": 0.73, "metrics": metrics}
    )

    assert row["val_ahi_pearson"] == expected
    assert row["best_model_score"] == 0.73


@pytest.mark.parametrize("monitor_evidence", ["missing_metric", "different_monitor", "explicit_null"])
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
    manifest_path.write_text(json.dumps(manifest))

    if monitor_evidence != "missing_metric":
        with pytest.raises(ValueError, match="lacks finite val_ahi_pearson objective evidence"):
            adaptive_hparam.digest_hparam_run(workflow)
        assert not (workflow / "adaptive" / "digests" / "round_000.csv").exists()
        return

    digest_path = adaptive_hparam.digest_hparam_run(workflow)

    row = _read_table(digest_path)[0]
    assert row["val_ahi_pearson"] == "0.73"
    assert "test_auroc" not in row
    assert row["best_model_score"] == "0.73"
    assert row["monitor"] == "val_ahi_pearson"
    assert row["checkpoint_path"] == str(Path(run["checkpoint_dir"]) / "epoch=3.ckpt")
