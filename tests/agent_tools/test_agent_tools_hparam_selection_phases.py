from pathlib import Path

import pytest

from agent_tools import hparam_selection


def test_hparam_selection_build_failure_does_not_start_publication(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        hparam_selection.artifacts,
        "read_hparam_plan",
        lambda _root: {"runs": [], "recipe": {"evaluation_policy": {}}},
    )
    publication_started = False

    def commit(_selection):
        nonlocal publication_started
        publication_started = True

    monkeypatch.setattr(hparam_selection, "_commit_hparam_selection", commit)

    with pytest.raises(ValueError, match="Recipe must define evaluation_policy.selection_metric and selection_mode"):
        hparam_selection.select_hparam_candidates("unused")

    assert publication_started is False


def test_hparam_selection_publication_order(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    workspace = tmp_path / "experiment"
    plan_root = tmp_path / "plan"
    ranking = workspace / "reports" / "ranking.csv"
    report = workspace / "reports" / "hparam_selection.md"
    audit = plan_root / "checkpoint_test_ranking.csv"
    run = {"step_id": "tune", "run_id": "run-001", "run_name": "candidate"}
    key = ("tune", "run-001")
    ranked = {
        **run,
        "metric": "test_score",
        "score": 0.9,
        "rank": 1,
        "checkpoint_path": "/checkpoints/epoch=1.ckpt",
        "checkpoint_sha256": "abc",
    }
    selected = {
        **ranked,
        "selection_task": "hparam_tune",
        "selection_mode": "max",
        "selection_split": "test",
    }
    selection = hparam_selection._HparamSelectionBuild(
        workspace=workspace,
        step_id="tune",
        metric="test_score",
        mode="max",
        selection_split="test",
        out=ranking,
        selection_report_out=report,
        report_run_keys={key},
        step_ranked=[ranked],
        all_ranked=[ranked],
        unscored_rows=[],
        checkpoint_audits_to_write=[(audit, [ranked])],
        current_registered=[(plan_root, {"runs": [run]})],
        plan_root_by_key={key: plan_root},
    )
    calls = []

    monkeypatch.setattr(hparam_selection, "write_rows", lambda path, _rows: calls.append(("write_rows", path)))
    monkeypatch.setattr(
        hparam_selection,
        "merge_run_manifest",
        lambda _workspace, _rows: calls.append(("merge_run_manifest", None)),
    )
    monkeypatch.setattr(
        hparam_selection,
        "read_run_manifest",
        lambda _workspace: calls.append(("read_run_manifest", None)) or [selected],
    )
    monkeypatch.setattr(
        hparam_selection,
        "_selection_report_steps",
        lambda _rows: calls.append(("selection_report_steps", None)) or [],
    )
    monkeypatch.setattr(
        hparam_selection.tracking,
        "hparam_selection_report_text",
        lambda _steps, *, root: calls.append(("render_report", root)) or "# Selection\n",
    )
    monkeypatch.setattr(
        hparam_selection.exp_io,
        "write_text_at",
        lambda path, _text: calls.append(("write_report", path)),
    )
    monkeypatch.setattr(
        hparam_selection,
        "_selection_event_exists",
        lambda _workspace, _payload: calls.append(("check_event", None)) or False,
    )
    monkeypatch.setattr(
        hparam_selection,
        "append_event",
        lambda _workspace, _event, _payload: calls.append(("append_event", None)),
    )

    assert hparam_selection._commit_hparam_selection(selection) == ranking
    assert calls == [
        ("write_rows", audit),
        ("write_rows", ranking),
        ("merge_run_manifest", None),
        ("read_run_manifest", None),
        ("selection_report_steps", None),
        ("render_report", workspace),
        ("write_report", report),
        ("merge_run_manifest", None),
        ("check_event", None),
        ("append_event", None),
    ]
