from copy import deepcopy
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest

from agent_tools import experiment_workspace, experiments


@pytest.mark.parametrize("scoped", [False, True])
def test_normalized_research_record_preserves_order_and_omits_empty_optionals(tmp_path, monkeypatch, scoped):
    entry = {
        "id": "obs-001",
        "recorded_at": "2026-09-07T01:02:03+00:00",
        "kind": "observation",
        "title": "Observed result",
        "actor": "agent:test",
        "source": "test",
        "evidence": [
            {"label": "second", "locator": "../reports/second.md", "sha256": "a" * 64},
            {"label": "first", "locator": "~/reports/first.md"},
        ],
        "body": "  First line.\r\nSecond line.\rThird line.  ",
        "authority": None,
        "supersedes": [],
    }
    if scoped:
        entry["scope"] = {"step_id": "train", "run_ids": ["run-002", "run-001"]}
        entry["occurred_at"] = "2026-09-06T01:02:03+00:00"
    original = deepcopy(entry)
    calls = []

    def read_step(root, step_id, *, remote):
        calls.append((root, step_id, remote))
        return {"experiment_id": "unit"}

    monkeypatch.setattr(experiment_workspace, "read_step_manifest", read_step)
    normalized = experiment_workspace._normalized_research_log_entry(
        entry,
        experiment_id="unit",
        managed_rows=[{"step_id": "train", "run_id": run_id} for run_id in ("run-001", "run-002")],
        root=tmp_path,
        remote="test-host",
    )
    expected = {key: value for key, value in original.items() if key not in {"authority", "supersedes"}}
    expected["recorded_at"] = "2026-09-07T01:02:03Z"
    expected["body"] = "First line.\nSecond line.\nThird line."
    if scoped:
        expected["occurred_at"] = "2026-09-06T01:02:03Z"
    assert normalized == expected
    assert entry == original
    assert normalized["evidence"] is not entry["evidence"]
    assert normalized["evidence"][0] is not entry["evidence"][0]
    assert calls == ([(tmp_path, "train", "test-host")] if scoped else [])
    if scoped:
        assert normalized["scope"]["run_ids"] is not entry["scope"]["run_ids"]
    block = experiment_workspace._research_log_block(normalized, "unit")
    assert block.endswith("### Record\n\nFirst line.\nSecond line.\nThird line.\n\n")
    assert block.index("Label: second") < block.index("Label: first")
    assert "Locator: ../reports/second.md\n" in block
    assert "Locator: ~/reports/first.md\n" in block
    assert "Authority:" not in block and "Supersedes:" not in block
    if scoped:
        assert "- Runs:\n  - `run-002`\n  - `run-001`\n" in block


def test_step_merge_retains_dynamic_values_and_raw_reader_identity(tmp_path, monkeypatch):
    recipe_path = Path("relative/recipe.yaml")
    inputs = {"data": ["index.csv"]}
    existing = {
        "step": {"id": "train", "phase": "train", "purpose": 7, "inputs": inputs},
        "experiment_id": 12,
        "recipe_path": recipe_path,
        "plans": [Path("/plans/b")],
    }
    merged = experiment_workspace.merge_step_manifest(existing, {"plans": ["/plans/b", Path("/plans/a")]})
    assert merged == {**existing, "plan_controller": "", "plans": ["/plans/b", "/plans/a"]}
    assert merged["step"]["inputs"] is inputs
    assert merged["recipe_path"] is recipe_path
    assert existing["plans"] == [Path("/plans/b")]
    payload = {**merged, "recipe_path": "", "plan_controller": "ordinary"}
    monkeypatch.setattr(experiment_workspace, "read_managed_yaml_mapping", lambda *args, **kwargs: payload)
    read = experiment_workspace._validated_step_manifest("fixture", tmp_path / "step.yaml", "train")
    assert read is payload
    assert read["step"]["purpose"] == 7
    assert read["experiment_id"] == 12
    assert read["step"]["inputs"] is inputs


@pytest.mark.parametrize("raw", [None, [], 17, {"title": 17}, {"status": "completed"}])
def test_completion_normalization_without_permission_passes_raw_value_through(tmp_path, raw):
    normalized, bindings = experiments._normalized_completed_metadata(raw, root=tmp_path, allow_completed=False)
    assert normalized is raw
    assert bindings == []


@pytest.mark.parametrize("binding", ["historical", "final", "selection", "null-selection"])
def test_completion_normalization_preserves_raw_values_and_binding_order(tmp_path, binding):
    baseline = {"type": "none", "rationale": "fixture"}
    core = {"id": 12, "title": 17, "objective": ["raw"], "root": tmp_path, "baseline": baseline}
    raw = {**core, "status": "completed", "completed_at": "2026-09-07T01:02:03Z"}
    expected_bindings = []
    if binding != "historical":
        raw.update(final_report=str(tmp_path / "reports/final.md"), final_report_sha256="a" * 64)
        expected_bindings.append(tmp_path / "reports/final.md")
    if binding in {"selection", "null-selection"}:
        raw["selection_report_sha256"] = "b" * 64 if binding == "selection" else None
    if binding == "selection":
        expected_bindings.append(tmp_path / "reports/hparam_selection.md")
    before = deepcopy(raw)
    normalized, bindings = experiments._normalized_completed_metadata(raw, root=tmp_path, allow_completed=True)
    assert normalized == core
    assert normalized["baseline"] is baseline
    assert normalized["objective"] is raw["objective"]
    assert bindings == expected_bindings
    assert raw == before


def test_experiment_record_types_reach_callers(tmp_path):
    probe = tmp_path / "experiment_record_probe.py"
    probe.write_text(
        textwrap.dedent("""\
            from pathlib import Path
            from typing import Any, Literal
            from agent_tools import experiment_workspace as workspace, experiments, research_log

            note = experiments.append_experiment_note("/workspace", "/entry.yaml")
            path: str = note["path"]
            appended: bool = note["appended"]
            entry_id: str = note["entry_id"]
            note["entry_ids"]  # type: ignore[typeddict-item]
            note["appended"] = "yes"  # type: ignore[typeddict-item]
            completion: experiments.ExperimentCompletionFields = {
                "status": "completed", "completed_at": "timestamp",
                "final_report": "/final.md", "final_report_sha256": "digest",
            }
            completed: Literal["completed"] = completion["status"]
            completion["status"] = "running"  # type: ignore[typeddict-item]
            completion["final_report_sha256"] = None  # type: ignore[typeddict-item]
            missing: experiments.ExperimentCompletionFields = {"status": "completed"}  # type: ignore[typeddict-item]

            merged = workspace.merge_step_manifest({}, {})
            plans: list[str] = merged["plans"]
            controller: str = merged["plan_controller"]
            merged["plan"]  # type: ignore[typeddict-item]
            merged["plans"] = [1]  # type: ignore[list-item]
            merged["plan_controller"] = None  # type: ignore[typeddict-item]
            merged["experiment_id"] = 12
            merged["recipe_path"] = Path("relative.yaml")
            merged["step"]["purpose"] = 7
            workspace.validate_step_registration("/workspace", merged)
            committed, created = workspace.commit_step_manifest("/workspace", merged)
            was_created: bool = created
            committed["plans"] = [1]  # type: ignore[list-item]
            raw = workspace.read_step_manifest("/workspace", "train")
            if raw is not None:
                raw["unvalidated_extra"] = None

            entry = workspace._normalized_research_log_entry(
                {}, experiment_id="unit", managed_rows=[], root=Path("/workspace"), remote=None,
            )
            normalized: research_log.NormalizedResearchLogEntry = entry
            body: str = normalized["body"]
            normalized["body"] = None  # type: ignore[typeddict-item]
            normalized["evidences"]  # type: ignore[typeddict-item]
            evidence = normalized["evidence"][0]
            locator: str = evidence["locator"]
            evidence["sha256"] = None  # type: ignore[typeddict-item]
            minimal_evidence: research_log.ResearchLogEvidence = {"label": "report", "locator": "../report"}
            scope: research_log.ResearchLogScope = {"step_id": "train"}
            scope["run_ids"] = [1]  # type: ignore[list-item]
            no_step: research_log.ResearchLogScope = {"run_ids": ["run-001"]}  # type: ignore[typeddict-item]
            research_log._research_log_block(normalized, "unit")
            experiment_workspace_block: str = workspace._research_log_block(normalized, "unit")
            raw_input: Any = None
            experiments._normalized_completed_metadata(raw_input, root=Path("/workspace"), allow_completed=True)
            """),
        encoding="utf-8",
    )
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mypy",
            "--config-file",
            str(root / "pyproject.toml"),
            "--follow-imports=silent",
            "--warn-unused-ignores",
            "--no-incremental",
            str(probe),
        ],
        cwd=root,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
