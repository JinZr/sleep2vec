from __future__ import annotations

import json
from pathlib import Path

from agent_tools import experiments


def test_experiment_note_normalizes_line_endings_and_preserves_authored_order(tmp_path: Path):
    spec = tmp_path / "experiment_spec.yaml"
    spec.write_text(
        "id: unit\n"
        "title: Unit experiment\n"
        "objective: Exercise research log normalization.\n"
        "baseline:\n"
        "  type: none\n"
        "  rationale: Unit fixture.\n"
    )
    root = tmp_path / "workspace"
    experiments.init_experiment(root, spec)
    step_dir = root / "steps" / "train"
    step_dir.mkdir(parents=True)
    (step_dir / "step.yaml").write_text(
        json.dumps(
            {
                "step": {"id": "train", "phase": "train", "purpose": "Train a candidate."},
                "experiment_id": "unit",
                "recipe_path": "",
                "plans": [],
            }
        )
    )
    (root / "run_manifest.tsv").write_text(
        "experiment_id\tstep_id\trun_id\tstatus\nunit\ttrain\trun-001\tcompleted\nunit\ttrain\trun-002\tcompleted\n"
    )
    entry = tmp_path / "entry.json"
    entry.write_text(
        json.dumps(
            {
                "id": "obs-001",
                "recorded_at": "2026-09-07T01:02:03+00:00",
                "occurred_at": "2026-09-06T01:02:03+00:00",
                "kind": "observation",
                "title": "Observed result",
                "actor": "agent:test",
                "source": "test",
                "evidence": [
                    {"label": "second", "locator": "../reports/second.md", "sha256": "a" * 64},
                    {"label": "first", "locator": "~/reports/first.md"},
                ],
                "body": "  First line.\r\nSecond line.\rThird line.  ",
                "scope": {"step_id": "train", "run_ids": ["run-002", "run-001"]},
                "authority": None,
                "supersedes": [],
            }
        )
    )

    assert experiments.append_experiment_note(root, entry)["appended"] is True

    text = (root / "RESEARCH_LOG.md").read_text()
    assert "- Recorded at: `2026-09-07T01:02:03Z`\n- Occurred at: `2026-09-06T01:02:03Z`\n" in text
    assert text.endswith("### Record\n\nFirst line.\nSecond line.\nThird line.\n\n")
    assert text.index("Label: second") < text.index("Label: first")
    assert "Locator: ../reports/second.md\n" in text
    assert "Locator: ~/reports/first.md\n" in text
    assert "- Runs:\n  - `run-002`\n  - `run-001`\n" in text
    assert "Authority:" not in text and "Supersedes:" not in text
