from __future__ import annotations

import ast
from contextlib import contextmanager
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

from agent_tool_test_helpers import (
    SUBPROCESS_WAIT_SECONDS,
    FakeLauncher,
    call_while_run_lock_holder_commits,
    run_cli,
    run_cli_subprocess,
    write_finetune_recipe,
    write_yaml,
)
import pytest
import yaml

from agent_tools import experiment_io, experiment_workspace, experiments, hparam_runtime, plans, run_artifacts
from agent_tools.experiment_workspace import (
    EXECUTION_IDENTITY_FIELDS,
    MANAGED_RUN_PATH_FIELDS,
    PROCESS_IDENTITY_FIELDS,
    SCHEDULER_BINDING_FIELDS,
    append_event,
    canonical_local_experiment_root,
    commit_step_manifest,
    ensure_experiment_workspace,
    file_sha256,
    initialize_run_manifest,
    managed_run_key,
    managed_run_parameters,
    merge_run_manifest,
    merge_run_row,
    merge_step_manifest,
    parameter_summary,
    read_run_manifest,
    read_step_manifest,
    resolve_run_row,
    run_identity,
    semantic_run_name,
    validate_frozen_run_update,
    validate_managed_run_rows,
    validate_scheduler_run_identity,
    validated_run_key,
)


def _hparam_recipe(tmp_path: Path) -> Path:
    base = write_finetune_recipe(tmp_path)
    return write_yaml(
        tmp_path / "tune.yaml",
        {
            "name": "managed_tune",
            "task": "hparam_tune",
            "variant": "sleep2vec",
            "base_recipe": str(base),
            "search": {"method": "grid", "max_runs": 1, "parameters": {"runtime.lr": [2e-6]}},
            "evaluation_policy": {
                "selection_metric": "val_ahi_pearson",
                "selection_mode": "max",
                "selection_split": "val",
                "external_test_locked": True,
                "test_after_fit": False,
                "final_eval_split": "test",
                "final_test_unlocked": False,
                "require_manual_unlock_for_final_test": True,
            },
            "decisions": {
                "task": {"value": "hparam_tune", "source": "explicit_recipe"},
                "label_name": {"value": "ahi", "source": "explicit_recipe"},
                "external_test_locked": {"value": True, "source": "explicit_recipe"},
                "train_val_test_policy": {"value": "val", "source": "explicit_recipe"},
                "overwrite_policy": {"value": False, "source": "explicit_recipe"},
                "final_eval_unlock": {"value": False, "source": "explicit_recipe"},
            },
        },
    )


def test_managed_plan_writes_semantic_run_workspace_without_schema_version(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path)
    plan_dir = tmp_path / "steps" / "tune" / "plan"

    result = run_cli("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir))
    assert result.returncode == 0, result.stderr
    plan = json.loads((plan_dir / "plan.json").read_text())
    run = plan["runs"][0]
    assert run["run_id"] == "run-000"
    assert run["run_name"] == "lr-2e-6"
    assert Path(run["run_dir"]).name == "run-000--lr-2e-6"
    assert Path(run["config"]).name == "config.yaml"
    assert Path(run["script"]).name == "launch.sh"
    assert plan["resolved_recipe_sha256"] == file_sha256(plan_dir / "recipe.resolved.yaml")
    assert (tmp_path / "experiment.yaml").exists()
    assert (tmp_path / "run_manifest.tsv").exists()
    assert not (tmp_path / "run_matrix.csv").exists()
    assert not (tmp_path / "reports" / "run_matrix.md").exists()
    assert "`RESEARCH_LOG.md`" in (tmp_path / "README.md").read_text()
    assert read_run_manifest(tmp_path)[0]["run_name"] == "lr-2e-6"
    managed_text = "\n".join(
        path.read_text()
        for path in [
            tmp_path / "experiment.yaml",
            tmp_path / "run_manifest.tsv",
            plan_dir / "recipe.resolved.yaml",
            plan_dir / "plan.json",
            Path(run["run_dir"]) / "run.json",
        ]
    )
    assert "schema_version" not in managed_text
    assert "runtime.lr" in managed_text
    assert "param.runtime.lr" not in managed_text


def test_new_plan_owned_workspace_initializes_research_log(tmp_path: Path):
    root = tmp_path / "workspace"
    recipe = {
        "experiment": {
            "id": "unit",
            "title": "Unit experiment",
            "objective": "Exercise research log initialization.",
            "root": str(root),
            "baseline": {"type": "none"},
        },
        "step": {
            "id": "prepare",
            "phase": "prepare",
            "purpose": "Prepare a plan.",
        },
    }

    ensure_experiment_workspace(recipe, root / "steps" / "prepare" / "plan")

    assert (root / "RESEARCH_LOG.md").read_text().startswith("# Research Log\n")
    assert "`RESEARCH_LOG.md`" in (root / "README.md").read_text()


def test_ordinary_step_rejects_new_primary_recipe_after_blocked_attempt(tmp_path: Path):
    blocked_recipe = write_finetune_recipe(tmp_path, include_label=False)
    blocked_dir = tmp_path / "plans" / "blocked"
    assert plans.build_plan(recipe_path=blocked_recipe, output_dir=blocked_dir).exit_code == 2

    recipe_payload = yaml.safe_load(blocked_recipe.read_text())
    recipe_payload["inputs"]["label_name"] = "ahi"
    successful_recipe = write_yaml(tmp_path / "successful-recipe.yaml", recipe_payload)
    successful_dir = tmp_path / "plans" / "successful"

    with pytest.raises(ValueError, match="primary recipe cannot change"):
        plans.build_plan(recipe_path=successful_recipe, output_dir=successful_dir)

    assert not successful_dir.exists()
    step = yaml.safe_load((tmp_path / "steps" / "unit-finetune" / "step.yaml").read_text())
    assert step["recipe_path"] == str(blocked_recipe)
    assert step["plans"] == [str(blocked_dir)]


def test_init_plan_and_mutation_share_canonical_absolute_root(tmp_path: Path):
    source = tmp_path / "source"
    recipe = _hparam_recipe(source)
    payload = yaml.safe_load(recipe.read_text())
    alias_root = tmp_path / "anchor" / ".." / "workspace"
    (tmp_path / "anchor").mkdir()
    payload["experiment"]["root"] = str(alias_root)
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    spec = tmp_path / "experiment-spec.yaml"
    spec.write_text(yaml.safe_dump({"experiment": payload["experiment"]}, sort_keys=False))
    canonical_root = alias_root.resolve()
    plan_dir = canonical_root / "plans" / "canonical"

    experiments.init_experiment(alias_root, spec)
    planned = run_cli("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir))
    assert planned.returncode == 0, planned.stderr
    monitored = experiments.monitor_experiment(alias_root)

    assert monitored["run_dir"] == str(canonical_root)
    assert yaml.safe_load((canonical_root / "experiment.yaml").read_text())["experiment"]["root"] == str(canonical_root)
    with (canonical_root / "experiment_manifest.tsv").open(newline="") as file_obj:
        experiment_rows = list(csv.DictReader(file_obj, delimiter="\t"))
    assert experiment_rows[0]["experiment_root"] == str(canonical_root)
    assert yaml.safe_load((plan_dir / "recipe.resolved.yaml").read_text())["experiment"]["root"] == str(canonical_root)


def test_relative_single_run_plan_persists_absolute_management_paths(tmp_path: Path, monkeypatch):
    source = tmp_path / "source"
    recipe = write_finetune_recipe(source)
    workspace = tmp_path / "workspace"
    payload = yaml.safe_load(recipe.read_text())
    payload["experiment"]["root"] = str(workspace)
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    monkeypatch.chdir(tmp_path)

    report = plans.build_plan(recipe_path=recipe, output_dir=Path("workspace/plan"))

    assert report.exit_code == 0
    plan_dir = workspace / "plan"
    run = json.loads((plan_dir / "plan.json").read_text())["runs"][0]
    for field in ("run_dir", "config", "script", "artifacts", "runtime_dir", "checkpoint_dir"):
        assert Path(run[field]).is_absolute()
    step = yaml.safe_load((workspace / "steps" / payload["step"]["id"] / "step.yaml").read_text())
    assert step["plans"] == [str(plan_dir)]
    assert Path(step["recipe_path"]).is_absolute()
    events = [json.loads(line) for line in (workspace / "events.jsonl").read_text().splitlines()]
    created = next(event for event in events if event["event_type"] == "plan_created")
    assert created["plan_dir"] == str(plan_dir)


def test_planning_recipe_source_pointer_is_absolute():
    source = Path(__file__).parents[2] / "recipes" / "examples" / "tiny_fixture_finetune.yaml"

    recipe, _, report = plans.evaluate_recipe(source.relative_to(Path(__file__).parents[2]))

    assert report.exit_code == 0
    assert recipe["_recipe_path"] == str(source.resolve())


@pytest.mark.parametrize("mutation", ["missing", "extra", "drift", "legacy"])
def test_hparam_plan_rejects_workspace_parameter_contract_drift(tmp_path: Path, mutation: str):
    recipe = _hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    result = run_cli("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir))
    assert result.returncode == 0, result.stderr
    manifest_path = tmp_path / "run_manifest.tsv"
    with manifest_path.open(newline="") as file_obj:
        rows = list(csv.DictReader(file_obj, delimiter="\t"))
    if mutation == "missing":
        rows[0].pop("runtime.lr")
    elif mutation == "extra":
        rows[0]["runtime.batch_size"] = "64"
    elif mutation == "drift":
        rows[0]["runtime.lr"] = "9e-06"
    else:
        rows[0]["param.runtime.lr"] = rows[0].pop("runtime.lr")
    with manifest_path.open("w", newline="") as file_obj:
        fieldnames = sorted({field for row in rows for field in row})
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="parameters differ|runtime\\.lr|Historical parameter fields"):
        run_artifacts.read_hparam_plan(plan_dir)


def test_hparam_plan_ignores_blank_shared_manifest_parameter_padding(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    historical = dict(payload)
    historical["step"] = {**payload["step"], "id": "historical"}
    historical["search"] = {
        **payload["search"],
        "parameters": {"runtime.batch_size": [32]},
    }
    historical_recipe = tmp_path / "historical.yaml"
    historical_recipe.write_text(yaml.safe_dump(historical, sort_keys=False))
    current = dict(payload)
    current["step"] = {**payload["step"], "id": "current"}
    current_recipe = tmp_path / "current.yaml"
    current_recipe.write_text(yaml.safe_dump(current, sort_keys=False))
    historical_plan = tmp_path / "plans" / "historical"
    current_plan = tmp_path / "plans" / "current"

    first = run_cli("plan", "--recipe", str(historical_recipe), "--output-dir", str(historical_plan))
    second = run_cli("plan", "--recipe", str(current_recipe), "--output-dir", str(current_plan))

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    with (tmp_path / "run_manifest.tsv").open(newline="") as file_obj:
        rows = list(csv.DictReader(file_obj, delimiter="\t"))
    current_row = next(row for row in rows if row["step_id"] == "current")
    assert current_row["runtime.batch_size"] == ""
    assert run_artifacts.read_hparam_plan(current_plan)["runs"][0]["runtime.lr"] == 2e-6


def test_plan_blocks_missing_experiment_metadata(tmp_path: Path):
    recipe = write_finetune_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    payload.pop("experiment")
    payload.pop("step")
    recipe.write_text(yaml.safe_dump(payload))

    result = run_cli("plan", "--recipe", str(recipe), "--output-dir", str(tmp_path / "plan"))

    assert result.returncode == 2
    assert "experiment" in result.stdout
    assert not (tmp_path / "plan").exists()


def test_plan_rejects_output_outside_experiment_root(tmp_path: Path):
    recipe = write_finetune_recipe(tmp_path / "experiment")

    result = run_cli("plan", "--recipe", str(recipe), "--output-dir", str(tmp_path / "outside"))

    assert result.returncode == 1
    assert "Plan output must be inside experiment.root" in result.stdout


def test_plan_rejects_nonempty_unmanaged_experiment_root(tmp_path: Path):
    recipe = write_finetune_recipe(tmp_path / "source")
    payload = yaml.safe_load(recipe.read_text())
    unmanaged_root = tmp_path / "old-results"
    unmanaged_root.mkdir()
    (unmanaged_root / "old.log").write_text("historical output\n")
    payload["experiment"]["root"] = str(unmanaged_root)
    recipe.write_text(yaml.safe_dump(payload))

    result = run_cli("plan", "--recipe", str(recipe), "--output-dir", str(unmanaged_root / "plan"))

    assert result.returncode == 1
    assert "Experiment root is non-empty" in result.stdout
    assert not (unmanaged_root / "experiment.yaml").exists()


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("id", "another-experiment", "different experiment"),
        ("title", "Another title", "experiment.title differs"),
    ],
)
def test_workspace_validates_existing_experiment_before_creating_directories(
    tmp_path: Path, field: str, value: str, error: str
):
    recipe_path = write_finetune_recipe(tmp_path)
    recipe = yaml.safe_load(recipe_path.read_text())
    existing = dict(recipe["experiment"])
    existing[field] = value
    (tmp_path / "experiment.yaml").write_text(yaml.safe_dump({"experiment": existing}, sort_keys=False))

    with pytest.raises(ValueError, match=error):
        ensure_experiment_workspace(recipe, tmp_path / "steps" / "unit-finetune" / "plan")

    assert not (tmp_path / "reports").exists()
    assert not (tmp_path / "steps").exists()


def test_launch_rejects_modified_frozen_script(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    assert run_cli("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    run = json.loads((plan_dir / "plan.json").read_text())["runs"][0]
    Path(run["script"]).write_text("#!/usr/bin/env bash\nexit 0\n")

    result = run_cli("hparam-launch", "--plan-dir", str(plan_dir))

    assert result.returncode == 1
    assert "hash" in result.stderr.lower()


def test_stop_requires_and_records_reason(tmp_path: Path, monkeypatch):
    recipe = _hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    assert run_cli("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    hparam_runtime.launch_hparam_runs(plan_dir, dry_run=False, hooks=FakeLauncher(verify_target=False).hooks())
    row = list(csv.DictReader((plan_dir / "launch_manifest.tsv").open(), delimiter="\t"))[0]
    pid_path = Path(row["pid_path"])
    identity = {"pid": 123, "process_group_id": 123, "process_start_token": "proc:unit-start"}
    pid_path.write_text(json.dumps(identity) + "\n")
    merge_run_manifest(
        tmp_path,
        [{"step_id": row["step_id"], "run_id": row["run_id"], **identity}],
    )
    monkeypatch.setattr(hparam_runtime.evidence, "stop_process_group", lambda *_args: None)

    with pytest.raises(ValueError, match="reason"):
        hparam_runtime.stop_hparam_run(plan_dir, "run-000", reason="")
    status_path = hparam_runtime.stop_hparam_run(plan_dir, "run-000", reason="validation diverged")

    assert "validation diverged" in status_path.read_text()
    assert "validation diverged" in (tmp_path / "events.jsonl").read_text()
    stopped = next(
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text().splitlines()
        if json.loads(line)["event_type"] == "run_stopped"
    )
    assert stopped["step_id"] == "unit-hparam-tune"
    assert stopped["run_id"] == "run-000"


def test_run_ids_are_scoped_by_step_in_experiment_manifest(tmp_path: Path):
    (tmp_path / "experiment.yaml").write_text("experiment:\n  id: unit\n")
    initialize_run_manifest(tmp_path)
    merge_run_manifest(
        tmp_path,
        [{"experiment_id": "unit", "step_id": "prepare-data", "run_id": "run-000", "status": "finished"}],
    )
    merge_run_manifest(
        tmp_path,
        [{"experiment_id": "unit", "step_id": "train-model", "run_id": "run-000", "status": "planned"}],
    )

    with (tmp_path / "run_manifest.tsv").open(newline="") as file_obj:
        rows = list(csv.DictReader(file_obj, delimiter="\t"))
    assert [(row["step_id"], row["run_id"]) for row in rows] == [
        ("prepare-data", "run-000"),
        ("train-model", "run-000"),
    ]


def test_merge_run_manifest_rejects_new_run_owned_by_a_different_experiment(tmp_path: Path):
    (tmp_path / "experiment.yaml").write_text(
        yaml.safe_dump(
            {
                "experiment": {
                    "id": "unit",
                    "title": "Unit",
                    "objective": "Test canonical ownership.",
                    "root": str(tmp_path),
                    "baseline": {"type": "none"},
                }
            },
            sort_keys=False,
        )
    )
    initialize_run_manifest(tmp_path)
    before = (tmp_path / "run_manifest.tsv").read_bytes()

    with pytest.raises(ValueError, match="different experiment"):
        merge_run_manifest(
            tmp_path,
            [{"experiment_id": "foreign", "step_id": "train", "run_id": "run-000"}],
        )

    assert (tmp_path / "run_manifest.tsv").read_bytes() == before


@pytest.mark.parametrize(
    ("row", "expected"),
    [
        ({"step_id": "step-a", "run_id": "run-007", "version": "shared"}, ("step-a", "run-007")),
        ({"step_id": "step-b", "run_id": "run-007", "version": "shared"}, ("step-b", "run-007")),
        ({"run_id": "run-007", "version": "shared"}, None),
        ({"version": "shared"}, None),
    ],
)
def test_managed_run_key_uses_step_and_run_identity(row: dict, expected: tuple[str, str] | None):
    assert managed_run_key(row) == expected


@pytest.mark.parametrize(
    "row",
    [
        {"step_id": "step-a", "run_id": "run-007"},
        {"step_id": 12, "run_id": 34},
    ],
)
def test_validated_run_key_matches_identity_owner(row):
    assert validated_run_key(row) == managed_run_key(row)


@pytest.mark.parametrize(
    "row",
    [
        {},
        {"step_id": "step-a"},
        {"run_id": "run-007"},
        {"step_id": "", "run_id": "run-007"},
        {"step_id": "step-a", "run_id": " "},
        {"step_id": None, "run_id": "run-007"},
    ],
)
def test_validated_run_key_requires_identity(row):
    assert managed_run_key(row) is None
    with pytest.raises(ValueError, match="Validated managed row has no run identity"):
        validated_run_key(row)


def test_resolve_run_row_prefers_managed_identity_over_duplicate_version():
    rows = [
        {"step_id": "step-a", "run_id": "run-000", "version": "shared", "marker": "a"},
        {"step_id": "step-b", "run_id": "run-000", "version": "shared", "marker": "b"},
    ]

    matched = resolve_run_row(rows, {"step_id": "step-b", "run_id": "run-000", "version": "shared"})

    assert matched == rows[1]


def test_resolve_run_row_does_not_fallback_when_complete_managed_identity_is_unmatched():
    rows = [{"step_id": "step-a", "run_id": "run-000", "version": "shared"}]

    assert resolve_run_row(rows, {"step_id": "step-b", "run_id": "run-000", "version": "shared"}) is None


def test_resolve_run_row_falls_back_to_unique_version():
    rows = [
        {"step_id": "step-a", "run_id": "run-000", "version": "unique"},
        {"step_id": "step-b", "run_id": "run-000", "version": "other"},
    ]

    assert resolve_run_row(rows, {"version": "unique"}) == rows[0]


def test_resolve_run_row_does_not_fall_back_to_run_id_when_version_is_unmatched():
    rows = [{"step_id": "train", "run_id": "run-000", "version": "current"}]

    assert resolve_run_row(rows, {"run_id": "run-000", "version": "stale"}) is None


def test_resolve_run_row_does_not_match_unscoped_run_id():
    rows = [{"step_id": "train", "run_id": "run-000", "version": "current"}]

    assert resolve_run_row(rows, {"run_id": "run-000"}) is None


def test_resolve_run_row_rejects_ambiguous_version():
    rows = [
        {"step_id": "step-a", "run_id": "run-000", "version": "shared"},
        {"step_id": "step-b", "run_id": "run-000", "version": "shared"},
    ]

    with pytest.raises(ValueError, match="Ambiguous runtime version"):
        resolve_run_row(rows, {"version": "shared"})


def test_managed_rows_reject_historical_identity():
    with pytest.raises(ValueError, match="read-only"):
        validate_managed_run_rows(
            [{"step_id": "train", "trial_id": "trial_000", "status": "running"}],
            source="legacy.tsv",
        )


def test_managed_rows_reject_duplicate_identity():
    row = {"step_id": "train", "run_id": "run-000"}

    with pytest.raises(ValueError, match="Duplicate managed run identity"):
        validate_managed_run_rows([row, dict(row)], source="run_manifest.tsv")


def test_managed_run_parameters_reject_legacy_prefix():
    assert managed_run_parameters(
        {
            "step_id": "train",
            "run_id": "run-000",
            "runtime.lr": 2e-6,
            "yaml:/model/router_frozen": True,
        }
    ) == {"runtime.lr": 2e-6, "yaml:/model/router_frozen": True}

    with pytest.raises(ValueError, match="Historical parameter fields"):
        managed_run_parameters({"param.runtime.lr": 2e-6})


@pytest.mark.parametrize(
    ("existing_status", "incoming", "expected_status"),
    [
        ("planned", {"status": "running"}, "running"),
        ("pending", {"status": "launched"}, "launched"),
        ("launched", {"status": "planned"}, "launched"),
        ("launched", {"status": "pending"}, "launched"),
        ("running", {"status": "planned"}, "running"),
        ("running", {"status": "pending"}, "running"),
        ("stopping", {"status": "planned"}, "stopping"),
        ("stopping", {"status": "pending"}, "stopping"),
        ("lost", {"status": "planned"}, "lost"),
        ("lost", {"status": "pending"}, "lost"),
        ("lost", {"status": "completed"}, "completed"),
        ("lost", {"status": "failed"}, "failed"),
        ("missing_pid", {"status": "pending"}, "missing_pid"),
        ("running", {"status": "failed"}, "failed"),
        ("planned", {"status": "superseded"}, "superseded"),
        ("pending", {"status": "superseded"}, "superseded"),
        ("launched", {"status": "superseded"}, "launched"),
        ("running", {"status": "superseded"}, "running"),
        ("completed", {"status": "failed"}, "failed"),
        ("finished", {"status": "failed"}, "failed"),
        ("completed", {"status": "running"}, "completed"),
        ("finished", {"status": "running"}, "finished"),
        ("failed", {"status": "running"}, "failed"),
        ("stopped", {"status": "running"}, "stopped"),
        ("launch_failed", {"status": "running"}, "launch_failed"),
        ("superseded", {"status": "running"}, "superseded"),
        ("running", {"score": 0.8}, "running"),
        ("running", {"status": ""}, ""),
        ("running", {"status": None}, None),
    ],
)
def test_merge_run_row_preserves_status_precedence(existing_status: str, incoming: dict, expected_status: str | None):
    existing = {"step_id": "train", "run_id": "run-000", "status": existing_status}

    merged = merge_run_row(existing, incoming)

    assert merged["status"] == expected_status


@pytest.mark.parametrize("incoming_status", ["queued", "running", "lost", "completed", "failed"])
@pytest.mark.parametrize(
    "incoming_stop_fields",
    [
        {},
        {"stop_requested_at": "", "stop_reason": "stale reason"},
        {"stop_requested_at": "2026-08-21T03:39:00Z", "stop_reason": "stale reason"},
    ],
)
def test_merge_run_row_preserves_stop_intent_against_stale_observations(
    incoming_status: str,
    incoming_stop_fields: dict,
):
    existing = {
        "step_id": "train",
        "run_id": "run-000",
        "status": "stopping",
        "stop_requested_at": "2026-08-21T03:40:00Z",
        "stop_reason": "validation diverged",
    }
    incoming = {
        "step_id": "train",
        "run_id": "run-000",
        "status": incoming_status,
        "scheduler_raw_state": "RUNNING",
        **incoming_stop_fields,
    }

    merged = merge_run_row(existing, incoming)

    assert merged["status"] == "stopping"
    assert merged["stop_requested_at"] == "2026-08-21T03:40:00Z"
    assert merged["stop_reason"] == "validation diverged"
    assert merged["scheduler_raw_state"] == "RUNNING"


@pytest.mark.parametrize("incoming_status", ["stopping", "stopped", "completed", "failed", "lost"])
def test_merge_run_row_accepts_observation_bound_to_current_stop_intent(incoming_status: str):
    existing = {
        "step_id": "train",
        "run_id": "run-000",
        "status": "stopping",
        "stop_requested_at": "2026-08-21T03:40:00Z",
        "stop_reason": "validation diverged",
    }

    merged = merge_run_row(
        existing,
        {
            "step_id": "train",
            "run_id": "run-000",
            "status": incoming_status,
            "stop_requested_at": "2026-08-21T03:40:00Z",
            "stop_reason": "validation diverged",
        },
    )

    assert merged["status"] == incoming_status
    assert merged["stop_requested_at"] == "2026-08-21T03:40:00Z"
    assert merged["stop_reason"] == "validation diverged"


@pytest.mark.parametrize("incoming_status", ["planned", "pending", "running"])
@pytest.mark.parametrize("stop_requested_at", ["", "2026-08-30T03:40:00Z"])
def test_merge_run_row_preserves_stopped_metadata_against_stale_observations(
    incoming_status: str, stop_requested_at: str
):
    existing = {
        "step_id": "train",
        "run_id": "run-000",
        "status": "stopped",
        "stop_requested_at": stop_requested_at,
        "stop_reason": "budget withdrawn",
        "stopped_at": "2026-08-30T03:41:00Z",
    }
    stale = {
        "status": incoming_status,
        "stop_requested_at": "",
        "stop_reason": "",
        "stopped_at": "",
    }

    merged = merge_run_row(existing, stale)

    assert merged == existing
    assert merge_run_row(merged, stale) == existing


def test_merge_run_row_is_idempotent():
    existing = {"step_id": "train", "run_id": "run-000", "status": "completed"}
    incoming = {"status": "running", "score": 0.8}

    once = merge_run_row(existing, incoming)

    assert merge_run_row(once, incoming) == once


def test_merge_run_manifest_allows_omitted_frozen_fields_and_fills_missing_values(tmp_path: Path):
    (tmp_path / "experiment.yaml").write_text("experiment:\n  id: unit\n")
    initialize_run_manifest(tmp_path)
    identity = {"experiment_id": "unit", "step_id": "train", "run_id": "run-000"}
    merge_run_manifest(tmp_path, [{**identity, "version": "v1", "status": "planned"}])

    merge_run_manifest(tmp_path, [{**identity, "script_sha256": "abc", "status": "running"}])
    rows = list(csv.DictReader((tmp_path / "run_manifest.tsv").open(), delimiter="\t"))

    assert rows[0]["version"] == "v1"
    assert rows[0]["script_sha256"] == "abc"
    assert rows[0]["status"] == "running"


@pytest.mark.parametrize("incoming_status", ["finished", "stopped"])
def test_merge_run_manifest_returns_the_canonical_rows_it_committed(tmp_path: Path, incoming_status: str):
    (tmp_path / "experiment.yaml").write_text("experiment:\n  id: unit\n")
    initialize_run_manifest(tmp_path)
    identity = {"experiment_id": "unit", "step_id": "train", "run_id": "run-000"}
    merge_run_manifest(tmp_path, [{**identity, "status": "failed"}])

    committed = merge_run_manifest(tmp_path, [{**identity, "status": incoming_status}])
    rows = list(csv.DictReader((tmp_path / "run_manifest.tsv").open(), delimiter="\t"))

    assert committed == rows
    assert committed[0]["status"] == "failed"


def test_append_event_leaves_original_log_intact_when_temporary_write_fails(tmp_path: Path, monkeypatch):
    events_path = tmp_path / "events.jsonl"
    events_path.write_text('{"event_type": "before"}\n')
    before = events_path.read_bytes()
    original_fsync = experiment_io.os.fsync

    def fail_fsync(_descriptor):
        raise OSError("injected temporary write failure")

    monkeypatch.setattr(experiment_io.os, "fsync", fail_fsync)
    with pytest.raises(OSError, match="injected temporary write failure"):
        append_event(tmp_path, "run_status_changed", {"run_id": "run-000"})

    assert events_path.read_bytes() == before
    monkeypatch.setattr(experiment_io.os, "fsync", original_fsync)
    append_event(tmp_path, "run_status_changed", {"run_id": "run-000"})
    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    assert [event["event_type"] for event in events] == ["before", "run_status_changed"]


def test_concurrent_append_events_share_the_sidecar_lock(tmp_path: Path, monkeypatch):
    original_lock = experiment_io.blocking_file_lock
    first_locked = threading.Event()
    second_attempted = threading.Event()
    release_first = threading.Event()
    state_lock = threading.Lock()
    attempts = 0

    @contextmanager
    def observe_lock(path):
        nonlocal attempts
        with state_lock:
            attempts += 1
            if attempts == 2:
                second_attempted.set()
        with original_lock(path):
            first_holder = not first_locked.is_set()
            if first_holder:
                first_locked.set()
                assert release_first.wait(timeout=10)
            yield

    monkeypatch.setattr(experiment_io, "blocking_file_lock", observe_lock)
    errors = []

    def write_event(event_type):
        try:
            append_event(tmp_path, event_type, {})
        except BaseException as exc:
            errors.append(exc)

    first = threading.Thread(target=write_event, args=("first",))
    second = threading.Thread(target=write_event, args=("second",))
    first.start()
    assert first_locked.wait(timeout=10)
    second.start()
    assert second_attempted.wait(timeout=10)
    release_first.set()
    first.join(timeout=20)
    second.join(timeout=20)

    assert not first.is_alive()
    assert not second.is_alive()
    assert errors == []
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert [event["event_type"] for event in events] == ["first", "second"]


def test_append_event_has_no_failing_tail_after_commit(tmp_path: Path, monkeypatch):
    (tmp_path / "events.jsonl").write_text('{"event_type": "before"}\n')
    original_replace = experiment_io.os.replace
    original_close = experiment_io.os.close
    committed = False

    def mark_commit(*args, **kwargs):
        nonlocal committed
        result = original_replace(*args, **kwargs)
        committed = True
        return result

    def fail_close_after_commit(descriptor):
        if committed:
            raise OSError("injected post-commit close failure")
        return original_close(descriptor)

    monkeypatch.setattr(experiment_io.os, "replace", mark_commit)
    monkeypatch.setattr(experiment_io.os, "close", fail_close_after_commit)

    append_event(tmp_path, "committed", {})

    monkeypatch.setattr(experiment_io.os, "close", original_close)
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert [event["event_type"] for event in events] == ["before", "committed"]


@pytest.mark.parametrize(
    "text, message",
    [
        ("", "empty"),
        ("  \n", "empty"),
        ("run_id\n", "step_id"),
        ("step_id\trun_id\ttrial_id\n", "trial_id"),
        ("step_id\trun_id\ntrain\trun-000\n", "experiment_id"),
        (
            "experiment_id\tstep_id\trun_id\nunit\ttrain\trun-000\nunit\ttrain\trun-000\n",
            "Duplicate",
        ),
        (
            "experiment_id\tstep_id\trun_id\tconfig\nunit\ttrain\trun-000\trelative/config.yaml\n",
            "non-absolute",
        ),
        ("experiment_id\tstep_id\trun_id\n \t \t \n", "experiment_id"),
    ],
)
def test_read_run_manifest_rejects_corrupt_canonical_tables(tmp_path: Path, text: str, message: str):
    (tmp_path / "run_manifest.tsv").write_text(text)

    with pytest.raises(ValueError, match=message):
        read_run_manifest(tmp_path)


def test_read_run_manifest_distinguishes_missing_from_valid_header_only(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="run_manifest.tsv"):
        read_run_manifest(tmp_path)

    initialize_run_manifest(tmp_path)

    assert (tmp_path / "run_manifest.tsv").read_text() == "step_id\trun_id\n"
    assert read_run_manifest(tmp_path) == []


@pytest.mark.parametrize(
    ("phase", "controller", "match"),
    [
        ("invalid", "ordinary", "step.phase"),
        ("train", "unassigned", "invalid plan_controller: unassigned"),
        ("train", "pipeline", "invalid plan_controller: pipeline"),
        ("train", "adaptive", "invalid plan_controller: adaptive"),
    ],
)
def test_read_step_manifest_rejects_invalid_phase_or_controller(
    tmp_path: Path, phase: str, controller: str, match: str
):
    path = tmp_path / "steps" / "train" / "step.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(
        "step:\n"
        "  id: train\n"
        f"  phase: {phase}\n"
        "  purpose: Train the model.\n"
        "experiment_id: unit\n"
        f"plan_controller: {controller}\n"
        "recipe_path: ''\n"
        "plans: []\n"
    )

    with pytest.raises(ValueError, match=match):
        read_step_manifest(tmp_path, "train")


def test_managed_yaml_reader_rejects_recursive_alias_without_hanging():
    code = (
        "from agent_tools.experiment_workspace import read_managed_yaml_mapping; "
        "read_managed_yaml_mapping('experiment: &recursive [*recursive]\\n', source='experiment.yaml')"
    )

    result = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True, timeout=2)

    assert result.returncode != 0
    assert "recursive YAML alias" in result.stderr


def test_merge_run_manifest_never_recreates_missing_canonical_table(tmp_path: Path):
    (tmp_path / "experiment.yaml").write_text("experiment:\n  id: unit\n")

    with pytest.raises(FileNotFoundError, match="run_manifest.tsv"):
        merge_run_manifest(tmp_path, [{"step_id": "train", "run_id": "run-000", "status": "planned"}])

    assert not (tmp_path / "run_manifest.tsv").exists()


def test_empty_canonical_commit_preserves_the_valid_identity_header(tmp_path: Path):
    (tmp_path / "experiment.yaml").write_text("experiment:\n  id: unit\n")
    initialize_run_manifest(tmp_path)

    assert merge_run_manifest(tmp_path, []) == []

    assert (tmp_path / "run_manifest.tsv").read_text() == "step_id\trun_id\n"
    assert read_run_manifest(tmp_path) == []


def test_concurrent_local_manifest_writers_preserve_distinct_runs(tmp_path: Path):
    (tmp_path / "experiment.yaml").write_text("experiment:\n  id: unit\n")
    initialize_run_manifest(tmp_path)
    code = (
        "import sys; "
        "from agent_tools.experiment_workspace import merge_run_manifest; "
        "merge_run_manifest(sys.argv[1], [{'experiment_id': 'unit', 'step_id': 'train', "
        "'run_id': sys.argv[2], 'status': 'planned'}])"
    )
    processes = [
        subprocess.Popen([sys.executable, "-c", code, str(tmp_path), run_id], text=True)
        for run_id in ("run-000", "run-001")
    ]

    assert [process.wait(timeout=SUBPROCESS_WAIT_SECONDS) for process in processes] == [0, 0]
    assert {row["run_id"] for row in read_run_manifest(tmp_path)} == {"run-000", "run-001"}


def test_concurrent_terminal_updates_use_the_existing_reducer(tmp_path: Path):
    (tmp_path / "experiment.yaml").write_text("experiment:\n  id: unit\n")
    initialize_run_manifest(tmp_path)
    merge_run_manifest(
        tmp_path,
        [{"experiment_id": "unit", "step_id": "train", "run_id": "run-000", "status": "running"}],
    )
    code = (
        "import sys; "
        "from agent_tools.experiment_workspace import merge_run_manifest; "
        "merge_run_manifest(sys.argv[1], [{'step_id': 'train', 'run_id': 'run-000', 'status': sys.argv[2]}])"
    )
    processes = [
        subprocess.Popen([sys.executable, "-c", code, str(tmp_path), status], text=True)
        for status in ("completed", "failed")
    ]

    assert [process.wait(timeout=SUBPROCESS_WAIT_SECONDS) for process in processes] == [0, 0]
    assert read_run_manifest(tmp_path)[0]["status"] == "failed"


def test_interrupted_atomic_replace_preserves_the_complete_old_manifest(tmp_path: Path, monkeypatch):
    (tmp_path / "experiment.yaml").write_text("experiment:\n  id: unit\n")
    initialize_run_manifest(tmp_path)
    merge_run_manifest(
        tmp_path,
        [{"experiment_id": "unit", "step_id": "train", "run_id": "run-000", "status": "planned"}],
    )
    before = (tmp_path / "run_manifest.tsv").read_bytes()
    monkeypatch.setattr(
        experiment_io.os,
        "replace",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("interrupted")),
    )

    with pytest.raises(OSError, match="interrupted"):
        merge_run_manifest(tmp_path, [{"step_id": "train", "run_id": "run-000", "status": "running"}])

    assert (tmp_path / "run_manifest.tsv").read_bytes() == before
    assert read_run_manifest(tmp_path)[0]["status"] == "planned"


def test_only_experiment_workspace_reads_or_writes_the_canonical_run_manifest():
    agent_tools_dir = Path(__file__).parents[2] / "agent_tools"
    offenders = []
    generic_io_names = {
        "open",
        "read_bytes",
        "read_rows",
        "read_rows_at",
        "read_text",
        "read_text_at",
        "write_bytes",
        "write_rows",
        "write_rows_at",
        "write_text",
        "write_text_at",
        "_read_rows",
        "_write_rows",
    }
    for path in agent_tools_dir.glob("*.py"):
        if path.name == "experiment_workspace.py":
            continue
        tree = ast.parse(path.read_text())
        for function in (node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)):
            locator_names = set()
            for node in ast.walk(function):
                if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                    continue
                value = node.value
                if value is None:
                    continue
                if not any(
                    isinstance(part, ast.Constant) and part.value == "run_manifest.tsv" for part in ast.walk(value)
                ):
                    continue
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                locator_names.update(
                    part.id for target in targets for part in ast.walk(target) if isinstance(part, ast.Name)
                )
            for node in ast.walk(function):
                if not isinstance(node, ast.Call):
                    continue
                function_name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
                if function_name not in generic_io_names:
                    continue
                direct = any(
                    isinstance(part, ast.Constant) and part.value == "run_manifest.tsv" for part in ast.walk(node)
                )
                indirect = any(isinstance(part, ast.Name) and part.id in locator_names for part in ast.walk(node))
                if direct or indirect:
                    offenders.append(f"{path.name}:{function.name}:{node.lineno}")
    assert offenders == []


def test_step_manifest_writes_merge_under_the_step_file_lock():
    agent_tools_dir = Path(__file__).parents[2] / "agent_tools"

    def function_calls(filename: str, function_name: str) -> set[str]:
        tree = ast.parse((agent_tools_dir / filename).read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == function_name)
        return {
            node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
        }

    assert {"blocking_file_lock", "merge_step_manifest", "atomic_replace"} <= function_calls(
        "experiment_workspace.py", "commit_step_manifest"
    )
    assert "commit_step_manifest" in function_calls("experiment_workspace.py", "ensure_experiment_workspace")


def test_canonical_local_experiment_root_resolves_aliases(tmp_path: Path):
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    alias_parent = tmp_path / "alias"
    alias_parent.symlink_to(real_parent, target_is_directory=True)

    root = canonical_local_experiment_root(alias_parent / "nested" / ".." / "workspace", tmp_path)

    assert root == (real_parent / "workspace").resolve()


@pytest.mark.parametrize("dangling", [False, True])
def test_canonical_local_experiment_root_rejects_root_symlink(tmp_path: Path, dangling: bool):
    target = tmp_path / "target"
    if not dangling:
        target.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(target, target_is_directory=True)

    with pytest.raises(ValueError, match="must not be a symlink"):
        canonical_local_experiment_root(alias, tmp_path)


@pytest.mark.parametrize("dangling", [False, True])
def test_canonical_local_experiment_root_rejects_root_symlink_after_dot_normalization(tmp_path: Path, dangling: bool):
    target = tmp_path / "target"
    if not dangling:
        target.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(target, target_is_directory=True)

    with pytest.raises(ValueError, match="must not be a symlink"):
        canonical_local_experiment_root(tmp_path / "missing" / ".." / "alias", tmp_path)


def test_experiment_init_rejects_local_symlink_root_before_writing(tmp_path: Path):
    target = tmp_path / "target"
    target.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(target, target_is_directory=True)
    spec = tmp_path / "experiment-spec.yaml"
    spec.write_text(
        yaml.safe_dump(
            {
                "experiment": {
                    "id": "unit",
                    "title": "Unit experiment",
                    "objective": "Exercise experiment workspace contracts.",
                    "root": str(alias),
                    "baseline": {"type": "none", "rationale": "Unit fixture."},
                }
            }
        )
    )

    with pytest.raises(ValueError, match="must not be a symlink"):
        experiments.init_experiment(alias, spec)

    assert list(target.iterdir()) == []


@pytest.mark.parametrize(
    "field",
    [
        "experiment_id",
        "run_name",
        "parameter_summary",
        "version",
        "config",
        "config_sha256",
        "script",
        "script_sha256",
        "run_dir",
        "artifacts",
        "runtime_dir",
        "checkpoint_dir",
        "target",
        "workdir",
        "gpus",
        "pid_path",
        "pid",
        "process_group_id",
        "process_start_token",
        "log_path",
        "command",
        "runtime.lr",
    ],
)
@pytest.mark.parametrize("incoming_value", ["changed", ""])
def test_merge_run_manifest_rejects_frozen_field_changes_before_writing(
    tmp_path: Path, field: str, incoming_value: str
):
    initialize_run_manifest(tmp_path)
    identity = {"experiment_id": "unit", "step_id": "train", "run_id": "run-000"}
    original = str(tmp_path / "original") if field in MANAGED_RUN_PATH_FIELDS else "original"
    changed = str(tmp_path / "changed") if field in MANAGED_RUN_PATH_FIELDS and incoming_value else incoming_value
    workspace_experiment_id = original if field == "experiment_id" else identity["experiment_id"]
    (tmp_path / "experiment.yaml").write_text(f"experiment:\n  id: {workspace_experiment_id}\n")
    status = "launched" if field in PROCESS_IDENTITY_FIELDS else "planned"
    initial = {**identity, field: original, "status": status}
    if field in EXECUTION_IDENTITY_FIELDS and field != "target":
        initial["target"] = "local"
    merge_run_manifest(tmp_path, [initial])
    before = (tmp_path / "run_manifest.tsv").read_bytes()

    with pytest.raises(ValueError, match=field.replace(".", r"\.")):
        merge_run_manifest(tmp_path, [{**identity, field: changed, "status": "running"}])

    assert (tmp_path / "run_manifest.tsv").read_bytes() == before


def test_merge_run_manifest_rejects_input_snapshot_changes_before_writing(tmp_path: Path):
    (tmp_path / "experiment.yaml").write_text("experiment:\n  id: unit\n")
    initialize_run_manifest(tmp_path)
    identity = {"experiment_id": "unit", "step_id": "train", "run_id": "run-000"}
    snapshots = [{"field": "inputs.ckpt_path", "path": "/tmp/model.ckpt", "sha256": "a" * 64}]
    merge_run_manifest(tmp_path, [{**identity, "input_snapshots": snapshots, "status": "planned"}])
    before = (tmp_path / "run_manifest.tsv").read_bytes()

    changed = [{**snapshots[0], "sha256": "b" * 64}]
    with pytest.raises(ValueError, match="input_snapshots"):
        merge_run_manifest(tmp_path, [{**identity, "input_snapshots": changed, "status": "running"}])

    assert (tmp_path / "run_manifest.tsv").read_bytes() == before


def test_frozen_validator_only_allows_trusted_execution_identity_initialization():
    existing = {"step_id": "train", "run_id": "run-000", "status": "planned"}
    incoming = {"step_id": "train", "run_id": "run-000", "target": "local", "workdir": "/foreign/workdir"}

    with pytest.raises(ValueError, match="execution identity"):
        validate_frozen_run_update(existing, incoming)

    validate_frozen_run_update(existing, incoming, allow_execution_identity_fill=True)


def test_frozen_validator_allows_only_one_trusted_process_identity_fill():
    existing = {"step_id": "train", "run_id": "run-000", "target": "local", "pid": "", "status": "launched"}

    validate_frozen_run_update(existing, {"pid": 123}, allow_execution_identity_fill=True)

    with pytest.raises(ValueError, match="pid"):
        validate_frozen_run_update({**existing, "pid": 123}, {"pid": 456}, allow_execution_identity_fill=True)


def test_commit_run_start_first_fills_different_planned_and_actual_runtime_commits(tmp_path: Path):
    (tmp_path / "experiment.yaml").write_text("experiment:\n  id: unit\n")
    initialize_run_manifest(tmp_path)
    merge_run_manifest(
        tmp_path,
        [
            {
                "experiment_id": "unit",
                "step_id": "train",
                "run_id": "run-000",
                "planned_runtime_commit": "a" * 40,
                "status": "planned",
            }
        ],
    )

    committed = experiment_workspace.commit_run_start(
        tmp_path,
        "train",
        "run-000",
        planned_runtime_commit="a" * 40,
        runtime_commit="b" * 40,
    )

    row = committed[0]
    assert row["status"] == "running"
    assert row["planned_runtime_commit"] == "a" * 40
    assert row["runtime_commit"] == "b" * 40
    assert read_run_manifest(tmp_path)[0] == row


def test_commit_run_start_accepts_monitor_winning_with_identical_provenance(tmp_path: Path):
    (tmp_path / "experiment.yaml").write_text("experiment:\n  id: unit\n")
    initialize_run_manifest(tmp_path)
    expected = {
        "experiment_id": "unit",
        "step_id": "train",
        "run_id": "run-000",
        "planned_runtime_commit": "a" * 40,
        "runtime_commit": "b" * 40,
        "status": "running",
    }
    merge_run_manifest(tmp_path, [expected])
    before = (tmp_path / "run_manifest.tsv").read_bytes()

    committed = experiment_workspace.commit_run_start(
        tmp_path,
        "train",
        "run-000",
        planned_runtime_commit="a" * 40,
        runtime_commit="b" * 40,
    )

    assert committed == [expected]
    assert (tmp_path / "run_manifest.tsv").read_bytes() == before


@pytest.mark.parametrize(
    ("planned_commit", "runtime_commit"),
    [
        pytest.param("c" * 40, "b" * 40, id="planned-mismatch"),
        pytest.param("a" * 40, "c" * 40, id="actual-mismatch"),
    ],
)
def test_commit_run_start_rejects_running_provenance_mismatch(tmp_path: Path, planned_commit: str, runtime_commit: str):
    (tmp_path / "experiment.yaml").write_text("experiment:\n  id: unit\n")
    initialize_run_manifest(tmp_path)
    merge_run_manifest(
        tmp_path,
        [
            {
                "experiment_id": "unit",
                "step_id": "train",
                "run_id": "run-000",
                "planned_runtime_commit": "a" * 40,
                "runtime_commit": "b" * 40,
                "status": "running",
            }
        ],
    )
    before = (tmp_path / "run_manifest.tsv").read_bytes()

    with pytest.raises(ValueError, match="running provenance differs"):
        experiment_workspace.commit_run_start(
            tmp_path,
            "train",
            "run-000",
            planned_runtime_commit=planned_commit,
            runtime_commit=runtime_commit,
        )

    assert (tmp_path / "run_manifest.tsv").read_bytes() == before


@pytest.mark.parametrize("field", ["planned_runtime_commit", "runtime_commit"])
def test_runtime_provenance_cannot_be_rewritten_after_run_start(tmp_path: Path, field: str):
    (tmp_path / "experiment.yaml").write_text("experiment:\n  id: unit\n")
    initialize_run_manifest(tmp_path)
    merge_run_manifest(
        tmp_path,
        [{"experiment_id": "unit", "step_id": "train", "run_id": "run-000", "status": "planned"}],
    )
    experiment_workspace.commit_run_start(
        tmp_path,
        "train",
        "run-000",
        planned_runtime_commit="a" * 40,
        runtime_commit="b" * 40,
    )
    before = (tmp_path / "run_manifest.tsv").read_bytes()

    with pytest.raises(ValueError, match=field):
        merge_run_manifest(
            tmp_path,
            [{"step_id": "train", "run_id": "run-000", field: "c" * 40, "status": "completed"}],
        )

    assert (tmp_path / "run_manifest.tsv").read_bytes() == before
    assert read_run_manifest(tmp_path)[0]["planned_runtime_commit"] == "a" * 40
    assert read_run_manifest(tmp_path)[0]["runtime_commit"] == "b" * 40


def test_stale_start_cannot_backfill_runtime_provenance_into_terminal_run(tmp_path: Path):
    (tmp_path / "experiment.yaml").write_text("experiment:\n  id: unit\n")
    initialize_run_manifest(tmp_path)
    merge_run_manifest(
        tmp_path,
        [
            {
                "experiment_id": "unit",
                "step_id": "train",
                "run_id": "run-000",
                "planned_runtime_commit": "a" * 40,
                "status": "completed",
            }
        ],
    )
    before = (tmp_path / "run_manifest.tsv").read_bytes()

    with pytest.raises(ValueError, match="cannot start from status completed"):
        experiment_workspace.commit_run_start(
            tmp_path,
            "train",
            "run-000",
            planned_runtime_commit="a" * 40,
            runtime_commit="b" * 40,
        )

    assert (tmp_path / "run_manifest.tsv").read_bytes() == before
    row = read_run_manifest(tmp_path)[0]
    assert row["planned_runtime_commit"] == "a" * 40
    assert row.get("runtime_commit", "") == ""


@pytest.mark.parametrize("field", ["planned_runtime_commit", "runtime_commit"])
@pytest.mark.parametrize("commit", ["a" * 39, "a" * 41, "a" * 64, "A" * 40, "g" * 40, 123])
def test_managed_rows_reject_malformed_runtime_provenance(field: str, commit: object):
    with pytest.raises(ValueError, match=rf"invalid {field}"):
        validate_managed_run_rows(
            [{"step_id": "train", "run_id": "run-000", field: commit}],
            source="unit manifest",
        )


def test_plan_registration_accepts_canonical_execution_identity_fill(tmp_path: Path):
    (tmp_path / "experiment.yaml").write_text("experiment:\n  id: unit\n")
    initialize_run_manifest(tmp_path)
    expected = {
        "experiment_id": "unit",
        "step_id": "train",
        "run_id": "run-000",
        "status": "planned",
    }
    merge_run_manifest(tmp_path, [expected])
    merge_run_manifest(
        tmp_path,
        [{"step_id": "train", "run_id": "run-000", "status": "launched", "target": "local", "gpus": "0"}],
    )

    assert experiment_workspace.plan_registration_rows_state(tmp_path, [expected], source="unit plan") == "present"


def test_plan_registration_reads_canonical_rows_only_under_run_lock(tmp_path: Path, monkeypatch):
    (tmp_path / "experiment.yaml").write_text("experiment:\n  id: unit\n")
    initialize_run_manifest(tmp_path)
    expected = {"experiment_id": "unit", "step_id": "train", "run_id": "run-000", "status": "planned"}
    merge_run_manifest(tmp_path, [expected])

    state = call_while_run_lock_holder_commits(
        monkeypatch,
        tmp_path,
        lambda: experiment_workspace.plan_registration_rows_state(tmp_path, [expected], source="unit plan"),
    )

    assert state == "present"


@pytest.mark.parametrize("reader", ["ensure_workspace", "next_run_index"])
def test_planning_reads_canonical_rows_only_under_run_lock(tmp_path: Path, monkeypatch, reader: str):
    root = tmp_path / "workspace"
    recipe = {
        "experiment": {
            "id": "unit",
            "title": "Unit experiment",
            "objective": "Exercise planning reads under the run lock.",
            "root": str(root),
            "baseline": {"type": "none"},
        },
        "step": {"id": "prepare", "phase": "prepare", "purpose": "Prepare a plan."},
    }
    plan_dir = root / "steps" / "prepare" / "plan"
    ensure_experiment_workspace(recipe, plan_dir)
    merge_run_manifest(
        root, [{"experiment_id": "unit", "step_id": "prepare", "run_id": "run-000", "status": "planned"}]
    )
    readers = {
        "ensure_workspace": lambda: ensure_experiment_workspace(recipe, plan_dir, validate_only=True),
        "next_run_index": lambda: experiment_workspace.next_run_index(recipe),
    }

    result = call_while_run_lock_holder_commits(monkeypatch, root, readers[reader])

    assert result == (1 if reader == "next_run_index" else (root, root / "steps" / "prepare"))


def _slurm_identity(tmp_path: Path) -> dict[str, str]:
    return {
        "scheduler_type": "slurm",
        "scheduler_script": str(tmp_path / "job.sbatch"),
        "scheduler_script_sha256": "a" * 64,
    }


@pytest.mark.parametrize("backend", ["direct", "slurm"])
@pytest.mark.parametrize("status", ["planned", "pending", "stopped"])
def test_managed_launch_evidence_ignores_status_and_slurm_plan_identity(tmp_path: Path, backend: str, status: str):
    row = {"scheduler_type": backend, "status": status}
    if backend == "slurm":
        row.update(**_slurm_identity(tmp_path), log_path=str(tmp_path / "slurm.log"))

    assert not experiment_workspace.has_managed_launch_evidence(row)


@pytest.mark.parametrize("backend", ["direct", "slurm"])
@pytest.mark.parametrize(
    "field",
    sorted(EXECUTION_IDENTITY_FIELDS | {"scheduler_job_id", "launched_at", "stop_requested_at"}),
)
def test_managed_launch_evidence_recognizes_each_runtime_binding(tmp_path: Path, backend: str, field: str):
    row = {"scheduler_type": backend, "status": "planned", field: "recorded"}
    if backend == "slurm":
        row.update(_slurm_identity(tmp_path))

    expected = not (backend == "slurm" and field == "log_path")
    assert experiment_workspace.has_managed_launch_evidence(row) is expected


@pytest.mark.parametrize("backend", ["direct", "slurm"])
@pytest.mark.parametrize("status", ["planned", "pending"])
def test_manifest_accepts_metadata_only_stop_without_launch_evidence(tmp_path: Path, backend: str, status: str):
    (tmp_path / "experiment.yaml").write_text("experiment:\n  id: unit\n")
    initialize_run_manifest(tmp_path)
    identity = {"experiment_id": "unit", "step_id": "train", "run_id": "run-000", "scheduler_type": backend}
    if backend == "slurm":
        identity.update(**_slurm_identity(tmp_path), log_path=str(tmp_path / "slurm.log"))
    merge_run_manifest(tmp_path, [{**identity, "status": status}])

    stopped = merge_run_manifest(
        tmp_path,
        [
            {
                "step_id": "train",
                "run_id": "run-000",
                "status": "stopped",
                "stop_reason": "budget withdrawn",
                "stopped_at": "2026-08-30T03:41:00Z",
            }
        ],
    )[0]

    assert stopped["status"] == "stopped"
    assert stopped["stop_reason"] == "budget withdrawn"
    assert stopped["stopped_at"] == "2026-08-30T03:41:00Z"
    for field in {"scheduler_job_id", "launched_at", "stop_requested_at"}:
        assert not stopped.get(field)
    for field, value in identity.items():
        assert stopped[field] == value
    assert read_run_manifest(tmp_path)[0] == stopped


@pytest.mark.parametrize("status", ["lost", "stopped"])
def test_scheduler_identity_allows_lost_and_stopped_runs_without_a_job(tmp_path: Path, status: str):
    # A job that aged out of squeue before its id was recorded is lost; a reasoned stop then resolves it.
    validate_scheduler_run_identity({"status": status, "target": "local", **_slurm_identity(tmp_path)})
    validate_scheduler_run_identity({"status": status, "scheduler_job_id": "3880", **_slurm_identity(tmp_path)})


@pytest.mark.parametrize("status", ["queued", "running", "stopping", "completed", "finished", "failed"])
def test_scheduler_identity_requires_job_for_other_active_and_terminal_states(tmp_path: Path, status: str):
    with pytest.raises(ValueError, match="requires scheduler_job_id"):
        validate_scheduler_run_identity({"status": status, **_slurm_identity(tmp_path)})


@pytest.mark.parametrize("status", ["planned", "pending", "launch_failed", "superseded"])
def test_scheduler_identity_rejects_job_on_unsubmitted_states(tmp_path: Path, status: str):
    with pytest.raises(ValueError, match="cannot define scheduler_job_id"):
        validate_scheduler_run_identity({"status": status, "scheduler_job_id": "3880", **_slurm_identity(tmp_path)})


def test_scheduler_identity_allows_one_trusted_job_binding(tmp_path: Path):
    existing = {
        "step_id": "train",
        "run_id": "run-000",
        "target": "local",
        "status": "planned",
        **_slurm_identity(tmp_path),
    }

    validate_frozen_run_update(existing, {"scheduler_job_id": "3880"}, allow_execution_identity_fill=True)
    assert SCHEDULER_BINDING_FIELDS == {"scheduler_job_id"}
    with pytest.raises(ValueError, match="scheduler_job_id"):
        validate_frozen_run_update(
            {**existing, "scheduler_job_id": "3880"},
            {"scheduler_job_id": "3881"},
            allow_execution_identity_fill=True,
        )


def test_scheduler_identity_is_backend_specific(tmp_path: Path):
    validate_scheduler_run_identity({"scheduler_type": "direct", "status": "planned"})
    validate_scheduler_run_identity({"status": "planned"})
    validate_scheduler_run_identity({"status": "planned", "target": "local", **_slurm_identity(tmp_path)})
    validate_scheduler_run_identity({"status": "queued", "scheduler_job_id": "3880", **_slurm_identity(tmp_path)})
    validate_scheduler_run_identity({"status": "stopping", "scheduler_job_id": "3880", **_slurm_identity(tmp_path)})

    with pytest.raises(ValueError, match="PID process identity"):
        validate_scheduler_run_identity(
            {"status": "queued", "scheduler_job_id": "3880", "pid": "42", **_slurm_identity(tmp_path)}
        )
    with pytest.raises(ValueError, match="cannot define Slurm"):
        validate_scheduler_run_identity(
            {"scheduler_type": "direct", "scheduler_script": "/plan/job.sbatch", "status": "planned"}
        )
    with pytest.raises(ValueError, match="cannot define Slurm"):
        validate_scheduler_run_identity({"scheduler_type": "direct", "scheduler_job_id": "3880", "status": "queued"})
    with pytest.raises(ValueError, match="missing scheduler plan identity"):
        validate_scheduler_run_identity({"scheduler_type": "slurm", "status": "planned"})
    with pytest.raises(ValueError, match="requires scheduler_job_id"):
        validate_scheduler_run_identity({"status": "queued", **_slurm_identity(tmp_path)})


def test_manifest_binds_scheduler_job_once_and_updates_observation(tmp_path: Path):
    (tmp_path / "experiment.yaml").write_text("experiment:\n  id: unit\n")
    initialize_run_manifest(tmp_path)
    identity = {
        "experiment_id": "unit",
        "step_id": "train",
        "run_id": "run-000",
        "target": "local",
        **_slurm_identity(tmp_path),
    }
    merge_run_manifest(tmp_path, [{**identity, "status": "planned"}])
    committed = merge_run_manifest(
        tmp_path,
        [
            {
                "step_id": "train",
                "run_id": "run-000",
                "status": "queued",
                "scheduler_job_id": "3880",
                "scheduler_raw_state": "PENDING",
            }
        ],
    )

    assert committed[0]["scheduler_job_id"] == "3880"
    assert committed[0]["scheduler_raw_state"] == "PENDING"
    updated = merge_run_manifest(
        tmp_path,
        [{"step_id": "train", "run_id": "run-000", "scheduler_raw_state": "RUNNING"}],
    )
    assert updated[0]["scheduler_raw_state"] == "RUNNING"
    with pytest.raises(ValueError, match="scheduler_job_id"):
        merge_run_manifest(
            tmp_path,
            [{"step_id": "train", "run_id": "run-000", "scheduler_job_id": "3881"}],
        )


def test_scheduler_active_status_does_not_regress_to_pending():
    for status in ("queued", "lost"):
        assert merge_run_row({"status": status}, {"status": "pending"})["status"] == status


def test_semantic_run_name_keeps_boolean_settings_readable():
    assert (
        semantic_run_name({"runtime.lr": 2e-6, "yaml:/model/router_frozen": True, "yaml:/loss/class_weights": False})
        == "lr-2e-6__class-weights-off__router-frozen"
    )


@pytest.mark.parametrize(
    ("parameters", "expected_name", "expected_summary"),
    [
        (
            {"runtime.weight_decay": 0.1, "runtime.lr": 2e-6},
            "lr-2e-6__weight-decay-0.1",
            "runtime.lr=2e-06; runtime.weight_decay=0.1",
        ),
        (
            {"yaml:/model/lr": 0.1, "runtime.lr": 2e-6},
            "lr-2e-6__yaml-model-lr-0.1",
            "runtime.lr=2e-06; yaml:/model/lr=0.1",
        ),
    ],
)
def test_semantic_run_identity_is_independent_of_parameter_key_order(parameters, expected_name, expected_summary):
    recipe = {"experiment": {"id": "unit"}, "step": {"id": "tune"}}
    reordered = json.loads(json.dumps(parameters, sort_keys=True))

    for mapping in (parameters, reordered):
        assert semantic_run_name(mapping) == expected_name
        assert parameter_summary(mapping) == expected_summary
        assert run_identity(recipe, 7, mapping) == {
            "run_id": "run-007",
            "run_name": expected_name,
            "version": f"unit__tune__run-007__{expected_name}",
        }


def test_long_semantic_identity_canonicalizes_nested_mappings_but_preserves_list_order():
    recipe = {"experiment": {"id": "unit"}, "step": {"id": "tune"}}
    parameters = {
        "yaml:/model/options": {"z": {"beta": 2, "alpha": 1}, "a": [{"z": 2, "a": 1}, "second"]},
        "yaml:/model/description": "long-description-" * 12,
        "yaml:/model/layers": [3, 1, 2],
        "runtime.lr": 2e-6,
    }
    reordered = json.loads(json.dumps(parameters, sort_keys=True))
    run_name = semantic_run_name(parameters)

    assert len(run_name) <= 100
    assert "--h" in run_name
    assert semantic_run_name(reordered) == run_name
    assert parameter_summary(reordered) == parameter_summary(parameters)
    assert run_identity(recipe, 7, reordered) == run_identity(recipe, 7, parameters)

    reordered["yaml:/model/layers"].reverse()
    assert semantic_run_name(reordered) != run_name
    assert parameter_summary(reordered) != parameter_summary(parameters)
    assert run_identity(recipe, 7, reordered) != run_identity(recipe, 7, parameters)


def test_step_manifest_merge_preserves_registered_fields_and_appends_plans():
    existing = {
        "step": {"id": "train", "phase": "train", "purpose": "Tune the model."},
        "experiment_id": "experiment",
        "plan_controller": "ordinary",
        "recipe_path": "recipes/first.yaml",
        "plans": ["/workspace/plan-a"],
    }

    merged = merge_step_manifest(
        existing,
        {
            "step": {"id": "train", "phase": "train", "purpose": "Tune the model."},
            "experiment_id": "experiment",
            "plan_controller": "ordinary",
            "recipe_path": "recipes/second.yaml",
            "plans": ["/workspace/plan-a", "/workspace/plan-b"],
        },
    )

    assert merged["step"] == existing["step"]
    assert merged["recipe_path"] == "recipes/first.yaml"
    assert merged["plans"] == ["/workspace/plan-a", "/workspace/plan-b"]


def test_concurrent_step_manifest_commits_preserve_both_plans(tmp_path: Path, monkeypatch):
    base_payload = {
        "step": {"id": "train", "phase": "train", "purpose": "Tune the model."},
        "experiment_id": "experiment",
        "plan_controller": "ordinary",
        "recipe_path": "/workspace/recipe.yaml",
        "plans": ["/workspace/plan-base"],
    }
    commit_step_manifest(tmp_path, base_payload)
    first_inside = threading.Event()
    real_exists = experiment_io.path_exists_at
    probes = 0
    probe_lock = threading.Lock()

    def slow_first_probe(path):
        # Hold the first committer between its read and its write while the second one starts.
        nonlocal probes
        if Path(path).name == "step.yaml":
            with probe_lock:
                probes += 1
                first = probes == 1
            if first:
                first_inside.set()
                time.sleep(0.3)
        return real_exists(path)

    monkeypatch.setattr(experiment_io, "path_exists_at", slow_first_probe)
    errors = []

    def commit(plan: str):
        try:
            commit_step_manifest(
                tmp_path,
                {
                    "step": {"id": "train", "phase": "train", "purpose": "Tune the model."},
                    "experiment_id": "experiment",
                    "plan_controller": "ordinary",
                    "recipe_path": "/workspace/recipe.yaml",
                    "plans": [plan],
                },
            )
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=commit, args=(plan,)) for plan in ("/workspace/plan-a", "/workspace/plan-b")]
    threads[0].start()
    assert first_inside.wait(timeout=5)
    threads[1].start()
    for thread in threads:
        thread.join(timeout=5)

    assert not any(thread.is_alive() for thread in threads)
    assert errors == []
    assert set(read_step_manifest(tmp_path, "train")["plans"]) == {
        "/workspace/plan-base",
        "/workspace/plan-a",
        "/workspace/plan-b",
    }


def test_step_manifest_merge_rejects_metadata_drift():
    existing = {
        "step": {"id": "train", "phase": "train", "purpose": "Tune the model."},
        "experiment_id": "experiment",
        "plan_controller": "ordinary",
        "recipe_path": "",
        "plans": [],
    }

    with pytest.raises(ValueError, match="phase"):
        merge_step_manifest(existing, {"step": {"phase": "analyze"}})


def test_step_manifest_commit_rejects_missing_controller_without_writing(tmp_path: Path):
    step = {"id": "train", "phase": "train", "purpose": "Tune the model."}
    step_dir = tmp_path / "steps" / "train"
    step_dir.mkdir(parents=True)
    step_path = step_dir / "step.yaml"
    step_path.write_text(
        yaml.safe_dump(
            {
                "step": step,
                "experiment_id": "experiment",
                "recipe_path": "",
                "plans": [],
            },
            sort_keys=False,
        )
    )
    before = step_path.read_bytes()

    with pytest.raises(ValueError, match="incomplete canonical envelope"):
        commit_step_manifest(
            tmp_path,
            {
                "step": step,
                "experiment_id": "experiment",
                "plan_controller": "ordinary",
                "recipe_path": "/workspace/recipe.yaml",
                "plans": ["/workspace/plan"],
            },
        )

    assert step_path.read_bytes() == before


@pytest.mark.parametrize(
    "existing",
    [
        "",
        "null\n",
        "{}\n",
        (
            "step:\n"
            "  id: unit-hparam-tune\n"
            "  phase: train\n"
            "  phase: analyze\n"
            "  purpose: Tune hyperparameters.\n"
            "experiment_id: unit-experiment\n"
            "recipe_path: /tmp/recipe.yaml\n"
            "plans: []\n"
        ),
        (
            "step:\n"
            "  id: unit-hparam-tune\n"
            "  phase: analyze\n"
            "  purpose: Tune hyperparameters.\n"
            "experiment_id: unit-experiment\n"
            "recipe_path: /tmp/recipe.yaml\n"
            "plans: []\n"
        ),
    ],
)
def test_planner_rejects_corrupt_existing_step_manifest_without_writing(tmp_path: Path, existing: str):
    recipe = _hparam_recipe(tmp_path)
    first = tmp_path / "plans" / "first"
    assert run_cli("plan", "--recipe", str(recipe), "--output-dir", str(first)).returncode == 0
    payload = json.loads((first / "plan.json").read_text())["recipe"]
    target = tmp_path / "steps" / payload["step"]["id"] / "step.yaml"
    target.write_text(existing)
    before = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    plan_dir = tmp_path / "plans" / "corrupt-step"

    result = run_cli("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir))

    assert result.returncode == 1
    assert "step manifest" in result.stderr
    assert not plan_dir.exists()
    assert {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before


def test_new_plan_continues_run_ids_within_the_same_step(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path)
    first = tmp_path / "plans" / "first"
    second = tmp_path / "plans" / "second"

    assert run_cli("plan", "--recipe", str(recipe), "--output-dir", str(first)).returncode == 0
    assert run_cli("plan", "--recipe", str(recipe), "--output-dir", str(second)).returncode == 0

    first_run = json.loads((first / "plan.json").read_text())["runs"][0]
    second_run = json.loads((second / "plan.json").read_text())["runs"][0]
    assert first_run["run_id"] == "run-000"
    assert second_run["run_id"] == "run-001"


def test_missing_canonical_manifest_never_resets_run_identity(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path)
    first = tmp_path / "plans" / "first"
    second = tmp_path / "plans" / "second"
    assert run_cli("plan", "--recipe", str(recipe), "--output-dir", str(first)).returncode == 0
    (tmp_path / "run_manifest.tsv").unlink()
    before = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}

    result = run_cli("plan", "--recipe", str(recipe), "--output-dir", str(second))

    assert result.returncode == 1
    assert "run_manifest.tsv" in result.stderr
    assert {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before
    assert not second.exists()


def test_planner_rejects_duplicate_workspace_ownership_without_writing(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path)
    first = tmp_path / "plans" / "first"
    second = tmp_path / "plans" / "second"
    assert run_cli("plan", "--recipe", str(recipe), "--output-dir", str(first)).returncode == 0
    manifest = tmp_path / "experiment.yaml"
    manifest.write_text(
        manifest.read_text().replace("  id: unit-experiment\n", "  id: foreign\n  id: unit-experiment\n")
    )
    before = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}

    result = run_cli("plan", "--recipe", str(recipe), "--output-dir", str(second))

    assert result.returncode == 1
    assert "Status: FAIL" in result.stdout
    assert "duplicate key" in result.stdout
    assert {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before
    assert not second.exists()


def test_completed_experiment_rejects_new_plan(tmp_path: Path):
    recipe = write_finetune_recipe(tmp_path)
    first = tmp_path / "plans" / "first"
    second = tmp_path / "plans" / "second"
    assert run_cli("plan", "--recipe", str(recipe), "--output-dir", str(first)).returncode == 0
    merge_run_manifest(tmp_path, [{**read_run_manifest(tmp_path)[0], "status": "finished"}])
    report = tmp_path / "final_source.md"
    report.write_text("# Final\n")
    assert run_cli("experiment-finalize", "--run-dir", str(tmp_path), "--report", str(report)).returncode == 0

    result = run_cli("plan", "--recipe", str(recipe), "--output-dir", str(second))

    assert result.returncode == 1
    assert "completed" in result.stdout
    assert not (second / "plan.json").exists()
    with (tmp_path / "run_manifest.tsv").open(newline="") as file_obj:
        rows = list(csv.DictReader(file_obj, delimiter="\t"))
    assert [(row["run_id"], row["status"]) for row in rows] == [("run-000", "finished")]


def test_single_run_plan_co_locates_frozen_snapshots(tmp_path: Path):
    recipe = write_finetune_recipe(tmp_path)
    source_config = Path(yaml.safe_load(recipe.read_text())["inputs"]["config"])
    plan_dir = tmp_path / "plan"

    result = run_cli("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir))

    assert result.returncode == 0, result.stderr
    run_dir = plan_dir / "runs" / "run-000--unit"
    assert {path.name for path in run_dir.iterdir()} == {
        "artifacts.json",
        "config.yaml",
        "launch.sh",
        "run.json",
    }
    run = json.loads((run_dir / "run.json").read_text())
    config_path = run_dir / "config.yaml"
    frozen_config = config_path.read_text()
    assert run["version"] == "unit-experiment__unit-finetune__run-000__unit"
    assert run["config_sha256"] == file_sha256(config_path)
    launch = (run_dir / "launch.sh").read_text()
    assert str(config_path) in launch
    assert str(source_config) not in launch
    assert run["version"] in launch
    source_config.write_text("changed: true\n")
    assert config_path.read_text() == frozen_config
    assert str(config_path) in (plan_dir / "run.sh").read_text()
    with (tmp_path / "run_manifest.tsv").open(newline="") as file_obj:
        rows = list(csv.DictReader(file_obj, delimiter="\t"))
    assert rows[0]["version"] == run["version"]
    assert rows[0]["runtime_dir"] == run["runtime_dir"]
    assert rows[0]["checkpoint_dir"] == run["checkpoint_dir"]


def test_single_run_versions_are_unique_across_repeated_plans(tmp_path: Path):
    recipe = write_finetune_recipe(tmp_path)
    first = tmp_path / "plans" / "first"
    second = tmp_path / "plans" / "second"

    assert run_cli("plan", "--recipe", str(recipe), "--output-dir", str(first)).returncode == 0
    assert run_cli("plan", "--recipe", str(recipe), "--output-dir", str(second)).returncode == 0

    first_run = json.loads((first / "runs" / "run-000--unit" / "run.json").read_text())
    second_run = json.loads((second / "runs" / "run-001--unit" / "run.json").read_text())
    assert first_run["run_name"] == second_run["run_name"] == "unit"
    assert first_run["version"] != second_run["version"]
    assert "run-000" in first_run["version"]
    assert "run-001" in second_run["version"]


def test_single_run_plan_directory_may_equal_fresh_experiment_root(tmp_path: Path):
    recipe = write_finetune_recipe(tmp_path / "source")
    workspace = tmp_path / "workspace"
    payload = yaml.safe_load(recipe.read_text())
    payload["experiment"]["root"] = str(workspace)
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))

    report = plans.build_plan(recipe_path=recipe, output_dir=workspace)

    assert report.exit_code == 0
    assert (workspace / "plan.json").is_file()
    assert (workspace / "experiment.yaml").is_file()
    assert [row["run_id"] for row in read_run_manifest(workspace)] == ["run-000"]
    assert not list(tmp_path.glob(".workspace.*.staging"))


@pytest.mark.parametrize("change_step", [False, True], ids=["same-step", "different-step"])
def test_repeated_single_run_plan_rejects_registered_output_directory(tmp_path: Path, change_step: bool):
    recipe = write_finetune_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    payload["artifacts"]["overwrite"] = True
    payload["decisions"]["overwrite_policy"] = {"value": True, "source": "explicit_recipe"}
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    plan_dir = tmp_path / "plan"

    first = run_cli("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir))
    assert first.returncode == 0, first.stderr
    first_run_dir = plan_dir / "runs" / "run-000--unit"
    first_run_bytes = {
        path.relative_to(first_run_dir): path.read_bytes() for path in first_run_dir.rglob("*") if path.is_file()
    }
    first_plan_bytes = (plan_dir / "plan.json").read_bytes()
    if change_step:
        payload["step"] = {
            "id": "unit-follow-up",
            "phase": "train",
            "purpose": "Exercise cross-step plan ownership.",
        }
        payload["artifacts"]["version_name"] = "unit-follow-up"
        recipe.write_text(yaml.safe_dump(payload, sort_keys=False))

    second = run_cli("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir))

    assert second.returncode == 1
    assert "Registered plan directories are immutable" in second.stdout
    assert {
        path.relative_to(first_run_dir): path.read_bytes() for path in first_run_dir.rglob("*") if path.is_file()
    } == first_run_bytes
    assert (plan_dir / "plan.json").read_bytes() == first_plan_bytes
    assert not (plan_dir / "runs" / "run-001--unit").exists()
    assert [row["run_id"] for row in read_run_manifest(tmp_path)] == ["run-000"]


def test_single_run_plan_recovers_registered_plan_after_manifest_failure(tmp_path: Path, monkeypatch):
    recipe = write_finetune_recipe(tmp_path)
    plan_dir = tmp_path / "plans" / "interrupted"
    real_merge = plans.merge_run_manifest
    failed = False

    def fail_first_merge(root, rows):
        nonlocal failed
        if not failed:
            failed = True
            raise RuntimeError("injected canonical commit failure")
        return real_merge(root, rows)

    monkeypatch.setattr(plans, "merge_run_manifest", fail_first_merge)
    with pytest.raises(RuntimeError, match="injected canonical commit failure"):
        plans.build_plan(recipe_path=recipe, output_dir=plan_dir)

    frozen_tree = run_artifacts.plan_tree_sha256(plan_dir)
    assert read_run_manifest(tmp_path) == []
    step = read_step_manifest(tmp_path, "unit-finetune")
    assert step["plans"] == [str(plan_dir)]

    with pytest.raises(ValueError, match="recover it before creating another plan"):
        plans.build_plan(recipe_path=recipe, output_dir=tmp_path / "plans" / "fresh")
    assert not (tmp_path / "plans" / "fresh").exists()

    other_payload = yaml.safe_load(recipe.read_text())
    other_payload["step"] = {
        "id": "other-step",
        "phase": "train",
        "purpose": "Register an intervening canonical run.",
    }
    other_payload["artifacts"]["version_name"] = "other"
    other_recipe = write_yaml(tmp_path / "other.yaml", other_payload)
    other_report = plans.build_plan(recipe_path=other_recipe, output_dir=tmp_path / "plans" / "other")
    assert other_report.exit_code == 0

    recovered = plans.build_plan(recipe_path=recipe, output_dir=plan_dir)

    assert recovered.exit_code == 0
    assert run_artifacts.plan_tree_sha256(plan_dir) == frozen_tree
    rows = read_run_manifest(tmp_path)
    assert {(row["step_id"], row["run_id"]) for row in rows} == {
        ("unit-finetune", "run-000"),
        ("other-step", "run-000"),
    }
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    created = [
        event for event in events if event["event_type"] == "plan_created" and event["plan_dir"] == str(plan_dir)
    ]
    assert len(created) == 1


@pytest.mark.parametrize("tamper", [False, True], ids=["exact", "drifted"])
def test_root_single_run_plan_recovers_registered_plan_after_manifest_failure(
    tmp_path: Path,
    monkeypatch,
    tamper: bool,
):
    recipe = write_finetune_recipe(tmp_path / "source")
    workspace = tmp_path / "workspace"
    payload = yaml.safe_load(recipe.read_text())
    payload["experiment"]["root"] = str(workspace)
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    real_merge = plans.merge_run_manifest
    monkeypatch.setattr(
        plans,
        "merge_run_manifest",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("injected canonical commit failure")),
    )

    with pytest.raises(RuntimeError, match="injected canonical commit failure"):
        plans.build_plan(recipe_path=recipe, output_dir=workspace)
    monkeypatch.setattr(plans, "merge_run_manifest", real_merge)

    assert read_step_manifest(workspace, "unit-finetune")["plans"] == [str(workspace)]
    assert read_run_manifest(workspace) == []
    if tamper:
        next((workspace / "runs").glob("*/run.json")).write_text('{"tampered": true}\n')

    if tamper:
        with pytest.raises(ValueError, match="differs from deterministic regeneration"):
            plans.build_plan(recipe_path=recipe, output_dir=workspace)
        assert read_run_manifest(workspace) == []
        assert not any(
            event["event_type"] == "plan_created"
            for event in (json.loads(line) for line in (workspace / "events.jsonl").read_text().splitlines())
        )
        return

    recovered = plans.build_plan(recipe_path=recipe, output_dir=workspace)

    assert recovered.exit_code == 0
    assert [row["run_id"] for row in read_run_manifest(workspace)] == ["run-000"]
    created = [
        event
        for event in (json.loads(line) for line in (workspace / "events.jsonl").read_text().splitlines())
        if event["event_type"] == "plan_created" and event["plan_dir"] == str(workspace)
    ]
    assert len(created) == 1
    assert not list(tmp_path.glob(".workspace.*.staging"))


@pytest.mark.parametrize("overwrite", [False, True])
@pytest.mark.parametrize("mutate", [False, True], ids=["exact", "drifted"])
def test_single_run_plan_handles_unowned_publication(
    tmp_path: Path,
    monkeypatch,
    overwrite: bool,
    mutate: bool,
):
    recipe = write_finetune_recipe(tmp_path)
    if overwrite:
        payload = yaml.safe_load(recipe.read_text())
        payload["artifacts"]["overwrite"] = True
        payload["decisions"]["overwrite_policy"] = {"value": True, "source": "explicit_recipe"}
        recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    plan_dir = tmp_path / "plans" / "interrupted"
    real_publish = plans.publish_staged_plan_locked

    def publish_then_fail(*args, **kwargs):
        real_publish(*args, **kwargs)
        raise RuntimeError("injected post-publication failure")

    monkeypatch.setattr(plans, "publish_staged_plan_locked", publish_then_fail)
    with pytest.raises(RuntimeError, match="injected post-publication failure"):
        plans.build_plan(recipe_path=recipe, output_dir=plan_dir)
    monkeypatch.setattr(plans, "publish_staged_plan_locked", real_publish)

    frozen_tree = run_artifacts.plan_tree_sha256(plan_dir)
    assert read_step_manifest(tmp_path, "unit-finetune", allow_missing=True) is None
    assert read_run_manifest(tmp_path) == []
    if mutate:
        next((plan_dir / "runs").glob("*/run.json")).write_text('{"tampered": true}\n')

    if mutate and not overwrite:
        with pytest.raises(ValueError, match="differs from deterministic regeneration"):
            plans.build_plan(recipe_path=recipe, output_dir=plan_dir)
        assert run_artifacts.plan_tree_sha256(plan_dir) != frozen_tree
        assert read_step_manifest(tmp_path, "unit-finetune", allow_missing=True) is None
        assert read_run_manifest(tmp_path) == []
        return

    recovered = plans.build_plan(recipe_path=recipe, output_dir=plan_dir)
    assert recovered.exit_code == 0
    assert run_artifacts.plan_tree_sha256(plan_dir) == frozen_tree
    assert read_step_manifest(tmp_path, "unit-finetune")["plans"] == [str(plan_dir)]
    assert [row["run_id"] for row in read_run_manifest(tmp_path)] == ["run-000"]
    created = [
        event
        for event in (json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines())
        if event["event_type"] == "plan_created" and event["plan_dir"] == str(plan_dir)
    ]
    assert len(created) == 1


@pytest.mark.parametrize("field", ["config", "runtime.foreign"])
def test_complete_registration_recovery_rejects_foreign_canonical_row(tmp_path: Path, field: str):
    recipe = write_finetune_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    assert plans.build_plan(recipe_path=recipe, output_dir=plan_dir).exit_code == 0
    manifest_path = tmp_path / "run_manifest.tsv"
    rows = read_run_manifest(tmp_path)
    rows[0][field] = str(tmp_path / "foreign.yaml") if field == "config" else "foreign"
    with manifest_path.open("w", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=sorted(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    before = manifest_path.read_bytes()

    with pytest.raises(ValueError, match=f"Frozen run field differs.*{field}"):
        plans.build_plan(recipe_path=recipe, output_dir=plan_dir)

    assert manifest_path.read_bytes() == before


def test_single_run_registration_recovery_rejects_changed_frozen_plan(tmp_path: Path, monkeypatch):
    recipe = write_finetune_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    real_merge = plans.merge_run_manifest
    monkeypatch.setattr(
        plans,
        "merge_run_manifest",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("injected canonical commit failure")),
    )
    with pytest.raises(RuntimeError, match="injected canonical commit failure"):
        plans.build_plan(recipe_path=recipe, output_dir=plan_dir)
    monkeypatch.setattr(plans, "merge_run_manifest", real_merge)

    frozen_tree = run_artifacts.plan_tree_sha256(plan_dir)
    payload = yaml.safe_load(recipe.read_text())
    payload["runtime"]["lr"] = 9e-6
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))

    with pytest.raises(ValueError, match="differs from deterministic regeneration"):
        plans.build_plan(recipe_path=recipe, output_dir=plan_dir)

    assert run_artifacts.plan_tree_sha256(plan_dir) == frozen_tree
    assert read_run_manifest(tmp_path) == []
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert not any(event["event_type"] == "plan_created" for event in events)
    assert not list(plan_dir.parent.glob(f".{plan_dir.name}.*.staging"))


@pytest.mark.parametrize("different_step", [False, True], ids=["same-step", "different-step"])
def test_plan_registration_serializes_run_index_allocation_and_workspace_initialization(
    tmp_path: Path,
    monkeypatch,
    different_step: bool,
):
    recipe = write_finetune_recipe(tmp_path)
    second_recipe = recipe
    if different_step:
        payload = yaml.safe_load(recipe.read_text())
        payload["step"] = {
            "id": "other-step",
            "phase": "train",
            "purpose": "Exercise concurrent workspace initialization.",
        }
        second_recipe = write_yaml(tmp_path / "other.yaml", payload)
    original_check = plans._assert_no_incomplete_step_registration
    first_checking = threading.Event()
    second_checking = threading.Event()
    release_first = threading.Event()
    calls_lock = threading.Lock()
    calls = 0

    def pause_first_check(recipe_payload, out):
        nonlocal calls
        with calls_lock:
            calls += 1
            call = calls
        if call == 1:
            first_checking.set()
            assert release_first.wait(timeout=10)
        else:
            second_checking.set()
        return original_check(recipe_payload, out)

    monkeypatch.setattr(plans, "_assert_no_incomplete_step_registration", pause_first_check)
    reports = []
    errors = []

    def create_plan(recipe_path, name):
        try:
            reports.append(plans.build_plan(recipe_path=recipe_path, output_dir=tmp_path / "plans" / name))
        except BaseException as exc:
            errors.append(exc)

    first = threading.Thread(target=create_plan, args=(recipe, "first"))
    second = threading.Thread(target=create_plan, args=(second_recipe, "second"))
    first.start()
    assert first_checking.wait(timeout=10)
    second.start()
    assert not second_checking.wait(timeout=0.2)
    release_first.set()
    first.join(timeout=20)
    second.join(timeout=20)

    assert not first.is_alive()
    assert not second.is_alive()
    assert errors == []
    assert [report.exit_code for report in reports] == [0, 0]
    run_ids = {
        json.loads((tmp_path / "plans" / name / "plan.json").read_text())["runs"][0]["run_id"]
        for name in ("first", "second")
    }
    assert run_ids == ({"run-000"} if different_step else {"run-000", "run-001"})


def test_plan_registration_serializes_fresh_workspace_initialization(tmp_path: Path, monkeypatch):
    source_dir = tmp_path / "source"
    recipe = write_finetune_recipe(source_dir)
    workspace = tmp_path / "fresh-workspace"
    payload = yaml.safe_load(recipe.read_text())
    payload["experiment"]["root"] = str(workspace)
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    second_payload = yaml.safe_load(recipe.read_text())
    second_payload["step"] = {
        "id": "other-step",
        "phase": "train",
        "purpose": "Exercise concurrent workspace initialization.",
    }
    second_recipe = source_dir / "other.yaml"
    second_recipe.write_text(yaml.safe_dump(second_payload, sort_keys=False))

    original_initialize = experiment_workspace.initialize_run_manifest
    first_manifest_written = threading.Event()
    release_first = threading.Event()
    second_checking = threading.Event()

    def pause_first_initialization(root):
        if threading.current_thread().name == "first-planner":
            first_manifest_written.set()
            if not release_first.wait(timeout=10):
                raise AssertionError("first planner was not released")
        return original_initialize(root)

    original_check = plans._assert_no_incomplete_step_registration

    def observe_second_check(recipe_payload, out):
        if threading.current_thread().name == "second-planner":
            second_checking.set()
        return original_check(recipe_payload, out)

    monkeypatch.setattr(experiment_workspace, "initialize_run_manifest", pause_first_initialization)
    monkeypatch.setattr(plans, "_assert_no_incomplete_step_registration", observe_second_check)
    reports = []
    errors = []

    def create_plan(recipe_path, name):
        try:
            reports.append(plans.build_plan(recipe_path=recipe_path, output_dir=workspace / "plans" / name))
        except BaseException as exc:
            errors.append(exc)

    first = threading.Thread(target=create_plan, args=(recipe, "first"), name="first-planner")
    second = threading.Thread(target=create_plan, args=(second_recipe, "second"), name="second-planner")
    first.start()
    assert first_manifest_written.wait(timeout=10)
    second.start()
    try:
        assert not second_checking.wait(timeout=0.5)
    finally:
        release_first.set()
    first.join(timeout=20)
    second.join(timeout=20)

    if errors:
        raise errors[0]
    assert len(reports) == 2
    assert all(report.exit_code == 0 for report in reports)
    assert {(row["step_id"], row["run_id"]) for row in read_run_manifest(workspace)} == {
        ("unit-finetune", "run-000"),
        ("other-step", "run-000"),
    }


def test_plan_registration_lock_is_shared_across_controller_temp_roots(tmp_path: Path):
    workspace = tmp_path / "workspace"
    recipe = write_finetune_recipe(workspace)
    release = tmp_path / "release"
    runner = """
import sys
import time
from pathlib import Path

from agent_tools import plans

recipe = Path(sys.argv[1])
output = Path(sys.argv[2])
entered = Path(sys.argv[3])
release = Path(sys.argv[4])
wait_for_release = sys.argv[5] == "wait"
original_check = plans._assert_no_incomplete_step_registration

def observe_check(recipe_payload, out):
    entered.touch()
    if wait_for_release:
        while not release.exists():
            time.sleep(0.01)
    return original_check(recipe_payload, out)

plans._assert_no_incomplete_step_registration = observe_check
report = plans.build_plan(recipe_path=recipe, output_dir=output)
raise SystemExit(report.exit_code)
"""
    processes = []
    second_entered_early = False
    try:
        for name in ("first", "second"):
            lock_root = tmp_path / f"{name}-tmp"
            lock_root.mkdir()
            env = {**os.environ, "TMPDIR": str(lock_root)}
            processes.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        "-c",
                        runner,
                        str(recipe),
                        str(workspace / "plans" / name),
                        str(tmp_path / f"{name}-entered"),
                        str(release),
                        "wait" if name == "first" else "continue",
                    ],
                    cwd=Path(__file__).parents[2],
                    env=env,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
            )
            if name == "first":
                entered = tmp_path / "first-entered"
                deadline = time.monotonic() + SUBPROCESS_WAIT_SECONDS
                while not entered.exists() and time.monotonic() < deadline:
                    time.sleep(0.01)
                assert entered.exists()
        time.sleep(0.5)
        second_entered_early = (tmp_path / "second-entered").exists()
    finally:
        release.touch()
    results = [process.communicate(timeout=SUBPROCESS_WAIT_SECONDS) for process in processes]

    assert not second_entered_early
    assert [process.returncode for process in processes] == [0, 0], results
    assert (tmp_path / "second-entered").exists()
    run_ids = {
        json.loads((workspace / "plans" / name / "plan.json").read_text())["runs"][0]["run_id"]
        for name in ("first", "second")
    }
    assert run_ids == {"run-000", "run-001"}
    assert {row["run_id"] for row in read_run_manifest(workspace)} == run_ids
    assert set(read_step_manifest(workspace, "unit-finetune")["plans"]) == {
        str(workspace / "plans" / "first"),
        str(workspace / "plans" / "second"),
    }
    events = [json.loads(line) for line in (workspace / "events.jsonl").read_text().splitlines()]
    assert len([event for event in events if event["event_type"] == "plan_created"]) == 2


def test_plan_publication_lock_is_shared_across_controller_temp_roots(tmp_path: Path):
    output = tmp_path / "plan"
    release = tmp_path / "release"
    runner = """
import sys
import time
from pathlib import Path

from agent_tools.plans import plan_publication_lock

output = Path(sys.argv[1])
entered = Path(sys.argv[2])
release = Path(sys.argv[3])
wait_for_release = sys.argv[4] == "wait"
with plan_publication_lock(output):
    entered.touch()
    if wait_for_release:
        while not release.exists():
            time.sleep(0.01)
"""
    processes = []
    second_entered_early = False
    try:
        for name in ("first", "second"):
            temp_root = tmp_path / f"{name}-tmp"
            temp_root.mkdir()
            env = {**os.environ, "TMPDIR": str(temp_root)}
            processes.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        "-c",
                        runner,
                        str(output),
                        str(tmp_path / f"{name}-entered"),
                        str(release),
                        "wait" if name == "first" else "continue",
                    ],
                    cwd=Path(__file__).parents[2],
                    env=env,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
            )
            if name == "first":
                entered = tmp_path / "first-entered"
                deadline = time.monotonic() + SUBPROCESS_WAIT_SECONDS
                while not entered.exists() and time.monotonic() < deadline:
                    time.sleep(0.01)
                assert entered.exists()
        time.sleep(0.5)
        second_entered_early = (tmp_path / "second-entered").exists()
    finally:
        release.touch()
    results = [process.communicate(timeout=SUBPROCESS_WAIT_SECONDS) for process in processes]

    assert not second_entered_early
    assert [process.returncode for process in processes] == [0, 0], results
    assert (tmp_path / "second-entered").exists()


def test_plan_registration_lock_cannot_deadlock_with_plan_output(tmp_path: Path):
    recipe = write_finetune_recipe(tmp_path)
    output = tmp_path / "steps" / "unit-finetune" / "step.yaml"

    # A real process, so a regression that deadlocks registration against plan output ends in the timeout, not a hang.
    result = run_cli_subprocess("plan", "--recipe", str(recipe), "--output-dir", str(output))

    assert result.returncode != 0


def test_monitor_observation_fields_are_not_persisted_in_the_run_manifest(tmp_path: Path):
    (tmp_path / "experiment.yaml").write_text("experiment:\n  id: unit\n")
    initialize_run_manifest(tmp_path)
    identity = {"experiment_id": "unit", "step_id": "train", "run_id": "run-000"}
    observation = {
        **identity,
        "status": "running",
        "health_status": "healthy_running",
        "log_tail": "epoch 1\nepoch 2",
        "log_age_seconds": 3,
        "monitored_at": "2026-01-01T00:00:00Z",
        "gpu_summary": "0, GPU-0, 1024",
    }

    committed = merge_run_manifest(tmp_path, [observation])

    header = (tmp_path / "run_manifest.tsv").read_text().splitlines()[0].split("\t")
    assert not set(header) & experiment_workspace.MONITOR_OBSERVATION_FIELDS
    assert committed == read_run_manifest(tmp_path)
    assert committed[0]["health_status"] == "healthy_running"
