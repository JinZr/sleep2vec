from __future__ import annotations

import ast
import csv
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import threading

from agent_tool_test_helpers import write_finetune_recipe, write_yaml
import pytest
import yaml

from agent_tools import experiment_io, experiment_workspace, experiments, hparam, hparam_runtime, plans, run_artifacts
from agent_tools.experiment_workspace import (
    EXECUTION_IDENTITY_FIELDS,
    MANAGED_RUN_PATH_FIELDS,
    append_event,
    canonical_local_experiment_root,
    ensure_experiment_workspace,
    file_sha256,
    initialize_run_manifest,
    managed_run_key,
    managed_run_parameters,
    merge_run_manifest,
    merge_run_row,
    merge_step_manifest,
    read_run_manifest,
    read_step_manifest,
    resolve_external_run_row,
    resolve_run_row,
    run_evidence_key,
    semantic_run_name,
    validate_frozen_run_update,
    validate_managed_run_rows,
)


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "agent_tools", *args], text=True, capture_output=True)


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
                "train_val_test_policy": {"value": "select on val", "source": "explicit_recipe"},
                "overwrite_policy": {"value": False, "source": "explicit_recipe"},
                "final_eval_unlock": {"value": False, "source": "explicit_recipe"},
            },
        },
    )


def test_managed_plan_writes_semantic_run_workspace_without_schema_version(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path)
    plan_dir = tmp_path / "steps" / "tune" / "plan"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir))

    assert result.returncode == 0, result.stderr
    run = json.loads((plan_dir / "plan.json").read_text())["runs"][0]
    assert run["run_id"] == "run-000"
    assert run["run_name"] == "lr-2e-6"
    assert Path(run["run_dir"]).name == "run-000--lr-2e-6"
    assert Path(run["config"]).name == "config.yaml"
    assert Path(run["script"]).name == "launch.sh"
    assert (tmp_path / "experiment.yaml").exists()
    assert (tmp_path / "run_matrix.csv").exists()
    assert (tmp_path / "run_manifest.tsv").exists()
    with (tmp_path / "run_matrix.csv").open(newline="") as file_obj:
        matrix = list(csv.DictReader(file_obj))
    assert matrix[0]["run_name"] == "lr-2e-6"
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


def test_registered_step_is_extended_by_plan_and_allows_dry_run_launch(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path)
    recipe_payload = yaml.safe_load(recipe.read_text())
    experiment_spec = tmp_path / "experiment-spec.yaml"
    experiment_spec.write_text(yaml.safe_dump({"experiment": recipe_payload["experiment"]}, sort_keys=False))
    step_spec = tmp_path / "step-spec.yaml"
    step_spec.write_text(
        yaml.safe_dump(
            {
                **recipe_payload["step"],
                "inputs": ["config.yaml"],
                "outputs": ["reports/ranking.csv"],
            },
            sort_keys=False,
        )
    )

    assert _run("experiment-init", "--run-dir", str(tmp_path), "--spec", str(experiment_spec)).returncode == 0
    registered = _run("experiment-register-step", "--run-dir", str(tmp_path), "--spec", str(step_spec))
    assert registered.returncode == 0, registered.stderr
    plan_dir = tmp_path / "plans" / "registered"
    planned = _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir))
    assert planned.returncode == 0, planned.stderr
    launched = _run("hparam-launch", "--plan-dir", str(plan_dir))
    assert launched.returncode == 0, launched.stderr

    step_manifest = yaml.safe_load((tmp_path / "steps" / recipe_payload["step"]["id"] / "step.yaml").read_text())
    assert set(step_manifest) == {"step", "experiment_id", "recipe_path", "plans"}
    assert step_manifest["step"]["inputs"] == ["config.yaml"]
    assert step_manifest["step"]["outputs"] == ["reports/ranking.csv"]
    assert step_manifest["experiment_id"] == recipe_payload["experiment"]["id"]
    assert step_manifest["recipe_path"]
    assert step_manifest["plans"] == [str(plan_dir.resolve())]
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert sum(event["event_type"] == "step_registered" for event in events) == 1


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
    planned = _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir))
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
    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir))
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


def test_plan_blocks_missing_experiment_metadata(tmp_path: Path):
    recipe = write_finetune_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    payload.pop("experiment")
    payload.pop("step")
    payload["_allow_unmanaged_context"] = True
    recipe.write_text(yaml.safe_dump(payload))

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(tmp_path / "plan"))

    assert result.returncode == 2
    assert "experiment" in result.stdout
    assert not (tmp_path / "plan").exists()


def test_plan_rejects_output_outside_experiment_root(tmp_path: Path):
    recipe = write_finetune_recipe(tmp_path / "experiment")

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(tmp_path / "outside"))

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

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(unmanaged_root / "plan"))

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
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    run = json.loads((plan_dir / "plan.json").read_text())["runs"][0]
    Path(run["script"]).write_text("#!/usr/bin/env bash\nexit 0\n")

    result = _run("hparam-launch", "--plan-dir", str(plan_dir))

    assert result.returncode == 1
    assert "hash" in result.stderr.lower()


def test_stop_requires_and_records_reason(tmp_path: Path, monkeypatch):
    recipe = _hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    monkeypatch.setattr(hparam_runtime, "_start_process", lambda _execution, _command: "launched")
    hparam.launch_hparam_runs(plan_dir, dry_run=False)
    row = list(csv.DictReader((plan_dir / "launch_manifest.tsv").open(), delimiter="\t"))[0]
    pid_path = Path(row["pid_path"])
    pid_path.write_text("123")
    monkeypatch.setattr(hparam_runtime.os, "kill", lambda _pid, _signal: None)

    with pytest.raises(ValueError, match="reason"):
        hparam.stop_hparam_run(plan_dir, "run-000", reason="")
    status_path = hparam.stop_hparam_run(plan_dir, "run-000", reason="validation diverged")

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
    report = (tmp_path / "reports" / "run_matrix.md").read_text()
    assert "prepare-data / run-000" in report
    assert "train-model / run-000" in report


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
    assert not (tmp_path / "run_matrix.csv").exists()


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


def test_run_evidence_key_uses_version_only_without_managed_identity():
    assert run_evidence_key({"step_id": "step-a", "run_id": "run-000", "version": "shared"}) == (
        "managed",
        "step-a",
        "run-000",
    )
    assert run_evidence_key({"run_id": "legacy", "version": "shared"}) == ("external", "shared")


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


def test_external_evidence_requires_the_managed_row_to_declare_the_same_experiment():
    rows = [{"step_id": "train", "run_id": "run-000", "version": "managed-v1"}]

    assert (
        resolve_external_run_row(
            rows,
            {
                "experiment_id": "foreign",
                "step_id": "train",
                "run_id": "run-000",
                "version": "managed-v1",
            },
        )
        is None
    )


def test_managed_rows_reject_historical_identity():
    with pytest.raises(ValueError, match="read-only"):
        validate_managed_run_rows(
            [{"step_id": "train", "trial_id": "trial_000", "status": "running"}],
            source="legacy.tsv",
            cardinality="one_per_run",
        )


def test_managed_rows_reject_duplicate_identity():
    row = {"step_id": "train", "run_id": "run-000"}

    with pytest.raises(ValueError, match="Duplicate managed run identity"):
        validate_managed_run_rows([row, dict(row)], source="run_manifest.tsv", cardinality="one_per_run")


def test_managed_rows_require_explicit_cardinality_and_allow_many_rows_per_run():
    row = {"step_id": "train", "run_id": "run-000"}

    validate_managed_run_rows([row, dict(row)], source="checkpoint_manifest.tsv", cardinality="many_per_run")
    with pytest.raises(ValueError, match="Unsupported managed row cardinality"):
        validate_managed_run_rows([row], source="run_manifest.tsv", cardinality="unknown")


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
        ("unknown_remote", {"status": "planned"}, "unknown_remote"),
        ("missing_pid", {"status": "pending"}, "missing_pid"),
        ("running", {"status": "failed"}, "failed"),
        ("planned", {"status": "superseded"}, "superseded"),
        ("pending", {"status": "superseded"}, "superseded"),
        ("launched", {"status": "superseded"}, "launched"),
        ("running", {"status": "superseded"}, "running"),
        ("unknown_remote", {"status": "superseded"}, "unknown_remote"),
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


@pytest.mark.parametrize("target_name", ["run_matrix.csv", "reports/run_matrix.md"])
@pytest.mark.parametrize("target_kind", ["directory", "hardlink"])
def test_merge_run_manifest_rejects_invalid_derived_targets_before_canonical_commit(
    tmp_path: Path, target_name: str, target_kind: str
):
    (tmp_path / "experiment.yaml").write_text("experiment:\n  id: unit\n")
    initialize_run_manifest(tmp_path)
    identity = {"experiment_id": "unit", "step_id": "train", "run_id": "run-000"}
    merge_run_manifest(tmp_path, [{**identity, "status": "planned"}])
    manifest_path = tmp_path / "run_manifest.tsv"
    target = tmp_path / target_name
    target.unlink()
    if target_kind == "directory":
        target.mkdir()
    else:
        target.hardlink_to(manifest_path)
    before = manifest_path.read_bytes()

    with pytest.raises(ValueError, match="Managed output"):
        merge_run_manifest(tmp_path, [{**identity, "status": "running"}])

    assert manifest_path.read_bytes() == before


def test_append_event_rejects_canonical_manifest_alias(tmp_path: Path):
    initialize_run_manifest(tmp_path)
    manifest_path = tmp_path / "run_manifest.tsv"
    events_path = tmp_path / "events.jsonl"
    events_path.hardlink_to(manifest_path)
    before = manifest_path.read_bytes()

    with pytest.raises(ValueError, match="Managed output"):
        append_event(tmp_path, "run_status_changed", {"run_id": "run-000"})

    assert manifest_path.read_bytes() == before


def test_merge_run_manifest_remote_commits_and_renders_the_same_rows(monkeypatch):
    existing = [{"experiment_id": "unit", "step_id": "train", "run_id": "run-000", "status": "failed"}]
    reads = []
    writes = {}

    def fake_read(path, *, remote=None):
        reads.append((Path(path).name, remote))
        return "experiment_id\tstep_id\trun_id\tstatus\nunit\ttrain\trun-000\tfailed\n"

    def fake_write_rows(path, rows, *, remote=None):
        writes[Path(path).name] = ([dict(row) for row in rows], remote)

    def fake_write_text(path, text, *, remote=None):
        writes[Path(path).name] = (text, remote)

    def fake_commit(path, text, _expected_sha256, *, remote=None):
        writes[Path(path).name] = (text, remote)
        return True

    monkeypatch.setattr(experiment_io, "path_exists_at", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(experiment_io, "validate_managed_output_paths", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(experiment_io, "read_text_at", fake_read)
    monkeypatch.setattr(experiment_io, "write_rows_at", fake_write_rows)
    monkeypatch.setattr(experiment_io, "write_text_at", fake_write_text)
    monkeypatch.setattr(experiment_io, "conditional_atomic_replace_text_at", fake_commit)

    committed = merge_run_manifest(
        "/remote/workspace",
        [{"step_id": "train", "run_id": "run-000", "status": "completed"}],
        remote="baichuan3",
    )

    assert committed == existing
    assert reads == [("run_manifest.tsv", "baichuan3")]
    assert "unit\trun-000\tfailed\ttrain" in writes["run_manifest.tsv"][0]
    assert writes["run_manifest.tsv"][1] == "baichuan3"
    assert writes["run_matrix.csv"] == (existing, "baichuan3")
    assert "| failed |" in writes["run_matrix.md"][0]
    assert writes["run_matrix.md"][1] == "baichuan3"


def test_merge_run_manifest_remote_read_failure_writes_nothing(monkeypatch):
    writes = []
    monkeypatch.setattr(experiment_io, "path_exists_at", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(experiment_io, "validate_managed_output_paths", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        experiment_io,
        "read_text_at",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("SSH read failed")),
    )
    monkeypatch.setattr(experiment_io, "write_rows_at", lambda *args, **kwargs: writes.append((args, kwargs)))
    monkeypatch.setattr(experiment_io, "write_text_at", lambda *args, **kwargs: writes.append((args, kwargs)))

    with pytest.raises(RuntimeError, match="SSH read failed"):
        merge_run_manifest(
            "/remote/workspace",
            [{"step_id": "train", "run_id": "run-000", "status": "running"}],
            remote="baichuan3",
        )

    assert writes == []


def test_merge_run_manifest_remote_new_key_checks_workspace_owner_before_writing(monkeypatch):
    reads = []
    writes = []

    def fake_read(path, *, remote=None):
        reads.append((Path(path).name, remote))
        if Path(path).name == "experiment.yaml":
            return "experiment:\n  id: unit\n"
        return "step_id\trun_id\n"

    monkeypatch.setattr(experiment_io, "path_exists_at", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(experiment_io, "validate_managed_output_paths", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(experiment_io, "read_text_at", fake_read)
    monkeypatch.setattr(experiment_io, "write_rows_at", lambda *args, **kwargs: writes.append((args, kwargs)))
    monkeypatch.setattr(experiment_io, "write_text_at", lambda *args, **kwargs: writes.append((args, kwargs)))

    with pytest.raises(ValueError, match="different experiment"):
        merge_run_manifest(
            "/remote/workspace",
            [{"experiment_id": "foreign", "step_id": "train", "run_id": "run-000"}],
            remote="baichuan3",
        )

    assert reads == [("run_manifest.tsv", "baichuan3"), ("experiment.yaml", "baichuan3")]
    assert writes == []


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


@pytest.mark.parametrize("alias_kind", ["symlink", "hardlink"])
def test_read_run_manifest_rejects_aliased_canonical_table(tmp_path: Path, alias_kind: str):
    outside = tmp_path / "outside.tsv"
    outside.write_text("experiment_id\tstep_id\trun_id\tstatus\nunit\ttrain-model\trun-000\tfailed\n")
    manifest = tmp_path / "run_manifest.tsv"
    if alias_kind == "symlink":
        manifest.symlink_to(outside)
    else:
        manifest.hardlink_to(outside)

    with pytest.raises(ValueError, match="Managed output paths must be independent regular files"):
        read_run_manifest(tmp_path)


def test_read_step_manifest_rejects_invalid_phase(tmp_path: Path):
    path = tmp_path / "steps" / "train" / "step.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(
        "step:\n"
        "  id: train\n"
        "  phase: invalid\n"
        "  purpose: Train the model.\n"
        "experiment_id: unit\n"
        "recipe_path: ''\n"
        "plans: []\n"
    )

    with pytest.raises(ValueError, match="step.phase"):
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
    with pytest.raises(FileNotFoundError, match="run_manifest.tsv"):
        merge_run_manifest(tmp_path, [{"step_id": "train", "run_id": "run-000", "status": "planned"}])

    assert not (tmp_path / "run_manifest.tsv").exists()


def test_empty_canonical_commit_preserves_the_valid_identity_header(tmp_path: Path):
    initialize_run_manifest(tmp_path)

    assert merge_run_manifest(tmp_path, []) == []

    assert (tmp_path / "run_manifest.tsv").read_text() == "step_id\trun_id\n"
    assert (tmp_path / "run_matrix.csv").read_text() == "step_id,run_id\n"
    assert read_run_manifest(tmp_path) == []


def test_empty_remote_canonical_commit_preserves_the_valid_matrix_identity_header(monkeypatch):
    writes = {}

    monkeypatch.setattr(experiment_io, "path_exists_at", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(experiment_io, "validate_managed_output_paths", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(experiment_io, "read_text_at", lambda *_args, **_kwargs: "step_id\trun_id\n")
    monkeypatch.setattr(
        experiment_io,
        "write_rows_at",
        lambda path, rows, *, remote=None: writes.update({Path(path).name: (rows, remote)}),
    )
    monkeypatch.setattr(
        experiment_io,
        "write_text_at",
        lambda path, text, *, remote=None: writes.update({Path(path).name: (text, remote)}),
    )
    monkeypatch.setattr(
        experiment_io,
        "conditional_atomic_replace_text_at",
        lambda path, text, _expected_sha256, *, remote=None: writes.update({Path(path).name: (text, remote)}) is None,
    )

    assert merge_run_manifest("/remote/workspace", [], remote="unit-host") == []

    assert writes["run_manifest.tsv"] == ("step_id\trun_id\n", "unit-host")
    assert writes["run_matrix.csv"] == ("step_id,run_id\n", "unit-host")


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

    assert [process.wait(timeout=10) for process in processes] == [0, 0]
    assert {row["run_id"] for row in read_run_manifest(tmp_path)} == {"run-000", "run-001"}


def test_concurrent_local_manifest_writer_holds_lock_through_projection(tmp_path: Path, monkeypatch):
    (tmp_path / "experiment.yaml").write_text("experiment:\n  id: unit\n")
    initialize_run_manifest(tmp_path)
    first_projection_started = threading.Event()
    finish_first_projection = threading.Event()
    first_projection_done = threading.Event()
    second_lock_attempted = threading.Event()
    local_lock = threading.Lock()
    lock_violations = []
    real_flock = fcntl.flock
    real_write_run_matrix = experiment_workspace.write_run_matrix

    def tracked_flock(_fd, operation):
        if operation == fcntl.LOCK_EX:
            if threading.current_thread().name == "second-merge":
                if local_lock.acquire(blocking=False):
                    if not first_projection_done.is_set():
                        lock_violations.append("second writer acquired the lock before the first projection completed")
                    second_lock_attempted.set()
                    return
                second_lock_attempted.set()
            local_lock.acquire()
        elif operation == fcntl.LOCK_UN:
            local_lock.release()
        else:
            real_flock(_fd, operation)

    def delayed_write_run_matrix(root, rows, *, remote=None):
        if {row["run_id"] for row in rows} == {"run-000"}:
            first_projection_started.set()
            assert finish_first_projection.wait(timeout=5)
            result = real_write_run_matrix(root, rows, remote=remote)
            first_projection_done.set()
            return result
        return real_write_run_matrix(root, rows, remote=remote)

    monkeypatch.setattr(experiment_workspace.fcntl, "flock", tracked_flock)
    monkeypatch.setattr(experiment_workspace, "write_run_matrix", delayed_write_run_matrix)
    errors = []

    def merge(run_id):
        try:
            merge_run_manifest(
                tmp_path,
                [{"experiment_id": "unit", "step_id": "train", "run_id": run_id, "status": "planned"}],
            )
        except Exception as exc:
            errors.append(exc)

    first = threading.Thread(target=merge, args=("run-000",), name="first-merge")
    second = threading.Thread(target=merge, args=("run-001",), name="second-merge")
    first.start()
    assert first_projection_started.wait(timeout=5)
    second.start()
    assert second_lock_attempted.wait(timeout=5)
    finish_first_projection.set()
    first.join(timeout=5)
    second.join(timeout=5)

    assert not first.is_alive() and not second.is_alive()
    assert errors == []
    assert lock_violations == []
    with (tmp_path / "run_matrix.csv").open(newline="") as file_obj:
        matrix = list(csv.DictReader(file_obj))
    assert {row["run_id"] for row in matrix} == {"run-000", "run-001"}


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

    assert [process.wait(timeout=10) for process in processes] == [0, 0]
    assert read_run_manifest(tmp_path)[0]["status"] == "failed"


def test_interrupted_atomic_replace_preserves_the_complete_old_manifest(tmp_path: Path, monkeypatch):
    (tmp_path / "experiment.yaml").write_text("experiment:\n  id: unit\n")
    initialize_run_manifest(tmp_path)
    merge_run_manifest(
        tmp_path,
        [{"experiment_id": "unit", "step_id": "train", "run_id": "run-000", "status": "planned"}],
    )
    before = (tmp_path / "run_manifest.tsv").read_bytes()
    monkeypatch.setattr(experiment_io.os, "replace", lambda *_args: (_ for _ in ()).throw(OSError("interrupted")))

    with pytest.raises(OSError, match="interrupted"):
        merge_run_manifest(tmp_path, [{"step_id": "train", "run_id": "run-000", "status": "running"}])

    assert (tmp_path / "run_manifest.tsv").read_bytes() == before
    assert read_run_manifest(tmp_path)[0]["status"] == "planned"


def test_projection_failure_does_not_roll_back_canonical_commit(tmp_path: Path, monkeypatch):
    (tmp_path / "experiment.yaml").write_text("experiment:\n  id: unit\n")
    initialize_run_manifest(tmp_path)
    real_write_run_matrix = experiment_workspace.write_run_matrix
    monkeypatch.setattr(
        experiment_workspace,
        "write_run_matrix",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("projection failed")),
    )

    with pytest.raises(OSError, match="projection failed"):
        merge_run_manifest(
            tmp_path,
            [{"experiment_id": "unit", "step_id": "train", "run_id": "run-000", "status": "running"}],
        )

    assert read_run_manifest(tmp_path)[0]["status"] == "running"
    monkeypatch.setattr(experiment_workspace, "write_run_matrix", real_write_run_matrix)
    merge_run_manifest(tmp_path, [])
    assert "running" in (tmp_path / "run_matrix.csv").read_text()


def test_remote_manifest_commit_retries_after_digest_conflict_without_losing_rows(monkeypatch):
    state = {"text": "step_id\trun_id\n", "attempts": 0}

    def fake_read(path, *, remote=None):
        if Path(path).name == "experiment.yaml":
            return "experiment:\n  id: unit\n"
        return state["text"]

    def fake_commit(_path, text, _expected_sha256, *, remote=None):
        state["attempts"] += 1
        if state["attempts"] == 1:
            state["text"] = "experiment_id\tstatus\tstep_id\trun_id\n" "unit\tplanned\ttrain\trun-001\n"
            return False
        state["text"] = text
        return True

    monkeypatch.setattr(experiment_io, "path_exists_at", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(experiment_io, "validate_managed_output_paths", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(experiment_io, "read_text_at", fake_read)
    monkeypatch.setattr(experiment_io, "conditional_atomic_replace_text_at", fake_commit)
    monkeypatch.setattr(experiment_io, "write_rows_at", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(experiment_io, "write_text_at", lambda *_args, **_kwargs: None)

    committed = merge_run_manifest(
        "/remote/workspace",
        [{"experiment_id": "unit", "step_id": "train", "run_id": "run-000", "status": "planned"}],
        remote="unit-host",
    )

    assert state["attempts"] == 2
    assert {row["run_id"] for row in committed} == {"run-000", "run-001"}
    assert "run-000" in state["text"] and "run-001" in state["text"]


def test_remote_manifest_commit_fails_after_three_digest_conflicts(monkeypatch):
    attempts = []
    monkeypatch.setattr(experiment_io, "path_exists_at", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(experiment_io, "validate_managed_output_paths", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(experiment_io, "read_text_at", lambda *_args, **_kwargs: "step_id\trun_id\n")
    monkeypatch.setattr(
        experiment_io,
        "conditional_atomic_replace_text_at",
        lambda *_args, **_kwargs: attempts.append(True) and False,
    )

    with pytest.raises(RuntimeError, match="three commit attempts"):
        merge_run_manifest("/remote/workspace", [], remote="unit-host")

    assert len(attempts) == 3


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


def test_step_manifest_producers_use_the_workspace_reader_and_merger():
    agent_tools_dir = Path(__file__).parents[2] / "agent_tools"
    producers = {}
    writer_names = {"write_text", "write_text_at"}
    for path in agent_tools_dir.glob("*.py"):
        tree = ast.parse(path.read_text())
        for function in (node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)):
            locator_names = set()
            for node in ast.walk(function):
                if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                    continue
                if not any(
                    isinstance(part, ast.Constant) and part.value == "step.yaml" for part in ast.walk(node.value)
                ):
                    continue
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                locator_names.update(
                    part.id for target in targets for part in ast.walk(target) if isinstance(part, ast.Name)
                )
            calls = {
                node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
                for node in ast.walk(function)
                if isinstance(node, ast.Call)
            }
            writes_step = any(
                isinstance(node, ast.Call)
                and (node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", ""))
                in writer_names
                and (
                    any(isinstance(part, ast.Constant) and part.value == "step.yaml" for part in ast.walk(node))
                    or any(isinstance(part, ast.Name) and part.id in locator_names for part in ast.walk(node))
                )
                for node in ast.walk(function)
            )
            if writes_step:
                producers[(path.name, function.name)] = calls

    assert set(producers) == {
        ("experiment_workspace.py", "ensure_experiment_workspace"),
        ("experiments.py", "register_experiment_step"),
    }
    assert all({"read_step_manifest", "merge_step_manifest"} <= calls for calls in producers.values())


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


def test_remote_output_validation_checks_root_itself_before_targets(monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(
            command,
            2,
            "",
            "Managed output paths must be independent regular files: /remote/workspace",
        )

    monkeypatch.setattr(experiment_io.subprocess, "run", fake_run)

    with pytest.raises(ValueError, match="/remote/workspace"):
        experiment_io.validate_managed_output_paths(
            "/remote/workspace",
            ["/remote/workspace/experiment.yaml"],
            remote="unit-host",
        )

    command, kwargs = calls[0]
    assert command[:2] == ["ssh", "unit-host"]
    assert command[-1].index("os.lstat(root)") < command[-1].index("for raw_target in targets")
    assert kwargs["timeout"] == experiment_io.SSH_TIMEOUT_SECONDS


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
        "host",
        "workdir",
        "gpus",
        "pid_path",
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
    initial = {**identity, field: original, "status": "planned"}
    if field in EXECUTION_IDENTITY_FIELDS and field != "target":
        initial["target"] = "local"
    merge_run_manifest(tmp_path, [initial])
    before = (tmp_path / "run_manifest.tsv").read_bytes()

    with pytest.raises(ValueError, match=field.replace(".", r"\.")):
        merge_run_manifest(tmp_path, [{**identity, field: changed, "status": "running"}])

    assert (tmp_path / "run_manifest.tsv").read_bytes() == before


def test_frozen_validator_only_allows_trusted_execution_identity_initialization():
    existing = {"step_id": "train", "run_id": "run-000", "status": "planned"}
    incoming = {"step_id": "train", "run_id": "run-000", "target": "ssh", "host": "foreign-host"}

    with pytest.raises(ValueError, match="execution identity"):
        validate_frozen_run_update(existing, incoming)

    validate_frozen_run_update(existing, incoming, allow_execution_identity_fill=True)


def test_semantic_run_name_keeps_boolean_settings_readable():
    assert (
        semantic_run_name({"runtime.lr": 2e-6, "yaml:/model/router_frozen": True, "yaml:/loss/class_weights": False})
        == "lr-2e-6__router-frozen__class-weights-off"
    )


def test_step_manifest_merge_preserves_registered_fields_and_appends_plans():
    existing = {
        "step": {
            "id": "train",
            "phase": "train",
            "purpose": "Tune the model.",
            "inputs": ["data.csv"],
            "outputs": ["ranking.csv"],
        },
        "experiment_id": "experiment",
        "recipe_path": "recipes/first.yaml",
        "plans": ["/workspace/plan-a"],
    }

    merged = merge_step_manifest(
        existing,
        {
            "step": {"id": "train", "phase": "train", "purpose": "Tune the model."},
            "experiment_id": "experiment",
            "recipe_path": "recipes/second.yaml",
            "plans": ["/workspace/plan-a", "/workspace/plan-b"],
        },
    )

    assert merged["step"]["inputs"] == ["data.csv"]
    assert merged["step"]["outputs"] == ["ranking.csv"]
    assert merged["recipe_path"] == "recipes/first.yaml"
    assert merged["plans"] == ["/workspace/plan-a", "/workspace/plan-b"]


def test_step_manifest_merge_rejects_metadata_drift():
    existing = {
        "step": {"id": "train", "phase": "train", "purpose": "Tune the model."},
        "experiment_id": "experiment",
        "recipe_path": "",
        "plans": [],
    }

    with pytest.raises(ValueError, match="phase"):
        merge_step_manifest(existing, {"step": {"phase": "analyze"}})


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
    ],
)
def test_planner_rejects_corrupt_existing_step_manifest_without_writing(tmp_path: Path, existing: str):
    recipe = _hparam_recipe(tmp_path)
    first = tmp_path / "plans" / "first"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(first)).returncode == 0
    payload = json.loads((first / "plan.json").read_text())["recipe"]
    target = tmp_path / "steps" / payload["step"]["id"] / "step.yaml"
    target.write_text(existing)
    before = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    plan_dir = tmp_path / "plans" / "corrupt-step"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir))

    assert result.returncode == 1
    assert "step manifest" in result.stderr
    assert {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before


def test_new_plan_continues_run_ids_within_the_same_step(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path)
    first = tmp_path / "plans" / "first"
    second = tmp_path / "plans" / "second"

    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(first)).returncode == 0
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(second)).returncode == 0

    first_run = json.loads((first / "plan.json").read_text())["runs"][0]
    second_run = json.loads((second / "plan.json").read_text())["runs"][0]
    assert first_run["run_id"] == "run-000"
    assert second_run["run_id"] == "run-001"


def test_missing_canonical_manifest_never_resets_run_identity(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path)
    first = tmp_path / "plans" / "first"
    second = tmp_path / "plans" / "second"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(first)).returncode == 0
    (tmp_path / "run_manifest.tsv").unlink()
    before = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(second))

    assert result.returncode == 1
    assert "run_manifest.tsv" in result.stderr
    assert {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before
    assert not second.exists()


def test_planner_rejects_duplicate_workspace_ownership_without_writing(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path)
    first = tmp_path / "plans" / "first"
    second = tmp_path / "plans" / "second"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(first)).returncode == 0
    manifest = tmp_path / "experiment.yaml"
    manifest.write_text(
        manifest.read_text().replace("  id: unit-experiment\n", "  id: foreign\n  id: unit-experiment\n")
    )
    before = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(second))

    assert result.returncode == 1
    assert "duplicate key" in result.stderr
    assert {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before
    assert not second.exists()


def test_completed_experiment_rejects_new_plan(tmp_path: Path):
    recipe = write_finetune_recipe(tmp_path)
    first = tmp_path / "plans" / "first"
    second = tmp_path / "plans" / "second"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(first)).returncode == 0
    (tmp_path / "run_manifest.tsv").write_text(
        "experiment_id\tstep_id\trun_id\tstatus\nunit-experiment\tunit-finetune\trun-000\tfinished\n"
    )
    report = tmp_path / "final_source.md"
    report.write_text("# Final\n")
    assert _run("experiment-finalize", "--run-dir", str(tmp_path), "--report", str(report)).returncode == 0

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(second))

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

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir))

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

    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(first)).returncode == 0
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(second)).returncode == 0

    first_run = json.loads((first / "runs" / "run-000--unit" / "run.json").read_text())
    second_run = json.loads((second / "runs" / "run-001--unit" / "run.json").read_text())
    assert first_run["run_name"] == second_run["run_name"] == "unit"
    assert first_run["version"] != second_run["version"]
    assert "run-000" in first_run["version"]
    assert "run-001" in second_run["version"]
