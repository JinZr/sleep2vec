from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from agent_tool_test_helpers import write_finetune_recipe, write_yaml
from agent_tools import cli, experiment_io, experiment_tracking, experiments, plans
from agent_tools.manifests import write_rows
from agent_tools.models import REPO_ROOT


def _init_workspace(root: Path) -> None:
    root.mkdir()
    experiment = {
        "id": "status-unit",
        "title": "Status unit experiment",
        "objective": "Exercise the read-only status contract.",
        "root": str(root),
        "baseline": "unit baseline",
    }
    (root / "experiment.yaml").write_text(yaml.safe_dump({"experiment": experiment}, sort_keys=False))
    (root / "run_manifest.tsv").write_text("step_id\trun_id\n")


def _add_plan(
    root: Path,
    *,
    step_id: str,
    status: str = "planned",
    task: str = "finetune",
    adaptive: bool = False,
    pipeline: bool = False,
    host: str | None = None,
) -> tuple[Path, dict[str, str]]:
    experiment = yaml.safe_load((root / "experiment.yaml").read_text())["experiment"]
    step = {"id": step_id, "phase": "train", "purpose": f"Run {step_id}."}
    plan_dir = root / "plans" / step_id
    run_dir = plan_dir / "runs" / "run-000--default"
    run_dir.mkdir(parents=True)
    config_path = run_dir / "config.yaml"
    script_path = run_dir / "launch.sh"
    artifacts_path = run_dir / "artifacts.json"
    config_path.write_text("model: unit\n")
    script_path.write_text("#!/bin/sh\nexit 0\n")
    artifacts_path.write_text("{}\n")
    config_sha256 = _sha256(config_path)
    script_sha256 = _sha256(script_path)
    run = {
        "experiment_id": experiment["id"],
        "step_id": step_id,
        "run_id": "run-000",
        "run_name": "default",
        "version": f"status-unit-{step_id}-run-000-default",
        "status": "planned",
        "config": str(config_path),
        "config_sha256": config_sha256,
        "script": str(script_path),
        "script_sha256": script_sha256,
        "run_dir": str(run_dir),
        "artifacts": str(artifacts_path),
        "runtime_dir": str(root / "runtime" / step_id),
        "checkpoint_dir": str(root / "runtime" / step_id / "checkpoints"),
    }
    if task == "hparam_tune":
        run["parameter_summary"] = "single resolved recipe"
    if host is not None:
        run["host"] = host
    recipe = {"task": task, "experiment": experiment, "step": step}
    if adaptive:
        recipe["adaptive"] = {"enabled": True}
    resolved_text = yaml.safe_dump(recipe, sort_keys=False)
    resolved_path = plan_dir / "recipe.resolved.yaml"
    resolved_path.write_text(resolved_text)
    plan = {"status": "PASS", "recipe": recipe, "runs": [run]}
    if task == "hparam_tune":
        plan["resolved_recipe_sha256"] = _sha256(resolved_path)
        (plan_dir / "run_all.sh").write_text("#!/bin/sh\nexit 0\n")
    else:
        (plan_dir / "run.sh").write_text(script_path.read_text())
    (plan_dir / "plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")

    step_dir = root / "steps" / step_id
    step_dir.mkdir(parents=True)
    (step_dir / "step.yaml").write_text(
        yaml.safe_dump(
            {
                "step": step,
                "experiment_id": experiment["id"],
                "recipe_path": "",
                "plans": [str(plan_dir)],
            },
            sort_keys=False,
        )
    )
    canonical = {
        **run,
        "status": status,
        "parameter_summary": run.get("parameter_summary", "single resolved recipe"),
    }
    if pipeline:
        canonical["pipeline_id"] = "pipeline-unit"
    existing = _read_manifest_rows(root)
    write_rows(root / "run_manifest.tsv", [*existing, canonical])
    return plan_dir, canonical


def _read_manifest_rows(root: Path) -> list[dict[str, str]]:
    text = (root / "run_manifest.tsv").read_text().splitlines()
    if len(text) <= 1:
        return []
    import csv

    return list(csv.DictReader(text, delimiter="\t"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _workspace_files(root: Path) -> dict[str, bytes]:
    return {str(path.relative_to(root)): path.read_bytes() for path in sorted(root.rglob("*")) if path.is_file()}


def _write_public_hparam_recipe(root: Path, parameters: dict) -> Path:
    base_recipe = write_finetune_recipe(root)
    return write_yaml(
        root / "tune.yaml",
        {
            "name": "status_public_hparam",
            "task": "hparam_tune",
            "variant": "sleep2vec",
            "base_recipe": str(base_recipe),
            "inputs": {},
            "search": {"method": "grid", "max_runs": 1, "parameters": parameters},
            "evaluation_policy": {
                "selection_metric": "val_ahi_pearson",
                "selection_mode": "max",
                "selection_split": "val",
                "final_eval_split": "test",
                "external_test_locked": True,
                "test_after_fit": False,
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


def test_experiment_status_accepts_public_generic_plan_without_runtime_directories(tmp_path):
    root = tmp_path / "experiment"
    payload = yaml.safe_load((REPO_ROOT / "recipes/examples/tiny_fixture_sleep2stat.yaml").read_text())
    recipe = write_yaml(root / "sleep2stat.yaml", payload)
    plan_dir = root / "plans" / "sleep2stat"

    assert plans.build_plan(recipe_path=recipe, output_dir=plan_dir).exit_code == 0
    planned = json.loads((plan_dir / "plan.json").read_text())["runs"][0]
    assert planned["runtime_dir"] == planned["checkpoint_dir"] == ""

    snapshot = experiments.experiment_status(root)

    assert snapshot["summary"]["state"] == "ready_to_launch"
    assert snapshot["decision"]["recommended_next"]["argv"] == ["bash", str(plan_dir / "run.sh")]


@pytest.mark.parametrize("parameters", [{"runtime.lr": [1e-6]}, {"yaml:/data/finetune_preset_path": [None]}])
def test_experiment_status_accepts_public_layered_hparam_plan(tmp_path, parameters):
    root = tmp_path / "experiment"
    recipe = _write_public_hparam_recipe(root, parameters)
    plan_dir = root / "plans" / "tune"

    assert plans.build_plan(recipe_path=recipe, output_dir=plan_dir).exit_code == 0
    plan = json.loads((plan_dir / "plan.json").read_text())
    resolved = yaml.safe_load((plan_dir / "recipe.resolved.yaml").read_text())
    assert "_base_recipe" in plan["recipe"] and "_local_recipe" in resolved
    parameter = next(iter(parameters))
    if parameters[parameter] == [None]:
        assert plan["runs"][0][parameter] is None
        assert _read_manifest_rows(root)[0][parameter] == ""

    snapshot = experiments.experiment_status(root)

    assert snapshot["summary"]["state"] == "ready_to_launch"


@pytest.mark.parametrize("drift", ["partial_runtime", "hparam_parameter_summary", "input_snapshots"])
def test_experiment_status_rejects_incomplete_registered_plan_identity(tmp_path, drift):
    root = tmp_path / "experiment"
    _init_workspace(root)
    task = "hparam_tune" if drift == "hparam_parameter_summary" else "sleep2stat"
    plan_dir, canonical = _add_plan(root, step_id="train", task=task)
    plan = json.loads((plan_dir / "plan.json").read_text())
    if drift == "partial_runtime":
        plan["runs"][0]["runtime_dir"] = ""
        canonical["runtime_dir"] = ""
    elif drift == "hparam_parameter_summary":
        del plan["runs"][0]["parameter_summary"]
    else:
        canonical["input_snapshots"] = [{"field": "inputs.config", "path": canonical["config"], "sha256": "0" * 64}]
    (plan_dir / "plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    write_rows(root / "run_manifest.tsv", [canonical])

    with pytest.raises(ValueError):
        experiments.experiment_status(root)


def test_experiment_status_snapshot_is_deterministic_and_keeps_recorded_evidence(tmp_path):
    root = tmp_path / "experiment"
    _init_workspace(root)
    _plan, canonical = _add_plan(root, step_id="train", status="unknown_scheduler", task="hparam_tune")
    canonical.update(
        {
            "scheduler_raw_state": "MISSING",
            "scheduler_reason": "Scheduler identity is incomplete.",
            "scheduler_observed_at": "2026-08-25T01:02:03Z",
            "checkpoint_count": "50",
            "run_manifest": str(root / "runtime" / "train" / "run_manifest.json"),
        }
    )
    write_rows(root / "run_manifest.tsv", [canonical])

    first = experiments.experiment_status(root)
    second = experiments.experiment_status(root)

    assert first == second
    assert set(first) == {
        "schema_version",
        "experiment",
        "lifecycle_source",
        "live_observation",
        "summary",
        "steps",
        "runs",
        "blockers",
        "decision",
    }
    assert "generated_at" not in first
    assert first["summary"] == {"state": "blocked", "run_count": 1, "status_counts": {"unknown_scheduler": 1}}
    assert first["runs"][0]["scheduler"]["observed_at"] == "2026-08-25T01:02:03Z"
    assert first["runs"][0]["process"]["pid"] is None
    assert first["runs"][0]["evidence"]["checkpoint_count"] == "50"
    assert first["decision"]["recommended_next"]["id"] == "experiment-monitor"
    assert first["decision"]["blocked_actions"] == ["adaptive_advance", "finalize", "resubmit"]


@pytest.mark.parametrize("status", ["unknown_scheduler", "unknown_remote", "missing_pid", "submitting"])
def test_experiment_status_uncertain_states_recommend_only_monitor(tmp_path, status):
    root = tmp_path / "experiment"
    _init_workspace(root)
    _add_plan(root, step_id="train", status=status)

    snapshot = experiments.experiment_status(root)

    assert snapshot["summary"]["state"] == "blocked"
    assert snapshot["decision"]["recommended_next"]["id"] == "experiment-monitor"
    assert snapshot["decision"]["other_legal_actions"] == []
    assert snapshot["decision"]["blocked_actions"] == ["adaptive_advance", "finalize", "resubmit"]


@pytest.mark.parametrize("status", ["queued", "launched", "running", "stopping"])
def test_experiment_status_active_states_recommend_monitor(tmp_path, status):
    root = tmp_path / "experiment"
    _init_workspace(root)
    _add_plan(root, step_id="train", status=status)

    snapshot = experiments.experiment_status(root)

    assert snapshot["summary"]["state"] == "in_progress"
    assert snapshot["decision"]["recommended_next"]["id"] == "experiment-monitor"
    assert "finalize" in snapshot["decision"]["blocked_actions"]
    assert "launch" in snapshot["decision"]["blocked_actions"]


def test_experiment_status_empty_workspace_does_not_guess_a_command(tmp_path):
    root = tmp_path / "experiment"
    _init_workspace(root)

    snapshot = experiments.experiment_status(root)

    assert snapshot["summary"] == {"state": "empty", "run_count": 0, "status_counts": {}}
    assert snapshot["blockers"][0]["code"] == "no_managed_runs"
    assert snapshot["decision"]["manual_choice_required"] is True
    assert snapshot["decision"]["recommended_next"] is None


def test_experiment_status_launch_actions_and_manual_choice(tmp_path):
    root = tmp_path / "experiment"
    _init_workspace(root)
    hparam_plan, _row = _add_plan(root, step_id="tune", task="hparam_tune")

    one = experiments.experiment_status(root)

    assert one["summary"]["state"] == "ready_to_launch"
    assert one["decision"]["recommended_next"]["argv"] == [
        "python",
        "-m",
        "agent_tools",
        "hparam-run-queue",
        "--plan-dir",
        str(hparam_plan),
        "--execute",
    ]

    generic_plan, _row = _add_plan(root, step_id="train")
    multiple = experiments.experiment_status(root)

    assert multiple["decision"]["manual_choice_required"] is True
    assert multiple["decision"]["recommended_next"] is None
    assert [action["id"] for action in multiple["decision"]["other_legal_actions"]] == [
        "run-plan",
        "hparam-run-queue",
    ]
    assert multiple["decision"]["other_legal_actions"][0]["argv"] == ["bash", str(generic_plan / "run.sh")]


@pytest.mark.parametrize(
    ("adaptive", "pipeline", "code"),
    [(True, False, "adaptive_phase_deferred"), (False, True, "pipeline_phase_deferred")],
)
def test_experiment_status_does_not_infer_adaptive_or_pipeline_launch(tmp_path, adaptive, pipeline, code):
    root = tmp_path / "experiment"
    _init_workspace(root)
    _add_plan(root, step_id="train", adaptive=adaptive, pipeline=pipeline)

    snapshot = experiments.experiment_status(root)

    assert snapshot["summary"]["state"] == "blocked"
    assert snapshot["decision"]["recommended_next"] is None
    assert snapshot["blockers"][0]["code"] == code


def test_experiment_status_scopes_deferred_plans_away_from_ordinary_launch(tmp_path):
    root = tmp_path / "experiment"
    _init_workspace(root)
    ordinary, _row = _add_plan(root, step_id="ordinary")
    _add_plan(root, step_id="adaptive", adaptive=True)
    _add_plan(root, step_id="pipeline", pipeline=True)

    snapshot = experiments.experiment_status(root)

    assert snapshot["summary"]["state"] == "ready_to_launch"
    assert snapshot["decision"]["recommended_next"]["argv"] == ["bash", str(ordinary / "run.sh")]
    assert snapshot["decision"]["blocked_actions"] == ["adaptive_advance", "pipeline_advance"]
    assert {(blocker["code"], blocker["step_id"]) for blocker in snapshot["blockers"]} == {
        ("adaptive_phase_deferred", "adaptive"),
        ("pipeline_phase_deferred", "pipeline"),
    }


def test_experiment_status_attributes_blockers_by_full_managed_run_key(tmp_path):
    root = tmp_path / "experiment"
    _init_workspace(root)
    _add_plan(root, step_id="first", status="unknown_scheduler")
    _add_plan(root, step_id="second", status="unknown_scheduler")
    _add_plan(root, step_id="terminal", status="failed")

    snapshot = experiments.experiment_status(root)
    blockers_by_step = {run["step_id"]: run["blockers"] for run in snapshot["runs"]}

    assert blockers_by_step["first"] == ["unknown_scheduler"]
    assert blockers_by_step["second"] == ["unknown_scheduler"]
    assert blockers_by_step["terminal"] == []


def test_experiment_status_defers_terminal_pipeline_finalization(tmp_path):
    root = tmp_path / "experiment"
    _init_workspace(root)
    _add_plan(root, step_id="evaluate", status="failed", pipeline=True)

    snapshot = experiments.experiment_status(root)

    assert snapshot["summary"]["state"] == "blocked"
    assert snapshot["decision"]["recommended_next"] is None
    assert snapshot["decision"]["other_legal_actions"] == []
    assert snapshot["decision"]["blocked_actions"] == ["finalize", "pipeline_advance"]
    assert snapshot["blockers"][0]["code"] == "pipeline_phase_deferred"

    manifest = yaml.safe_load((root / "experiment.yaml").read_text())
    manifest["experiment"].update({"status": "completed", "completed_at": "2026-08-25T02:00:00Z"})
    (root / "experiment.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))
    assert experiments.experiment_status(root)["summary"]["state"] == "completed"


def test_experiment_status_terminal_and_completed_contract(tmp_path):
    root = tmp_path / "experiment"
    _init_workspace(root)
    _add_plan(root, step_id="train", status="failed")

    ready = experiments.experiment_status(root)

    assert ready["summary"]["state"] == "ready_to_finalize"
    finalize = ready["decision"]["other_legal_actions"][0]
    assert finalize["id"] == "experiment-finalize"
    assert finalize["required_inputs"] == ["report_path"]

    manifest = yaml.safe_load((root / "experiment.yaml").read_text())
    manifest["experiment"].update({"status": "completed", "completed_at": "2026-08-25T02:00:00Z"})
    (root / "experiment.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))
    completed = experiments.experiment_status(root)
    assert completed["summary"]["state"] == "completed"
    assert completed["decision"]["recommended_next"] is None

    rows = _read_manifest_rows(root)
    rows[0]["status"] = "running"
    write_rows(root / "run_manifest.tsv", rows)
    with pytest.raises(ValueError, match="Completed experiment metadata conflicts"):
        experiments.experiment_status(root)


def test_experiment_status_is_zero_write_and_ignores_projections(tmp_path, monkeypatch):
    root = tmp_path / "experiment"
    _init_workspace(root)
    plan_dir, _row = _add_plan(root, step_id="train", status="unknown_scheduler")
    before = _workspace_files(root)

    def unexpected(*_args, **_kwargs):
        raise AssertionError("experiment-status attempted a write or live observation")

    for name in (
        "write_text_at",
        "write_rows_at",
        "conditional_atomic_replace_text_at",
        "append_event_at",
        "mkdir_experiment_dirs",
    ):
        monkeypatch.setattr(experiment_io, name, unexpected)
    monkeypatch.setattr(experiments, "merge_run_manifest", unexpected)
    monkeypatch.setattr(experiment_tracking, "monitor_run_row", unexpected)

    baseline = experiments.experiment_status(root)
    assert _workspace_files(root) == before

    (root / "run_status.tsv").write_text("not\ta\tvalid\tprojection\n")
    (plan_dir / "launch_manifest.tsv").write_text("status\ncompleted\n")
    (root / "reports").mkdir()
    (root / "reports" / "monitor.md").write_text("completed\n")
    (root / "events.jsonl").write_text("not-json\n")
    (root / "wandb").mkdir()
    (root / "wandb" / "runs.tsv").write_text("status\ncompleted\n")

    assert experiments.experiment_status(root) == baseline


def test_experiment_status_human_output_quotes_advisory_argv(tmp_path):
    root = tmp_path / "experiment with spaces"
    _init_workspace(root)
    _add_plan(root, step_id="train")

    rendered = experiment_tracking.format_experiment_status(experiments.experiment_status(root))

    assert "recorded evidence, not live" in rendered
    assert "Next legal action" in rendered
    assert "'" in rendered
    assert "Advisory only; this output does not authorize execution." in rendered


def test_experiment_status_cli_returns_one_for_contract_errors(tmp_path, capsys):
    root = tmp_path / "experiment"
    _init_workspace(root)
    _add_plan(root, step_id="train")
    rows = _read_manifest_rows(root)
    rows[0]["status"] = "invented"
    write_rows(root / "run_manifest.tsv", rows)

    assert cli.main(["experiment-status", "--run-dir", str(root), "--json"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "unsupported status" in captured.err


@pytest.mark.parametrize(
    "drift",
    ["config", "run_script", "resolved_recipe", "plan_alias", "plan_escape", "canonical_run"],
)
def test_experiment_status_rejects_registered_plan_drift(tmp_path, drift):
    root = tmp_path / "experiment"
    _init_workspace(root)
    plan_dir, canonical = _add_plan(root, step_id="train")

    if drift == "config":
        Path(canonical["config"]).write_text("model: changed\n")
    elif drift == "run_script":
        (plan_dir / "run.sh").write_text("#!/bin/sh\nexit 1\n")
    elif drift == "resolved_recipe":
        (plan_dir / "recipe.resolved.yaml").write_text("task: changed\n")
    elif drift == "plan_alias":
        plan_path = plan_dir / "plan.json"
        target = plan_dir / "plan.real.json"
        plan_path.rename(target)
        plan_path.symlink_to(target.name)
    elif drift == "plan_escape":
        step_path = root / "steps" / "train" / "step.yaml"
        step_manifest = yaml.safe_load(step_path.read_text())
        step_manifest["plans"] = [str(tmp_path / "outside")]
        step_path.write_text(yaml.safe_dump(step_manifest, sort_keys=False))
    else:
        (root / "run_manifest.tsv").write_text("step_id\trun_id\n")

    with pytest.raises(ValueError):
        experiments.experiment_status(root)


def test_experiment_status_rejects_plan_registered_by_multiple_steps(tmp_path):
    root = tmp_path / "experiment"
    _init_workspace(root)
    plan_dir, _canonical = _add_plan(root, step_id="train")
    second_step = root / "steps" / "evaluate"
    second_step.mkdir(parents=True)
    (second_step / "step.yaml").write_text(
        yaml.safe_dump(
            {
                "step": {"id": "evaluate", "phase": "evaluate", "purpose": "Evaluate."},
                "experiment_id": "status-unit",
                "recipe_path": "",
                "plans": [str(plan_dir)],
            },
            sort_keys=False,
        )
    )

    with pytest.raises(ValueError, match="more than one managed step"):
        experiments.experiment_status(root)


def test_experiment_status_routes_all_registered_reads_to_remote(monkeypatch):
    root = Path("/remote/experiment")
    experiment = {
        "id": "status-unit",
        "title": "Status unit experiment",
        "objective": "Remote status.",
        "root": str(root),
        "baseline": "unit baseline",
    }
    step = {"id": "train", "phase": "train", "purpose": "Train."}
    manifest = {
        "step": step,
        "experiment_id": "status-unit",
        "recipe_path": "",
        "plans": [str(root / "plans" / "train")],
    }
    row = {
        "experiment_id": "status-unit",
        "step_id": "train",
        "run_id": "run-000",
        "run_name": "default",
        "status": "unknown_remote",
    }
    calls = []

    def managed_workspace(candidate, *, remote, allow_completed):
        calls.append(("workspace", candidate, remote, allow_completed))
        return experiment, [row]

    def registered_steps(candidate, *, experiment_id, remote):
        calls.append(("steps", candidate, experiment_id, remote))
        return [manifest]

    def registered_plan(candidate, *, workspace, step_manifest, workspace_rows, remote):
        calls.append(("plan", candidate, workspace, step_manifest, workspace_rows, remote))
        candidate_path = Path(candidate)
        return {
            "path": str(candidate),
            "task": "finetune",
            "adaptive": False,
            "pipeline": False,
            "run_keys": [("train", "run-000")],
            "launch_script": str(candidate_path / "run.sh"),
        }

    monkeypatch.setattr(experiments, "_managed_workspace", managed_workspace)
    monkeypatch.setattr(experiments, "read_registered_steps", registered_steps)
    monkeypatch.setattr(experiments.artifacts, "read_registered_plan", registered_plan)

    snapshot = experiments.experiment_status(root, remote="baichuan3")

    assert calls[0] == ("workspace", root, "baichuan3", True)
    assert calls[1] == ("steps", root, "status-unit", "baichuan3")
    assert calls[2][-1] == "baichuan3"
    assert snapshot["experiment"]["remote"] == "baichuan3"
    assert snapshot["decision"]["recommended_next"]["argv"][-2:] == ["--remote", "baichuan3"]
