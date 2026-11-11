from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from test_agent_tools_experiment_status import (
    _add_plan,
    _init_workspace,
    _read_manifest_rows,
    _sha256,
    _workspace_files,
    _write_public_hparam_recipe,
)
from test_agent_tools_experiment_status import _stub_execution_target  # noqa: F401
import yaml

from agent_tools import (
    cli,
    configs,
    decision_paths,
    decisions,
    experiment_io,
    experiment_tracking,
    experiments,
    managed_scheduler,
    plans,
    recipes,
)
from agent_tools.adapters.registry import all_adapters
from agent_tools.manifests import write_rows


def test_experiment_status_is_zero_write_and_ignores_projections(tmp_path, monkeypatch):
    root = tmp_path / "experiment"
    _init_workspace(root)
    plan_dir, _row = _add_plan(
        root,
        step_id="tune",
        task="hparam_tune",
        status="unknown_scheduler",
    )
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
    monkeypatch.setattr(plans, "evaluate_recipe", unexpected)
    monkeypatch.setattr(decisions, "evaluate_consultation_gates", unexpected)
    monkeypatch.setattr(recipes, "load_consultation_policy", unexpected)
    monkeypatch.setattr(decision_paths, "path_issues", unexpected)
    monkeypatch.setattr(configs, "load_config_summary_for_recipe", unexpected)
    for adapter_type in {type(adapter) for adapter in all_adapters()}:
        for name in ("task_issues", "preflight_issues", "configured_input_issues"):
            monkeypatch.setattr(adapter_type, name, unexpected)

    baseline = experiments.experiment_status(root)
    assert _workspace_files(root) == before

    (root / "experiment_manifest.tsv").write_text("not\ta\tvalid\tprojection\n")
    (root / "run_status.tsv").write_text("not\ta\tvalid\tprojection\n")
    (plan_dir / "launch_manifest.tsv").write_text("status\ncompleted\n")
    (root / "reports").mkdir()
    (root / "reports" / "monitor.md").write_text("completed\n")
    (root / "events.jsonl").write_text("not-json\n")
    (root / "wandb").mkdir()
    (root / "wandb" / "runs.tsv").write_text("status\ncompleted\n")

    assert experiments.experiment_status(root) == baseline


def test_experiment_status_contract_error_is_zero_write(tmp_path, monkeypatch, capsys):
    root = tmp_path / "experiment"
    _init_workspace(root)
    plan_dir, _row = _add_plan(root, step_id="train")
    plan = json.loads((plan_dir / "plan.json").read_text())
    plan["recipe"]["unknown"] = True
    (plan_dir / "plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    resolved = yaml.safe_load((plan_dir / "recipe.resolved.yaml").read_text())
    resolved["unknown"] = True
    (plan_dir / "recipe.resolved.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False))
    before = _workspace_files(root)

    def unexpected(*_args, **_kwargs):
        raise AssertionError("experiment-status attempted a write")

    for name in (
        "write_text_at",
        "write_rows_at",
        "conditional_atomic_replace_text_at",
        "append_event_at",
        "mkdir_experiment_dirs",
    ):
        monkeypatch.setattr(experiment_io, name, unexpected)
    monkeypatch.setattr(experiments, "merge_run_manifest", unexpected)

    assert cli.main(["experiment-status", "--run-dir", str(root), "--json"]) == 1
    assert _workspace_files(root) == before
    assert "Traceback" not in capsys.readouterr().err


def test_experiment_status_does_not_use_following_read_for_experiment_manifest(tmp_path, monkeypatch):
    root = tmp_path / "experiment"
    _init_workspace(root)
    _add_plan(root, step_id="train")
    manifest = root / "experiment.yaml"
    outside = tmp_path / "outside-experiment.yaml"
    outside.write_bytes(manifest.read_bytes())
    real_read = experiment_io.read_text_at
    followed = False

    def swap_then_read(path):
        nonlocal followed
        if Path(path) == manifest:
            manifest.unlink()
            manifest.symlink_to(outside)
            followed = True
        return real_read(path)

    monkeypatch.setattr(experiment_io, "read_text_at", swap_then_read)

    snapshot = experiments.experiment_status(root)

    assert snapshot["experiment"]["id"] == "status-unit"
    assert not followed


def test_experiment_status_human_output_quotes_advisory_argv(tmp_path):
    root = tmp_path / "experiment with spaces"
    _init_workspace(root)
    _add_plan(root, step_id="train")

    rendered = experiment_tracking.format_experiment_status(experiments.experiment_status(root))

    assert "recorded evidence, not live" in rendered
    assert "| Step | Phase | Plan controller | Status counts |" in rendered
    assert (
        "| Run | Canonical | Execution transport | Scheduler | Process | Checkpoints | Runtime manifest | Blocker |"
        in rendered
    )
    assert "Test evidence" not in rendered
    assert "Next legal action" in rendered
    assert "'" in rendered
    assert "control host:" not in rendered
    assert "Advisory only; this output does not authorize execution." in rendered


def test_experiment_status_human_output_scopes_same_code_blockers(tmp_path):
    root = tmp_path / "experiment"
    _init_workspace(root)
    for step_id in ("first", "second"):
        step_dir = root / "steps" / step_id
        step_dir.mkdir(parents=True)
        (step_dir / "step.yaml").write_text(
            yaml.safe_dump(
                {
                    "step": {"id": step_id, "phase": "evaluate", "purpose": f"Run {step_id}."},
                    "experiment_id": "status-unit",
                    "plan_controller": "ordinary",
                    "recipe_path": "",
                    "plans": [],
                },
                sort_keys=False,
            )
        )

    rendered = experiment_tracking.format_experiment_status(experiments.experiment_status(root))

    assert "`unmaterialized_step` [step=first]" in rendered
    assert "`unmaterialized_step` [step=second]" in rendered


def test_experiment_status_projects_execution_target_without_control_host():
    root = Path("/experiment")
    row = {
        "step_id": "train",
        "run_id": "run-000",
        "run_name": "default",
        "status": "planned",
        "target": "local",
    }
    registered_steps = [
        {
            "manifest": {
                "step": {"id": "train", "phase": "train", "purpose": "Train."},
                "plan_controller": "ordinary",
            },
            "plans": [
                {
                    "path": str(root / "plans" / "train"),
                    "task": "finetune",
                    "run_keys": [("train", "run-000")],
                    "launch_script": str(root / "plans" / "train" / "run.sh"),
                }
            ],
        }
    ]

    snapshot = experiment_tracking.experiment_status_snapshot(
        {"id": "status-unit", "title": "Local status"},
        registered_steps,
        [row],
        root=root,
    )
    action = snapshot["decision"]["recommended_next"]

    assert "control_host" not in action
    assert snapshot["runs"][0]["execution"] == {"target": "local"}
    rendered = experiment_tracking.format_experiment_status(snapshot)
    assert "| train | train | ordinary | planned=1 |" in rendered
    assert "| train / run-000 — default | planned | local |" in rendered
    assert "control host:" not in rendered
    assert f"bash {root / 'plans' / 'train' / 'run.sh'}" in rendered


def test_experiment_status_projects_recorded_scheduler_node():
    root = Path("/experiment")
    plan_dir = root / "plans" / "train"
    row = {
        "step_id": "train",
        "run_id": "run-000",
        "run_name": "default",
        "status": "running",
        "scheduler_type": "slurm",
        "scheduler_submit_token": "unit-token",
        "scheduler_script": str(plan_dir / "run-000.sbatch"),
        "scheduler_script_sha256": "a" * 64,
        "scheduler_result_path": str(plan_dir / "run-000.result.json"),
        "allocation_identity_path": str(plan_dir / "run-000.allocation.json"),
        "scheduler_job_id": "3880",
        "scheduler_node": "gpu-node-02",
    }
    registered_steps = [
        {
            "manifest": {
                "step": {"id": "train", "phase": "train", "purpose": "Train."},
                "plan_controller": "ordinary",
            },
            "plans": [
                {
                    "path": str(plan_dir),
                    "task": "finetune",
                    "run_keys": [("train", "run-000")],
                    "launch_script": str(plan_dir / "run.sh"),
                }
            ],
        }
    ]

    snapshot = experiment_tracking.experiment_status_snapshot(
        {"id": "status-unit", "title": "Scheduler status"},
        registered_steps,
        [row],
        root=root,
    )

    assert snapshot["runs"][0]["scheduler"]["node"] == "gpu-node-02"
    assert "node=gpu-node-02" in experiment_tracking.format_experiment_status(snapshot)


def test_experiment_status_keeps_local_hparam_queue_on_controller(tmp_path):
    root = tmp_path / "experiment"
    _init_workspace(root)
    plan_dir, _row = _add_plan(
        root,
        step_id="tune",
        task="hparam_tune",
        status="pending",
    )

    snapshot = experiments.experiment_status(root)
    action = snapshot["decision"]["recommended_next"]

    assert action["id"] == "hparam-run-queue"
    assert "control_host" not in action
    assert action["argv"] == [
        "python",
        "-m",
        "agent_tools",
        "hparam-run-queue",
        "--plan-dir",
        str(plan_dir),
        "--execute",
    ]
    assert "control host:" not in experiment_tracking.format_experiment_status(snapshot)


def test_experiment_status_defers_the_hparam_queue_to_an_active_runner(tmp_path):
    root = tmp_path / "experiment"
    _init_workspace(root)
    plan_dir, _row = _add_plan(
        root,
        step_id="tune",
        task="hparam_tune",
        status="pending",
    )
    log_path = plan_dir / "queue_runner.log"

    with managed_scheduler.queue_runner_lock(plan_dir, record_pid=True):
        snapshot = experiments.experiment_status(root)
        text = experiment_tracking.format_experiment_status(snapshot)

    assert snapshot["summary"]["state"] == "in_progress"
    assert snapshot["decision"]["recommended_next"]["id"] == "experiment-monitor"
    assert snapshot["decision"]["other_legal_actions"] == []
    [blocker] = [blocker for blocker in snapshot["blockers"] if blocker["code"] == "queue_runner_active"]
    assert blocker["step_id"] == "tune"
    assert f"active; pid={os.getpid()}; log={log_path}" in blocker["message"]
    assert "- `queue_runner_active` [step=tune]: A detached hparam-run-queue runner" in text
    assert "hparam-run-queue --plan-dir" not in text

    # A released lock leaves the file behind; the probe then reports no runner and the queue is advised again.
    released = experiments.experiment_status(root)
    assert released["decision"]["recommended_next"]["id"] == "hparam-run-queue"
    assert not any(blocker["code"] == "queue_runner_active" for blocker in released["blockers"])


def test_experiment_monitor_reports_queue_runner_liveness_without_starting_one(tmp_path, monkeypatch, capsys):
    root = tmp_path / "experiment"
    _init_workspace(root)
    plan_dir, _row = _add_plan(
        root,
        step_id="tune",
        task="hparam_tune",
        status="pending",
    )
    log_path = plan_dir / "queue_runner.log"
    # Keep run observation inert; this test pins only the runner probe and its report.
    monkeypatch.setattr(
        experiment_tracking,
        "monitor_run_row",
        lambda _root, row, _previous_rows, **_kwargs: {
            "step_id": row["step_id"],
            "run_id": row["run_id"],
            "status": row["status"],
        },
    )

    assert experiments.monitor_experiment(root)["queue_runners"] == []

    with managed_scheduler.queue_runner_lock(plan_dir, record_pid=True):
        assert cli.main(["experiment-monitor", "--run-dir", str(root)]) == 0
    report = (root / "reports" / "monitor.md").read_text()
    after = experiments.monitor_experiment(root)

    active_text = f"active; pid={os.getpid()}; log={log_path}"
    assert capsys.readouterr().out.splitlines()[0] == f"Queue runner {plan_dir}: {active_text}"
    assert f"## Queue runners\n\n- `{plan_dir}`: {active_text}\n" in report
    assert after["queue_runners"] == [
        {"plan_dir": str(plan_dir), "active": False, "pid": None, "log_path": str(log_path)}
    ]
    assert _read_manifest_rows(root)[0]["status"] == "pending"
    assert not log_path.exists()


def test_experiment_status_cli_converts_malformed_yaml_to_exit_one(tmp_path, capsys):
    root = tmp_path / "experiment"
    root.mkdir()
    (root / "experiment.yaml").write_text("experiment: [\n")
    (root / "run_manifest.tsv").write_text("step_id\trun_id\n")

    assert cli.main(["experiment-status", "--run-dir", str(root)]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "invalid YAML" in captured.err
    assert "Traceback" not in captured.err


def test_experiment_status_rejects_unknown_experiment_envelope_field(tmp_path):
    root = tmp_path / "experiment"
    _init_workspace(root)
    manifest_path = root / "experiment.yaml"
    manifest = yaml.safe_load(manifest_path.read_text())
    manifest["duplicate_owner"] = {"status": "completed"}
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False))

    with pytest.raises(ValueError, match="only the experiment owner mapping"):
        experiments.experiment_status(root)


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


def test_experiment_status_cli_returns_one_for_non_mapping_plan_run(tmp_path, capsys):
    root = tmp_path / "experiment"
    _init_workspace(root)
    plan_dir, _canonical = _add_plan(root, step_id="train")
    plan = json.loads((plan_dir / "plan.json").read_text())
    plan["runs"] = [None]
    (plan_dir / "plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")

    assert cli.main(["experiment-status", "--run-dir", str(root), "--json"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("error:")
    assert "Traceback" not in captured.err


@pytest.mark.parametrize("binding", ["experiment", "step"])
def test_experiment_status_cli_returns_one_for_non_mapping_plan_binding(tmp_path, capsys, binding):
    root = tmp_path / "experiment"
    _init_workspace(root)
    plan_dir, _canonical = _add_plan(root, step_id="train")
    plan_path = plan_dir / "plan.json"
    plan = json.loads(plan_path.read_text())
    plan["recipe"][binding] = "invalid"
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    resolved_path = plan_dir / "recipe.resolved.yaml"
    resolved = yaml.safe_load(resolved_path.read_text())
    resolved[binding] = "invalid"
    resolved_path.write_text(yaml.safe_dump(resolved, sort_keys=False))
    before = _workspace_files(root)

    assert cli.main(["experiment-status", "--run-dir", str(root), "--json"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert f"{binding} must be a mapping" in captured.err
    assert "Traceback" not in captured.err
    assert _workspace_files(root) == before


@pytest.mark.parametrize("adaptive", [{"enabled": True}, None])
def test_experiment_status_cli_rejects_frozen_adaptive_block(tmp_path, capsys, adaptive):
    root = tmp_path / "experiment"
    _init_workspace(root)
    plan_dir, _canonical = _add_plan(root, step_id="train", task="hparam_tune")
    plan = json.loads((plan_dir / "plan.json").read_text())
    plan["recipe"]["adaptive"] = adaptive
    plan["recipe"]["_local_recipe"]["adaptive"] = adaptive
    resolved = yaml.safe_load((plan_dir / "recipe.resolved.yaml").read_text())
    resolved["adaptive"] = adaptive
    resolved["_local_recipe"]["adaptive"] = adaptive
    resolved_path = plan_dir / "recipe.resolved.yaml"
    resolved_path.write_text(yaml.safe_dump(resolved, sort_keys=False))
    plan["resolved_recipe_sha256"] = _sha256(resolved_path)
    (plan_dir / "plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")

    assert cli.main(["experiment-status", "--run-dir", str(root), "--json"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Unknown recipe field for task=hparam_tune: adaptive" in captured.err
    assert "Traceback" not in captured.err


@pytest.mark.parametrize(
    ("parameter", "mutation"),
    [
        ("yaml:/finetune/task/output_dim", "missing_key"),
        ("yaml:/data/data_channel_names/0", "missing_index"),
        ("yaml:/finetune/task/output_dim", "wrong_parent_type"),
        ("yaml:/finetune/task/output_dim", "malformed_yaml"),
    ],
)
def test_experiment_status_cli_converts_corrupt_frozen_hparam_config_errors(
    tmp_path,
    capsys,
    parameter,
    mutation,
):
    root = tmp_path / "experiment"
    recipe = _write_public_hparam_recipe(root, {parameter: [31 if parameter.endswith("output_dim") else "ppg"]})
    plan_dir = root / "plans" / "tune"
    assert plans.build_plan(recipe_path=recipe, output_dir=plan_dir).exit_code == 0

    source_config = plan_dir / "config.source.yaml"
    if mutation == "malformed_yaml":
        source_config.write_text("[unclosed")
    else:
        config = yaml.safe_load(source_config.read_text())
        if mutation == "missing_key":
            del config["finetune"]["task"]
        elif mutation == "missing_index":
            config["data"]["data_channel_names"] = []
        else:
            config["finetune"] = 1
        source_config.write_text(yaml.safe_dump(config, sort_keys=False))

    plan_path = plan_dir / "plan.json"
    plan = json.loads(plan_path.read_text())
    source_sha256 = _sha256(source_config)
    for snapshot in plan["recipe"]["input_snapshots"]:
        if snapshot["field"] == "inputs.config":
            snapshot["sha256"] = source_sha256
    resolved_path = plan_dir / "recipe.resolved.yaml"
    resolved = yaml.safe_load(resolved_path.read_text())
    resolved["input_snapshots"] = plan["recipe"]["input_snapshots"]
    resolved_path.write_text(yaml.safe_dump(resolved, sort_keys=False))
    plan["resolved_recipe_sha256"] = _sha256(resolved_path)
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")

    before = _workspace_files(root)
    assert cli.main(["experiment-status", "--run-dir", str(root), "--json"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Registered plan frozen config is corrupt" in captured.err
    assert "Traceback" not in captured.err
    assert _workspace_files(root) == before


@pytest.mark.parametrize("mutation", ["malformed_yaml", "invalid_analyzer"])
def test_experiment_status_cli_converts_corrupt_sleep2stat_config_error(tmp_path, capsys, mutation):
    root = tmp_path / "experiment"
    _init_workspace(root)
    plan_dir, canonical = _add_plan(root, step_id="analyze", task="sleep2stat")
    config_path = Path(canonical["config"])
    if mutation == "malformed_yaml":
        config_path.write_text("[unclosed")
    else:
        config_path.write_text(
            yaml.safe_dump(
                {"run": {"output_dir": str(root / "analysis")}, "analyzers": ["invalid"], "reducers": []},
                sort_keys=False,
            )
        )
    config_sha256 = _sha256(config_path)

    plan_path = plan_dir / "plan.json"
    plan = json.loads(plan_path.read_text())
    plan["recipe"]["input_snapshots"][0]["sha256"] = config_sha256
    plan["runs"][0]["config_sha256"] = config_sha256
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    resolved_path = plan_dir / "recipe.resolved.yaml"
    resolved = yaml.safe_load(resolved_path.read_text())
    resolved["input_snapshots"] = plan["recipe"]["input_snapshots"]
    resolved_path.write_text(yaml.safe_dump(resolved, sort_keys=False))
    canonical["config_sha256"] = config_sha256
    write_rows(root / "run_manifest.tsv", [canonical])

    before = _workspace_files(root)
    assert cli.main(["experiment-status", "--run-dir", str(root), "--json"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Registered plan frozen config is corrupt" in captured.err
    assert "Traceback" not in captured.err
    assert _workspace_files(root) == before
