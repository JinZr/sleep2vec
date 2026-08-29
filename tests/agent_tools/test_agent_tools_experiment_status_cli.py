from __future__ import annotations

import json
from pathlib import Path
import subprocess

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
    decision_paths,
    decisions,
    experiment_io,
    experiment_tracking,
    experiments,
    plan_context,
    plans,
    recipes,
)
from agent_tools.adapters import all_adapters
from agent_tools.manifests import write_rows


def test_experiment_status_is_zero_write_and_ignores_projections(tmp_path, monkeypatch):
    root = tmp_path / "experiment"
    _init_workspace(root)
    plan_dir, _row = _add_plan(
        root,
        step_id="tune",
        task="hparam_tune",
        status="unknown_scheduler",
        adaptive=True,
    )
    _add_plan(root, step_id="evaluate", pipeline=True)
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
    monkeypatch.setattr(plan_context, "load_config_summary_for_recipe", unexpected)
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
    pipeline_root = root / "pipelines" / "pipeline-unit"
    pipeline_root.mkdir(parents=True)
    (pipeline_root / "pipeline.json").write_text("{\n")
    (pipeline_root / "jobs.tsv").write_text("status\ncompleted\n")
    (root / "adaptive").mkdir()
    (root / "adaptive" / "workflow.json").write_text('{"completed": true}\n')

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

    def swap_then_read(path, *, remote=None):
        nonlocal followed
        if Path(path) == manifest:
            manifest.unlink()
            manifest.symlink_to(outside)
            followed = True
        return real_read(path, remote=remote)

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
                    "plan_controller": "unassigned",
                    "recipe_path": "",
                    "plans": [],
                },
                sort_keys=False,
            )
        )

    rendered = experiment_tracking.format_experiment_status(experiments.experiment_status(root))

    assert "`unmaterialized_step` [step=first]" in rendered
    assert "`unmaterialized_step` [step=second]" in rendered


def test_experiment_status_separates_control_host_from_execution_transport():
    root = Path("/remote/experiment")
    row = {
        "step_id": "train",
        "run_id": "run-000",
        "run_name": "default",
        "status": "planned",
        "target": "ssh",
        "host": "gpu-worker",
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
        {"id": "status-unit", "title": "Remote status"},
        registered_steps,
        [row],
        root=root,
        remote="baichuan3",
    )
    action = snapshot["decision"]["recommended_next"]

    assert action["control_host"] == "baichuan3"
    assert snapshot["runs"][0]["execution"] == {"target": "ssh", "host": "gpu-worker"}
    rendered = experiment_tracking.format_experiment_status(snapshot)
    assert "| train | train | ordinary | planned=1 |" in rendered
    assert "ssh:gpu-worker" in rendered
    assert "control host: `baichuan3`" in rendered
    assert f"bash {root / 'plans' / 'train' / 'run.sh'}" in rendered


def test_experiment_status_keeps_local_hparam_queue_on_controller(tmp_path):
    root = tmp_path / "experiment"
    _init_workspace(root)
    plan_dir, _row = _add_plan(
        root,
        step_id="tune",
        task="hparam_tune",
        status="pending",
        host="baichuan3",
    )

    snapshot = experiments.experiment_status(root)
    action = snapshot["decision"]["recommended_next"]

    assert action["id"] == "hparam-run-queue"
    assert action["control_host"] is None
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


def test_experiment_status_keeps_remote_finalize_on_controller():
    root = Path("/remote/experiment")
    row = {"step_id": "train", "run_id": "run-000", "run_name": "default", "status": "completed"}
    registered_steps = [
        {
            "manifest": {
                "step": {"id": "train", "phase": "train", "purpose": "Train."},
                "plan_controller": "ordinary",
            },
            "plans": [
                {
                    "path": str(root / "plans" / "train"),
                    "run_keys": [("train", "run-000")],
                }
            ],
        }
    ]

    snapshot = experiment_tracking.experiment_status_snapshot(
        {"id": "status-unit", "title": "Remote status"},
        registered_steps,
        [row],
        root=root,
        remote="baichuan3",
    )
    action = snapshot["decision"]["other_legal_actions"][0]

    assert action["control_host"] is None
    assert action["argv"][-2:] == ["--remote", "baichuan3"]


def test_experiment_status_cli_converts_remote_timeout_to_exit_one(monkeypatch, capsys):
    def timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(["ssh", "baichuan3"], 10)

    monkeypatch.setattr(cli, "experiment_status", timeout)

    assert cli.main(["experiment-status", "--run-dir", "/remote/experiment", "--remote", "baichuan3"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("error:")
    assert "Traceback" not in captured.err


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


@pytest.mark.parametrize("adaptive", [None, False, "invalid", ["invalid"]])
def test_experiment_status_cli_returns_one_for_non_mapping_adaptive(tmp_path, capsys, adaptive):
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
    assert "adaptive must be a mapping" in captured.err
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
