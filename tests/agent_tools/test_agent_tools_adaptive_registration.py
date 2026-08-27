from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from agent_tools import adaptive_hparam, hparam_runtime, managed_scheduler, manifests, plan_hparam, run_artifacts
from tests.agent_tools import adaptive_hparam_test_support as test_support
from tests.agent_tools.adaptive_hparam_test_support import _adaptive_recipe, _agent_recipe, _read_table, _run

_stub_execution_snapshot_preflight = test_support._stub_execution_snapshot_preflight


def test_adaptive_init_preflight_leaves_blocked_root_untouched_then_retries(tmp_path: Path):
    source = tmp_path / "source"
    recipe = _adaptive_recipe(source)
    workspace = tmp_path / "adaptive-workspace"
    payload = yaml.safe_load(recipe.read_text())
    payload["experiment"]["root"] = str(workspace)
    payload["decisions"]["label_name"] = {"value": "ASK_USER", "source": "explicit_recipe"}
    recipe.write_text(yaml.safe_dump(payload))

    blocked = _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workspace))

    assert blocked.returncode != 0
    assert not workspace.exists()
    payload["decisions"]["label_name"] = {"value": "ahi", "source": "explicit_recipe"}
    recipe.write_text(yaml.safe_dump(payload))

    retry = _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workspace))

    assert retry.returncode == 0, retry.stderr
    assert (workspace / "adaptive" / "rounds" / "round_000" / "plan.json").exists()


def test_generic_plan_rejects_adaptive_recipe_before_workspace_mutation(tmp_path: Path):
    source = tmp_path / "source"
    recipe = _adaptive_recipe(source)
    workspace = tmp_path / "adaptive-workspace"
    payload = yaml.safe_load(recipe.read_text())
    payload["experiment"]["root"] = str(workspace)
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))

    result = _run(
        "plan",
        "--recipe",
        str(recipe),
        "--output-dir",
        str(workspace / "plans" / "generic"),
    )

    assert result.returncode == 1
    assert "Adaptive recipes must be initialized with hparam-adaptive-init" in result.stdout
    assert not workspace.exists()


def test_generic_plan_allows_disabled_adaptive_block(tmp_path: Path):
    source = tmp_path / "source"
    recipe = _adaptive_recipe(source)
    workspace = tmp_path / "static-workspace"
    payload = yaml.safe_load(recipe.read_text())
    payload["experiment"]["root"] = str(workspace)
    payload["adaptive"]["enabled"] = False
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    output_dir = workspace / "plans" / "static"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 0, result.stderr or result.stdout
    assert (output_dir / "plan.json").exists()
    assert not (workspace / "adaptive" / "workflow.json").exists()


def test_adaptive_init_creates_round_zero_without_modifying_original_recipe(tmp_path: Path):
    recipe = _adaptive_recipe(tmp_path)
    before = recipe.read_text()
    workflow_dir = tmp_path / "workflow"

    result = _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir))

    assert result.returncode == 0, result.stderr
    assert recipe.read_text() == before
    assert (workflow_dir / "adaptive" / "workflow.json").exists()
    round_plan_path = workflow_dir / "adaptive" / "rounds" / "round_000" / "plan.json"
    assert round_plan_path.exists()
    round_plan = json.loads(round_plan_path.read_text())
    assert "--test-after-fit" in round_plan["runs"][0]["command"]
    assert "--no-test-after-fit" not in round_plan["runs"][0]["command"]
    assert (workflow_dir / "adaptive" / "run_registry.tsv").exists()
    assert "adaptive_init" in (tmp_path / "events.jsonl").read_text()


def test_adaptive_workflow_commit_accepts_relative_round_path(tmp_path: Path, monkeypatch):
    recipe = _adaptive_recipe(tmp_path)
    adaptive_hparam.init_adaptive_workflow(recipe, tmp_path / "workflow")
    monkeypatch.chdir(tmp_path)

    plan = run_artifacts.read_hparam_plan(Path("workflow/adaptive/rounds/round_000"))

    assert [run["run_id"] for run in plan["runs"]] == ["run-000"]


def test_adaptive_workflow_commit_resolves_symlink_ancestor_before_dot_dot(tmp_path: Path):
    recipe = _adaptive_recipe(tmp_path)
    real_parent = tmp_path / "real-parent"
    workflow_dir = adaptive_hparam.init_adaptive_workflow(recipe, real_parent / "fresh-workflow")
    anchor = real_parent / "anchor"
    anchor.mkdir()
    access = tmp_path / "access"
    access.mkdir()
    (access / "hop").symlink_to(anchor, target_is_directory=True)
    round_dir = access / "hop" / ".." / "fresh-workflow" / "adaptive" / "rounds" / "round_000"

    plan = run_artifacts.read_hparam_plan(round_dir)

    assert round_dir.resolve() == workflow_dir / "adaptive" / "rounds" / "round_000"
    assert [run["run_id"] for run in plan["runs"]] == ["run-000"]


@pytest.mark.parametrize("marker_state", ["missing", "root_drift"])
def test_adaptive_workflow_commit_cannot_be_skipped_by_dot_dot_path(tmp_path: Path, marker_state: str):
    recipe = _adaptive_recipe(tmp_path)
    workflow_dir = adaptive_hparam.init_adaptive_workflow(recipe, tmp_path / "workflow")
    workflow_path = workflow_dir / "adaptive" / "workflow.json"
    round_dir = workflow_dir / "adaptive" / "rounds" / ".." / "rounds" / "round_000"
    if marker_state == "missing":
        workflow_path.unlink()
        expected_error = FileNotFoundError
        expected_message = "initialization is not committed"
    else:
        workflow = json.loads(workflow_path.read_text())
        workflow["root"] = str(tmp_path / "other-workflow")
        workflow_path.write_text(json.dumps(workflow))
        expected_error = ValueError
        expected_message = "commit marker differs"

    with pytest.raises(expected_error, match=expected_message):
        run_artifacts.read_hparam_plan(round_dir)


def test_adaptive_workflow_commit_rejects_round_symlink(tmp_path: Path):
    recipe = _adaptive_recipe(tmp_path)
    workflow_dir = adaptive_hparam.init_adaptive_workflow(recipe, tmp_path / "workflow")
    alias = tmp_path / "round-alias"
    alias.symlink_to(workflow_dir / "adaptive" / "rounds" / "round_000", target_is_directory=True)

    with pytest.raises(ValueError, match="must not be a symlink"):
        run_artifacts.read_hparam_plan(alias)


def test_adaptive_init_materialization_failure_keeps_round_unpublished_and_retries(tmp_path: Path, monkeypatch):
    recipe = _adaptive_recipe(tmp_path)
    workflow_dir = tmp_path / "workflow"
    round_dir = workflow_dir / "adaptive" / "rounds" / "round_000"
    original_write_text = plan_hparam.write_text

    def fail_plan_markdown(path, text, *, executable=False):
        if Path(path).name == "plan.md":
            raise OSError("injected plan materialization failure")
        return original_write_text(path, text, executable=executable)

    monkeypatch.setattr(plan_hparam, "write_text", fail_plan_markdown)

    with pytest.raises(OSError, match="injected plan materialization failure"):
        adaptive_hparam.init_adaptive_workflow(recipe, workflow_dir)

    assert not round_dir.exists()
    assert not (workflow_dir / "adaptive" / "workflow.json").exists()
    assert not list((tmp_path / "steps").glob("*/step.yaml"))
    assert _read_table(tmp_path / "run_manifest.tsv") == []
    assert not list((workflow_dir / "adaptive" / "rounds").glob(".*.staging"))

    monkeypatch.setattr(plan_hparam, "write_text", original_write_text)
    adaptive_hparam.init_adaptive_workflow(recipe, workflow_dir)

    plan_text = (round_dir / "plan.json").read_text()
    assert ".staging" not in plan_text
    assert [row["run_id"] for row in _read_table(tmp_path / "run_manifest.tsv")] == ["run-000"]


def test_adaptive_init_target_preflight_failure_leaves_no_registration(tmp_path: Path, monkeypatch):
    recipe = _adaptive_recipe(tmp_path)
    workflow_dir = tmp_path / "workflow"
    manifest_before = (tmp_path / "run_manifest.tsv").read_bytes()
    monkeypatch.setattr(
        managed_scheduler,
        "inspect_execution_target",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("target argv rejected")),
    )

    with pytest.raises(RuntimeError, match="Round 000 plan failed"):
        adaptive_hparam.init_adaptive_workflow(recipe, workflow_dir)

    assert not (workflow_dir / "adaptive" / "rounds" / "round_000").exists()
    assert not (workflow_dir / "adaptive" / "workflow.json").exists()
    assert not (workflow_dir / "adaptive" / "run_registry.tsv").exists()
    assert not list((workflow_dir / "adaptive" / "rounds").glob(".*.staging"))
    assert not (tmp_path / "steps").exists()
    assert (tmp_path / "run_manifest.tsv").read_bytes() == manifest_before


@pytest.mark.parametrize("tamper", [False, True])
def test_adaptive_init_recovers_published_round_before_canonical_registration(
    tmp_path: Path, monkeypatch, tamper: bool
):
    recipe = _adaptive_recipe(tmp_path)
    workflow_dir = tmp_path / "workflow"
    round_dir = workflow_dir / "adaptive" / "rounds" / "round_000"
    original_commit = plan_hparam.commit_hparam_plan
    monkeypatch.setattr(
        plan_hparam,
        "commit_hparam_plan",
        lambda _round_dir, **_kwargs: (_ for _ in ()).throw(OSError("injected canonical registration failure")),
    )

    with pytest.raises(OSError, match="injected canonical registration failure"):
        adaptive_hparam.init_adaptive_workflow(recipe, workflow_dir)

    assert (round_dir / "plan.json").is_file()
    assert (round_dir / "recipe.resolved.yaml").is_file()
    assert (round_dir / "round_recipe.yaml").is_file()
    assert not (workflow_dir / "adaptive" / "workflow.json").exists()
    assert not list((tmp_path / "steps").glob("*/step.yaml"))
    assert _read_table(tmp_path / "run_manifest.tsv") == []
    plan_bytes = (round_dir / "plan.json").read_bytes()
    with pytest.raises(FileNotFoundError):
        hparam_runtime.launch_hparam_runs(round_dir)
    assert not (round_dir / "launch_manifest.tsv").exists()
    monkeypatch.setattr(hparam_runtime, "monitor_hparam_runs", lambda *_args, **_kwargs: pytest.fail("monitored"))
    monkeypatch.setattr(hparam_runtime, "launch_hparam_runs", lambda *_args, **_kwargs: pytest.fail("launched"))
    monkeypatch.setattr(hparam_runtime.time, "sleep", lambda *_args, **_kwargs: pytest.fail("slept"))
    with pytest.raises(FileNotFoundError):
        hparam_runtime.run_hparam_queue(round_dir, dry_run=False)

    if tamper:
        (round_dir / "plan.md").write_text("tampered\n")

    monkeypatch.setattr(plan_hparam, "commit_hparam_plan", original_commit)
    original_build_plan = adaptive_hparam.build_plan
    build_calls = 0

    def count_build(**kwargs):
        nonlocal build_calls
        build_calls += 1
        return original_build_plan(**kwargs)

    monkeypatch.setattr(adaptive_hparam, "build_plan", count_build)
    if tamper:
        with pytest.raises(ValueError, match="differs from deterministic regeneration"):
            adaptive_hparam.init_adaptive_workflow(recipe, workflow_dir)
    else:
        adaptive_hparam.init_adaptive_workflow(recipe, workflow_dir)

    assert (round_dir / "plan.json").read_bytes() == plan_bytes
    assert build_calls == 1
    if tamper:
        assert _read_table(tmp_path / "run_manifest.tsv") == []
        assert not (workflow_dir / "adaptive" / "workflow.json").exists()
    else:
        assert [row["run_id"] for row in _read_table(tmp_path / "run_manifest.tsv")] == ["run-000"]
        assert (workflow_dir / "adaptive" / "workflow.json").is_file()


def test_adaptive_init_recovers_canonical_round_before_workflow_commit(tmp_path: Path, monkeypatch):
    recipe = _adaptive_recipe(tmp_path)
    workflow_dir = tmp_path / "workflow"
    round_dir = workflow_dir / "adaptive" / "rounds" / "round_000"
    workflow_path = workflow_dir / "adaptive" / "workflow.json"
    original_replace = adaptive_hparam.exp_io.conditional_atomic_replace_text_at

    def fail_workflow_commit(path, text, expected_sha256, *, remote=None, **kwargs):
        if Path(path) == workflow_path:
            raise OSError("injected workflow commit failure")
        return original_replace(path, text, expected_sha256, remote=remote, **kwargs)

    monkeypatch.setattr(adaptive_hparam.exp_io, "conditional_atomic_replace_text_at", fail_workflow_commit)

    with pytest.raises(OSError, match="injected workflow commit failure"):
        adaptive_hparam.init_adaptive_workflow(recipe, workflow_dir)

    plan_bytes = (round_dir / "plan.json").read_bytes()
    assert [row["run_id"] for row in _read_table(tmp_path / "run_manifest.tsv")] == ["run-000"]
    assert len(_read_table(workflow_dir / "adaptive" / "run_registry.tsv")) == 1
    assert not workflow_path.exists()
    with pytest.raises(FileNotFoundError, match="initialization is not committed"):
        hparam_runtime.launch_hparam_runs(round_dir)
    assert not (round_dir / "launch_manifest.tsv").exists()

    monkeypatch.setattr(adaptive_hparam.exp_io, "conditional_atomic_replace_text_at", original_replace)
    monkeypatch.setattr(adaptive_hparam, "build_plan", lambda **_kwargs: pytest.fail("retry rebuilt round 000"))
    adaptive_hparam.init_adaptive_workflow(recipe, workflow_dir)
    adaptive_hparam.init_adaptive_workflow(recipe, workflow_dir)

    assert (round_dir / "plan.json").read_bytes() == plan_bytes
    assert [row["run_id"] for row in _read_table(tmp_path / "run_manifest.tsv")] == ["run-000"]
    assert len(_read_table(workflow_dir / "adaptive" / "run_registry.tsv")) == 1
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert len([event for event in events if event["event_type"] == "plan_created"]) == 1
    assert len([event for event in events if event["event_type"] == "adaptive_init"]) == 1


def test_adaptive_init_does_not_publish_marker_when_registry_validation_fails(tmp_path: Path, monkeypatch):
    recipe = _adaptive_recipe(tmp_path)
    workflow_dir = tmp_path / "workflow"
    round_dir = workflow_dir / "adaptive" / "rounds" / "round_000"
    workflow_path = workflow_dir / "adaptive" / "workflow.json"
    original_registry_writer = adaptive_hparam._write_initial_registry

    def write_invalid_registry(root, initial_round_dir, plan):
        original_registry_writer(root, initial_round_dir, plan)
        registry_path = root / "adaptive" / "run_registry.tsv"
        rows = _read_table(registry_path)
        rows[0]["config"] = str(tmp_path / "other-config.yaml")
        manifests.write_rows(registry_path, rows)

    monkeypatch.setattr(adaptive_hparam, "_write_initial_registry", write_invalid_registry)

    with pytest.raises(ValueError, match="Frozen run field differs"):
        adaptive_hparam.init_adaptive_workflow(recipe, workflow_dir)

    assert (round_dir / "plan.json").is_file()
    assert _read_table(tmp_path / "run_manifest.tsv")
    assert not workflow_path.exists()


@pytest.mark.parametrize("missing_name", ["recipe.resolved.yaml", "plan.md"])
def test_adaptive_init_rejects_visible_incomplete_round_without_repair(tmp_path: Path, missing_name: str):
    recipe = _adaptive_recipe(tmp_path)
    workflow_dir = adaptive_hparam.init_adaptive_workflow(recipe, tmp_path / "workflow")
    round_dir = workflow_dir / "adaptive" / "rounds" / "round_000"
    workflow_path = workflow_dir / "adaptive" / "workflow.json"
    workflow_path.unlink()
    missing_path = round_dir / missing_name
    missing_path.unlink()
    plan_before = (round_dir / "plan.json").read_bytes()
    manifest_path = tmp_path / "run_manifest.tsv"
    manifest_before = manifest_path.read_bytes()

    with pytest.raises(FileNotFoundError):
        adaptive_hparam.init_adaptive_workflow(recipe, workflow_dir)

    assert not missing_path.exists()
    assert (round_dir / "plan.json").read_bytes() == plan_before
    assert manifest_path.read_bytes() == manifest_before
    assert not workflow_path.exists()


def test_adaptive_rounds_keep_frozen_runtime_identity_and_allow_capacity_updates(tmp_path: Path, monkeypatch):
    recipe = _adaptive_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    payload["execution"] = {}
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    workflow_dir = adaptive_hparam.init_adaptive_workflow(recipe, tmp_path / "workflow")
    round_zero = workflow_dir / "adaptive" / "rounds" / "round_000"
    first_plan = json.loads((round_zero / "plan.json").read_text())
    frozen_identity = {field: first_plan["recipe"]["execution"][field] for field in ("python", "runtime_commit")}
    run = first_plan["runs"][0]
    digest = workflow_dir / "adaptive" / "digests" / "round_000.csv"
    manifests.write_rows(digest, [{**run, "test_auroc": 0.73}])
    payload = yaml.safe_load(recipe.read_text())
    payload["execution"]["max_concurrent"] = 2
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    monkeypatch.setattr(plan_hparam, "repo_summary", lambda: pytest.fail("later rounds must not resolve HEAD again"))

    suggestion = adaptive_hparam.suggest_next_round(workflow_dir)
    suggested = yaml.safe_load(suggestion.read_text())
    next_dir = workflow_dir / "adaptive" / "rounds" / "round_001"
    staging_dir = adaptive_hparam._stage_round(
        next_dir,
        adaptive_hparam.load_recipe_with_base(suggestion),
        suggestion,
        1,
        None,
    )
    adaptive_hparam._publish_staged_round(staging_dir, next_dir)
    plan_hparam.commit_hparam_plan(next_dir)

    assert suggested["execution"]["max_concurrent"] == 2
    assert {field: suggested["execution"][field] for field in ("python", "runtime_commit")} == frozen_identity
    second_plan = json.loads((next_dir / "plan.json").read_text())
    assert {
        field: second_plan["recipe"]["execution"][field] for field in ("python", "runtime_commit")
    } == frozen_identity


@pytest.mark.parametrize("tamper", ["missing", "extra"])
def test_adaptive_workflow_rejects_invalid_execution_identity_before_suggestion_write(tmp_path: Path, tamper: str):
    recipe = _adaptive_recipe(tmp_path)
    workflow_dir = adaptive_hparam.init_adaptive_workflow(recipe, tmp_path / "workflow")
    round_zero = workflow_dir / "adaptive" / "rounds" / "round_000"
    run = json.loads((round_zero / "plan.json").read_text())["runs"][0]
    digest = workflow_dir / "adaptive" / "digests" / "round_000.csv"
    manifests.write_rows(digest, [{**run, "test_auroc": 0.73}])
    workflow_path = workflow_dir / "adaptive" / "workflow.json"
    workflow = json.loads(workflow_path.read_text())
    if tamper == "missing":
        workflow["execution_identity"].pop("runtime_commit")
    else:
        workflow["execution_identity"]["max_concurrent"] = 8
    workflow_path.write_text(json.dumps(workflow))
    events_path = tmp_path / "events.jsonl"
    events_before = events_path.read_bytes()

    with pytest.raises(ValueError, match="frozen execution identity"):
        adaptive_hparam.suggest_next_round(workflow_dir)

    assert not (workflow_dir / "adaptive" / "suggestions").exists()
    assert events_path.read_bytes() == events_before


@pytest.mark.parametrize(
    ("field", "value"),
    [("python", "/other/python"), ("runtime_commit", "b" * 40)],
)
def test_adaptive_source_rejects_frozen_execution_identity_drift_before_suggestion_write(
    tmp_path: Path, field: str, value: str
):
    recipe = _adaptive_recipe(tmp_path)
    workflow_dir = adaptive_hparam.init_adaptive_workflow(recipe, tmp_path / "workflow")
    round_zero = workflow_dir / "adaptive" / "rounds" / "round_000"
    run = json.loads((round_zero / "plan.json").read_text())["runs"][0]
    digest = workflow_dir / "adaptive" / "digests" / "round_000.csv"
    manifests.write_rows(digest, [{**run, "test_auroc": 0.73}])
    payload = yaml.safe_load(recipe.read_text())
    payload["execution"][field] = value
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    events_path = tmp_path / "events.jsonl"
    events_before = events_path.read_bytes()

    with pytest.raises(ValueError, match=f"execution.{field} differs"):
        adaptive_hparam.suggest_next_round(workflow_dir)

    assert not (workflow_dir / "adaptive" / "suggestions").exists()
    assert events_path.read_bytes() == events_before


@pytest.mark.parametrize(
    ("field", "value"),
    [("objective_metric", "val_loss"), ("objective_mode", "min")],
)
def test_adaptive_source_rejects_frozen_objective_drift_before_suggestion_write(tmp_path: Path, field: str, value: str):
    recipe = _adaptive_recipe(tmp_path)
    workflow_dir = adaptive_hparam.init_adaptive_workflow(recipe, tmp_path / "workflow")
    round_zero = workflow_dir / "adaptive" / "rounds" / "round_000"
    run = json.loads((round_zero / "plan.json").read_text())["runs"][0]
    digest = workflow_dir / "adaptive" / "digests" / "round_000.csv"
    manifests.write_rows(digest, [{**run, "test_auroc": 0.73, "val_loss": 0.42}])
    payload = yaml.safe_load(recipe.read_text())
    payload["adaptive"][field] = value
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    events_path = tmp_path / "events.jsonl"
    events_before = events_path.read_bytes()

    with pytest.raises(ValueError, match=rf"adaptive\.{field} differs"):
        adaptive_hparam.suggest_next_round(workflow_dir)

    assert not (workflow_dir / "adaptive" / "suggestions").exists()
    assert events_path.read_bytes() == events_before


@pytest.mark.parametrize(
    ("field", "value"),
    [("objective_metric", "val_loss"), ("objective_mode", "min")],
)
def test_agent_proposal_source_rejects_frozen_objective_drift_before_digest_write(
    tmp_path: Path, field: str, value: str
):
    recipe = _agent_recipe(tmp_path)
    workflow_dir = adaptive_hparam.init_adaptive_workflow(recipe, tmp_path / "workflow")
    payload = yaml.safe_load(recipe.read_text())
    payload["adaptive"][field] = value
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    events_path = tmp_path / "events.jsonl"
    events_before = events_path.read_bytes()

    with pytest.raises(ValueError, match=rf"adaptive\.{field} differs"):
        adaptive_hparam.adaptive_step(workflow_dir)

    assert not (workflow_dir / "adaptive" / "digests").exists()
    assert not (workflow_dir / "adaptive" / "proposal_inputs").exists()
    assert events_path.read_bytes() == events_before


def test_adaptive_source_accepts_equivalent_default_objective_after_init(tmp_path: Path):
    recipe = _adaptive_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    payload["adaptive"]["objective_metric"] = None
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    workflow_dir = adaptive_hparam.init_adaptive_workflow(recipe, tmp_path / "workflow")
    round_zero = workflow_dir / "adaptive" / "rounds" / "round_000"
    run = json.loads((round_zero / "plan.json").read_text())["runs"][0]
    digest = workflow_dir / "adaptive" / "digests" / "round_000.csv"
    manifests.write_rows(digest, [{**run, "test_auroc": 0.73}])
    payload = yaml.safe_load(recipe.read_text())
    payload["adaptive"].pop("objective_metric")
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))

    suggestion = adaptive_hparam.suggest_next_round(workflow_dir)

    workflow = json.loads((workflow_dir / "adaptive" / "workflow.json").read_text())
    assert workflow["objective_metric"] == "test_auroc"
    assert suggestion.exists()


def test_adaptive_init_initializes_fresh_experiment_root(tmp_path: Path):
    source_dir = tmp_path / "source"
    recipe = _adaptive_recipe(source_dir)
    payload = yaml.safe_load(recipe.read_text())
    workflow_dir = tmp_path / "fresh-workflow"
    payload["experiment"]["root"] = str(workflow_dir)
    recipe.write_text(yaml.safe_dump(payload))

    assert not workflow_dir.exists()
    result = _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir))

    assert result.returncode == 0, result.stderr
    assert (workflow_dir / "experiment.yaml").exists()
    assert (workflow_dir / "adaptive" / "workflow.json").exists()
    assert (workflow_dir / "adaptive" / "rounds" / "round_000" / "plan.json").exists()
    assert _read_table(workflow_dir / "run_manifest.tsv")


def test_adaptive_init_preflights_round_recipe_before_workspace_mutation(tmp_path: Path):
    recipe = _adaptive_recipe(tmp_path)
    workflow_dir = tmp_path / "workflow"
    round_recipe = workflow_dir / "adaptive" / "rounds" / "round_000" / "round_recipe.yaml"
    round_recipe.parent.mkdir(parents=True)
    round_recipe.symlink_to(tmp_path / "run_manifest.tsv")
    manifest_before = (tmp_path / "run_manifest.tsv").read_bytes()
    events_path = tmp_path / "events.jsonl"

    with pytest.raises(ValueError, match="Managed output"):
        adaptive_hparam.init_adaptive_workflow(recipe, workflow_dir)

    assert (tmp_path / "run_manifest.tsv").read_bytes() == manifest_before
    assert not events_path.exists()
    assert not (workflow_dir / "adaptive" / "workflow.json").exists()


def test_adaptive_init_rejects_symlink_root_before_writing(tmp_path: Path):
    source = tmp_path / "source"
    recipe = _adaptive_recipe(source)
    real_root = tmp_path / "real-workflow"
    alias_root = tmp_path / "workflow-alias"
    alias_root.symlink_to(real_root, target_is_directory=True)
    payload = yaml.safe_load(recipe.read_text())
    payload["experiment"]["root"] = str(real_root)
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))

    with pytest.raises(ValueError, match="experiment root must not be a symlink"):
        adaptive_hparam.init_adaptive_workflow(recipe, alias_root)

    assert alias_root.is_symlink()
    assert not real_root.exists()


def test_adaptive_workflow_root_drift_fails_before_suggestion_write(tmp_path: Path):
    recipe = _adaptive_recipe(tmp_path)
    workflow_dir = tmp_path / "workflow"
    assert _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir)).returncode == 0
    workflow_path = workflow_dir / "adaptive" / "workflow.json"
    workflow = json.loads(workflow_path.read_text())
    workflow["root"] = str(tmp_path / "other-workflow")
    workflow_path.write_text(json.dumps(workflow))
    events = (tmp_path / "events.jsonl").read_bytes()

    result = _run("hparam-suggest", "--workflow-dir", str(workflow_dir))

    assert result.returncode == 1
    assert "workflow root differs" in result.stderr
    assert not (workflow_dir / "adaptive" / "suggestions").exists()
    assert (tmp_path / "events.jsonl").read_bytes() == events


def test_adaptive_events_use_frozen_workspace_after_source_root_changes(tmp_path: Path):
    recipe = _adaptive_recipe(tmp_path)
    workflow_dir = adaptive_hparam.init_adaptive_workflow(recipe, tmp_path / "workflow")
    redirected_root = tmp_path / "redirected-workspace"
    payload = yaml.safe_load(recipe.read_text())
    payload["experiment"]["root"] = str(redirected_root)
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))

    adaptive_hparam._append_event(workflow_dir, "ownership_probe", {})

    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert events[-1]["event_type"] == "ownership_probe"
    assert not (redirected_root / "events.jsonl").exists()


def test_adaptive_suggest_rejects_source_contract_drift_before_writing(tmp_path: Path):
    recipe = _adaptive_recipe(tmp_path)
    workflow_dir = tmp_path / "workflow"
    assert _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir)).returncode == 0
    round_dir = workflow_dir / "adaptive" / "rounds" / "round_000"
    run = json.loads((round_dir / "plan.json").read_text())["runs"][0]
    digest = workflow_dir / "adaptive" / "digests" / "round_000.csv"
    manifests.write_rows(digest, [{**run, "test_auroc": 0.73}])
    payload = yaml.safe_load(recipe.read_text())
    payload["search"]["max_run"] = 1
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    events_before = (tmp_path / "events.jsonl").read_bytes()

    result = _run("hparam-suggest", "--workflow-dir", str(workflow_dir))

    assert result.returncode == 1
    assert "search.max_run" in result.stdout + result.stderr
    assert not (workflow_dir / "adaptive" / "suggestions").exists()
    assert (tmp_path / "events.jsonl").read_bytes() == events_before


def test_adaptive_relative_recipe_locator_fails_before_suggestion_write(tmp_path: Path, monkeypatch):
    recipe = _adaptive_recipe(tmp_path)
    workflow_dir = tmp_path / "workflow"
    assert _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir)).returncode == 0
    workflow_path = workflow_dir / "adaptive" / "workflow.json"
    workflow = json.loads(workflow_path.read_text())
    workflow["recipe_path"] = recipe.name
    workflow_path.write_text(json.dumps(workflow))
    before = {path.relative_to(workflow_dir): path.read_bytes() for path in workflow_dir.rglob("*") if path.is_file()}
    monkeypatch.chdir(recipe.parent)

    with pytest.raises(ValueError, match="recipe_path must be absolute"):
        adaptive_hparam.suggest_next_round(workflow_dir)

    assert {
        path.relative_to(workflow_dir): path.read_bytes() for path in workflow_dir.rglob("*") if path.is_file()
    } == before


def test_adaptive_legacy_registry_fails_before_monitor_or_digest_write(tmp_path: Path):
    recipe = _adaptive_recipe(tmp_path)
    workflow_dir = tmp_path / "workflow"
    assert _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir)).returncode == 0
    (workflow_dir / "adaptive" / "trial_registry.tsv").write_text("trial_id\ntrial_000\n")
    round_dir = workflow_dir / "adaptive" / "rounds" / "round_000"

    result = _run("hparam-digest", "--run-dir", str(workflow_dir))

    assert result.returncode == 1
    assert "Legacy adaptive registry is read-only" in result.stderr
    assert not (round_dir / "run_status.tsv").exists()
    assert not (workflow_dir / "adaptive" / "digests").exists()


def test_adaptive_registry_must_bind_current_round_before_monitor(tmp_path: Path):
    recipe = _adaptive_recipe(tmp_path)
    workflow_dir = tmp_path / "workflow"
    assert _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir)).returncode == 0
    registry = workflow_dir / "adaptive" / "run_registry.tsv"
    registry.write_text("step_id\trun_id\tround\tround_dir\n")
    round_dir = workflow_dir / "adaptive" / "rounds" / "round_000"

    result = _run("hparam-digest", "--run-dir", str(workflow_dir))

    assert result.returncode == 1
    assert "registry is missing the current plan run" in result.stderr
    assert not (round_dir / "run_status.tsv").exists()
    assert not (workflow_dir / "adaptive" / "digests").exists()


@pytest.mark.parametrize("registry_fault", ["foreign", "unmanaged", "config_drift"])
def test_adaptive_registry_ownership_fails_before_workflow_mutation(tmp_path: Path, registry_fault: str):
    recipe = _adaptive_recipe(tmp_path)
    workflow_dir = tmp_path / "workflow"
    assert _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir)).returncode == 0
    registry_path = workflow_dir / "adaptive" / "run_registry.tsv"
    registry = _read_table(registry_path)
    if registry_fault == "foreign":
        registry[0]["experiment_id"] = "foreign-experiment"
    elif registry_fault == "unmanaged":
        registry.append(
            {
                **registry[0],
                "experiment_id": "foreign-experiment",
                "step_id": "foreign-step",
                "run_id": "run-999",
                "version": "foreign-version",
            }
        )
    else:
        registry[0]["config"] = str(tmp_path / "other-config.yaml")
    manifests.write_rows(registry_path, registry)
    before = {path.relative_to(workflow_dir): path.read_bytes() for path in workflow_dir.rglob("*") if path.is_file()}

    with pytest.raises(ValueError, match="canonical manifest|Frozen run field differs"):
        adaptive_hparam._workflow(workflow_dir)

    assert {
        path.relative_to(workflow_dir): path.read_bytes() for path in workflow_dir.rglob("*") if path.is_file()
    } == before


def test_adaptive_registry_rejects_header_only_legacy_identity(tmp_path: Path):
    recipe = _adaptive_recipe(tmp_path)
    workflow_dir = tmp_path / "workflow"
    assert _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir)).returncode == 0
    registry_path = workflow_dir / "adaptive" / "run_registry.tsv"
    registry_path.write_text("trial_id\tround\n")

    with pytest.raises(ValueError, match="Historical trial_id fields"):
        adaptive_hparam._workflow(workflow_dir)

    assert registry_path.read_text() == "trial_id\tround\n"


def test_adaptive_stop_scan_ignores_header_only_legacy_projection(tmp_path: Path):
    recipe_path = _adaptive_recipe(tmp_path, max_rounds=1)
    workflow_dir = tmp_path / "workflow"
    assert _run("hparam-adaptive-init", "--recipe", str(recipe_path), "--output-dir", str(workflow_dir)).returncode == 0
    round_dir = workflow_dir / "adaptive" / "rounds" / "round_000"
    status_path = round_dir / "run_status.tsv"
    status_path.write_text("trial_id\tstatus\n")
    recipe = adaptive_hparam.load_recipe_with_base(recipe_path)

    adaptive_hparam._stop_bad_running_runs(workflow_dir, round_dir, recipe)

    assert status_path.read_text() == "trial_id\tstatus\n"
