"""Reading frozen registered plans and the artifacts a finished run leaves behind.

Domain-free reader. ``read_registered_plan`` checks only a published plan's
binding to its workspace, deliberately without consultation, recompilation,
config/path probes, or live observation -- reading a published plan must not
re-run the decisions that produced it, and frozen bytes are verified where they
are used. Also owns checkpoint-name parsing, metric extraction, and rank
assignment. Hparam plan registration is owned here too and, unlike plan reads,
does probe: ``preflight_hparam_plan`` checks a staged plan's candidate configs
and output paths through ``plan_hparam``, probes the execution target and
freezes that snapshot into the plan, and ``commit_hparam_plan`` registers the
published plan's run-manifest rows.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import math
import os
from pathlib import Path
import stat
import subprocess
from typing import Any, Iterator, TypedDict, cast

from . import execution_snapshot, experiment_io as exp_io, plan_contract, plan_hparam
from .decision_models import USER_DECISIONS_FILENAME
from .experiment_workspace import (
    append_event,
    ensure_experiment_workspace,
    experiment_root,
    file_sha256,
    managed_run_key,
    managed_run_parameters,
    merge_run_manifest,
    plan_registration_rows_state,
    read_managed_yaml_mapping,
    read_run_manifest,
    read_step_manifest,
    validated_run_key,
)
from .manifests import read_json, write_json, write_text
from .models import JsonValue


class RegisteredPlanSelection(TypedDict):
    metric: str
    mode: str
    split: str


class RegisteredPlanSummary(TypedDict):
    path: str
    task: str
    run_keys: list[tuple[str, str]]
    launch_script: str
    selection: RegisteredPlanSelection | None
    recipe: dict[str, Any]
    runs: list[dict[str, Any]]


class RegisteredPlanStep(TypedDict):
    manifest: dict[str, Any]
    plans: list[RegisteredPlanSummary]


def _resolved_recipe_view(recipe: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in recipe.items() if key != "_recipe_path"}


def _read_plan_documents(
    plan_dir: Path,
    *,
    require_resolved_sha256: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    plan_path, resolved_recipe_path = plan_contract.pass_plan_control_paths(plan_dir)
    if not plan_path.exists():
        raise FileNotFoundError(f"Missing hparam plan: {plan_path}")
    plan = read_json(plan_path)
    if not resolved_recipe_path.exists():
        raise FileNotFoundError(f"Missing frozen hparam recipe: {resolved_recipe_path}")
    resolved_recipe_bytes = resolved_recipe_path.read_bytes()
    if (
        require_resolved_sha256
        and plan.get("resolved_recipe_sha256") != hashlib.sha256(resolved_recipe_bytes).hexdigest()
    ):
        raise ValueError(f"Frozen hparam recipe SHA-256 is missing or changed: {resolved_recipe_path}")
    resolved_recipe = read_managed_yaml_mapping(
        resolved_recipe_bytes.decode(),
        source=f"Frozen registered recipe {resolved_recipe_path}",
    )
    return plan, resolved_recipe


def is_registered_blocked_plan(
    plan_dir: str | Path,
    *,
    workspace: str | Path,
) -> bool:
    """Recognize a blocked control bundle at a caller-supplied registered path.

    Requires absolute paths without '..', with plan_dir inside workspace; reads
    without changing artifacts. True means the expected
    blocked files are readable and the directory layout passes the bundle checks,
    not that this function looked up registration or validated question contents.
    False does not establish a valid PASS plan: missing blocked markers or a
    resolved recipe without plan.json also return False. plan.json coexisting
    with blocked markers raises, as do unsafe bundles and checked layout
    conflicts. Read failures propagate. Use the full reader for runnable plans.
    """
    plan_dir = Path(plan_dir)
    workspace = Path(workspace)
    # Reject corrupt registrations before probing paths outside the canonical workspace.
    if not workspace.is_absolute() or not plan_dir.is_absolute() or ".." in workspace.parts or ".." in plan_dir.parts:
        raise ValueError(f"Registered plan must use an absolute canonical workspace path: {plan_dir}")
    try:
        plan_dir.relative_to(workspace)
    except ValueError as exc:
        raise ValueError(f"Registered plan is outside its managed workspace: {plan_dir}") from exc
    plan_path = plan_dir / "plan.json"
    resolved_recipe_path = plan_dir / "recipe.resolved.yaml"
    user_decisions_path = plan_dir / USER_DECISIONS_FILENAME
    blocked_only_paths = plan_contract.blocked_plan_control_paths(plan_dir)
    # Existence checks may follow local aliases, so validate every possible control file's ancestry first.
    try:
        exp_io.validate_managed_output_paths(
            workspace,
            [plan_path, resolved_recipe_path, *blocked_only_paths],
        )
    except ValueError as exc:
        raise ValueError(f"Registered plan control bundle is missing or aliased: {plan_dir}") from exc
    plan_exists = exp_io.path_exists_at(plan_path)
    blocked_entries = [path for path in blocked_only_paths if exp_io.path_exists_at(path)]
    if plan_exists:
        if blocked_entries:
            raise ValueError(f"Registered plan contains both PASS and blocked planning artifacts: {plan_dir}")
        return False
    blocked_path = plan_dir / "plan.blocked.md"
    if not exp_io.path_exists_at(blocked_path) or exp_io.path_exists_at(resolved_recipe_path):
        return False
    if plan_dir == workspace:
        residue = [
            path
            for path in plan_contract.pass_plan_artifact_paths(plan_dir)
            if path not in {plan_path, resolved_recipe_path} and exp_io.path_exists_at(path)
        ]
        if residue:
            raise ValueError(
                f"Registered blocked plan contains PASS planning artifacts: {', '.join(map(str, residue))}"
            )
    blocked_files: list[str | Path] = [plan_dir / "questions.json", plan_dir / "questions.md", blocked_path]
    if user_decisions_path in blocked_entries:
        blocked_files.append(user_decisions_path)
    draft_path = plan_dir / "plan.draft.json"
    if draft_path in blocked_entries:
        blocked_files.append(draft_path)
    exp_io.read_managed_files_at(
        workspace,
        blocked_files,
        exact_directory_entries=plan_dir != workspace,
    )
    return True


def read_registered_plan(
    plan_dir: str | Path,
    *,
    workspace_experiment: dict[str, Any],
    step_manifest: dict[str, Any],
    workspace_rows: list[dict[str, Any]],
) -> RegisteredPlanSummary:
    """Read a registered plan and check its binding to caller-supplied workspace state.

    workspace_experiment, step_manifest and workspace_rows are the caller's
    canonical snapshots; this function checks the plan against them rather than
    refreshing those snapshots. Checks that the step registers the plan, that
    plan.json's recipe equals recipe.resolved.yaml (and, for an hparam plan, its
    frozen SHA-256), that the recipe's experiment and step metadata match the
    workspace and step, and that run_manifest.tsv holds every plan run with the plan's
    run parameters.

    Frozen configs and scripts are not re-read or recompiled here: the generated
    scripts verify their inputs at run time and the launch paths verify what they
    start or submit. Returns path, task, run_keys, launch_script, selection,
    recipe and runs, rather than the original plan.json mapping. Does not update
    lifecycle state or launch runs. Binding violations raise ValueError;
    read/parse failures propagate."""
    plan_dir = Path(plan_dir)
    if str(plan_dir) not in [str(path) for path in step_manifest["plans"]]:
        raise ValueError(f"Plan is not registered by its managed step: {plan_dir}")
    plan, resolved_recipe = _read_plan_documents(plan_dir)
    resolved_recipe_path = plan_dir / "recipe.resolved.yaml"
    recipe = plan.get("recipe")
    if not isinstance(recipe, dict) or _resolved_recipe_view(recipe) != resolved_recipe:
        raise ValueError(f"Registered plan recipe differs from recipe.resolved.yaml: {resolved_recipe_path}")
    task = str(recipe.get("task") or "")
    if task == "hparam_tune" and plan.get("resolved_recipe_sha256") != file_sha256(resolved_recipe_path):
        raise ValueError(f"Frozen hparam recipe SHA-256 is missing or changed: {resolved_recipe_path}")
    expected_experiment = {
        field: workspace_experiment.get(field) for field in ("id", "title", "objective", "root", "baseline")
    }
    if recipe.get("experiment") != expected_experiment:
        raise ValueError(f"Registered plan experiment metadata differs from the managed workspace: {plan_dir}")
    if recipe.get("step") != {field: step_manifest["step"][field] for field in ("id", "phase", "purpose")}:
        raise ValueError(f"Registered plan step metadata differs from its managed step: {plan_dir}")

    runs = plan.get("runs")
    if not isinstance(runs, list) or any(not isinstance(run, dict) for run in runs):
        raise ValueError(f"Registered plan runs must be a list of mappings: {plan_dir / 'plan.json'}")
    plan_keys = [validated_run_key(run) for run in runs]
    workspace_by_key = {managed_run_key(row): row for row in workspace_rows}
    for (step_id, run_id), run in zip(plan_keys, runs):
        row = workspace_by_key.get((step_id, run_id))
        if row is None:
            raise ValueError(f"Workspace run_manifest.tsv is missing registered plan run: {step_id} / {run_id}")
        # Selection reports read run parameters from run_manifest.tsv, so they must match the plan.
        # Empty cells are other steps' parameter columns in the shared manifest header.
        if _nonempty_text_parameters(row) != _nonempty_text_parameters(run):
            raise ValueError(f"Workspace run parameters differ from plan: {step_id} / {run_id}")

    selection: RegisteredPlanSelection | None = None
    if task == "hparam_tune":
        evaluation = recipe["evaluation_policy"] if isinstance(recipe.get("evaluation_policy"), dict) else {}
        selection = {
            "metric": str(evaluation.get("selection_metric") or ""),
            "mode": str(evaluation.get("selection_mode") or ""),
            "split": str(evaluation.get("selection_split") or ""),
        }
    return {
        "path": str(plan_dir),
        "task": task,
        "run_keys": plan_keys,
        "launch_script": str(plan_dir / ("run_all.sh" if task == "hparam_tune" else "run.sh")),
        "selection": selection,
        "recipe": recipe,
        "runs": runs,
    }


def _nonempty_text_parameters(row: Mapping[str, object]) -> dict[str, str]:
    # run_manifest.tsv stores each parameter as str(value), so compare plan values in that form.
    return {field: str(value) for field, value in managed_run_parameters(row).items() if value not in (None, "")}


def hparam_plan_workspace(plan_dir: Path) -> Path:
    """Return the experiment root that plan.json names, to pick the run lock before the locked read.

    Reads only plan.json and validates nothing else: every value a caller acts on comes from
    read_hparam_plan under that lock."""
    plan_path = plan_dir / "plan.json"
    if not plan_path.exists():
        raise FileNotFoundError(f"Missing hparam plan: {plan_path}")
    recipe = read_json(plan_path).get("recipe")
    workspace = experiment_root(recipe) if isinstance(recipe, dict) else None
    if workspace is None:
        raise ValueError("Invalid hparam workspace binding: experiment.root is required.")
    return workspace


def read_hparam_plan(run_dir: Path, *, require_workspace_state: bool = True) -> plan_contract.HparamPlan:
    """Read a local hparam plan and return the full plan.json mapping.

    Checks the recorded recipe.resolved.yaml digest and that plan.json's recipe equals it, since
    launch commands are rebuilt from that recipe's execution block. By default it also checks the
    canonical workspace state: the plan's managed step registers this directory and
    run_manifest.tsv holds every plan run. Callers hold the workspace run lock for that read;
    require_workspace_state=False skips it for a plan that is not registered yet.

    Frozen configs, scripts and input snapshots are verified where they are used, by launch.sh,
    job.sbatch and the launch command's runtime_identity check, not here. Reads artifacts only.
    Invalid contracts raise ValueError; file and parsing failures propagate."""
    plan_path = run_dir / "plan.json"
    plan, resolved_recipe = _read_plan_documents(run_dir, require_resolved_sha256=True)
    recipe = plan.get("recipe")
    if not isinstance(recipe, dict) or _resolved_recipe_view(recipe) != resolved_recipe:
        raise ValueError(f"Hparam plan recipe differs from recipe.resolved.yaml: {run_dir / 'recipe.resolved.yaml'}")
    runs = plan.get("runs")
    if not isinstance(runs, list) or not runs or any(not isinstance(run, dict) for run in runs):
        raise ValueError(f"Hparam plan must define a non-empty runs list of mappings: {plan_path}")
    if require_workspace_state:
        workspace = experiment_root(recipe)
        if workspace is None:
            raise ValueError("Invalid hparam workspace binding: experiment.root is required.")
        step_manifest = read_step_manifest(workspace, str(recipe["step"]["id"]))
        assert step_manifest is not None  # Missing steps raise unless allow_missing is explicitly enabled.
        if str(run_dir.resolve()) not in [str(path) for path in step_manifest.get("plans") or []]:
            raise ValueError(f"Hparam plan is not registered by its managed step: {run_dir}")
        manifest_keys = {managed_run_key(row) for row in read_run_manifest(workspace)}
        missing = [run for run in runs if managed_run_key(run) not in manifest_keys]
        if missing:
            labels = ", ".join(f"{run.get('step_id')} / {run.get('run_id')}" for run in missing)
            raise ValueError(f"Workspace run_manifest.tsv is missing plan runs: {labels}")
    # The checks above establish recipe, run rows, and the resolved recipe digest without replacing the mapping.
    return cast(plan_contract.HparamPlan, plan)


def _physical_plan_path(path: Path, semantic_dir: Path, physical_dir: Path) -> Path:
    try:
        relative = path.relative_to(semantic_dir)
    except ValueError:
        return path
    return physical_dir / relative


class HparamRegistrationPreflightError(ValueError):
    pass


def preflight_hparam_plan(physical_out: str | Path, *, semantic_out: str | Path) -> str:
    physical_dir = Path(physical_out)
    plan_dir = Path(semantic_out)
    plan = read_hparam_plan(physical_dir, require_workspace_state=False)
    recipe = plan["recipe"]
    run_configs = [
        (
            run,
            _physical_plan_path(Path(str(run["config"])), plan_dir, physical_dir).read_bytes(),
        )
        for run in plan["runs"]
    ]
    plan_hparam.validate_hparam_run_configs(recipe, run_configs)
    _hparam_registration_state(plan)
    plan_hparam.validate_hparam_output_paths(plan_dir, plan)
    execution_value = recipe.get("execution")
    execution = execution_value if isinstance(execution_value, dict) else {}
    validation_runs = []
    for run in plan["runs"]:
        validation_run = dict(run)
        validation_run["script"] = str(_physical_plan_path(Path(str(run["script"])), plan_dir, physical_dir))
        validation_runs.append(validation_run)
    snapshot = _inspect_hparam_execution_target(execution, validation_runs)
    snapshot_path = physical_dir / execution_snapshot.EXECUTION_SNAPSHOT_NAME
    execution_snapshot.write_execution_snapshot_file(snapshot_path, snapshot)
    card = plan_hparam.render_hparam_preflight_card(recipe, snapshot, run_configs)
    plan_markdown = physical_dir / "plan.md"
    write_text(plan_markdown, f"{plan_markdown.read_text().rstrip()}\n\n{card}\n")
    plan_payload = read_json(physical_dir / "plan.json")
    plan_payload["execution_snapshot"] = {
        "path": str(plan_dir / execution_snapshot.EXECUTION_SNAPSHOT_NAME),
        "sha256": file_sha256(snapshot_path),
    }
    # Keep plan.json as the terminal physical-plan manifest after adding the frozen target evidence.
    write_json(physical_dir / "plan.json", plan_payload)
    return card


def _inspect_hparam_execution_target(
    execution: Mapping[str, JsonValue], runs: Sequence[Mapping[str, JsonValue]]
) -> execution_snapshot.ExecutionSnapshot:
    return execution_snapshot.inspect_execution_target(execution, runs, plan_label="hparam")


def commit_hparam_plan(out: str | Path, *, emit_event: bool = True) -> plan_contract.HparamPlan:
    plan_dir = Path(out).expanduser()
    if not plan_dir.is_absolute():
        plan_dir = plan_dir.resolve()
    try:
        plan = read_hparam_plan(plan_dir, require_workspace_state=False)
        if "execution_snapshot" not in plan:
            raise ValueError(f"Hparam plan lacks registration preflight evidence: {plan_dir}")
        recipe = plan["recipe"]
        ensure_experiment_workspace(
            recipe,
            plan_dir,
            validate_only=True,
            allow_published_plan=True,
        )
        root, manifest_rows = _hparam_registration_state(plan)
    except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as exc:
        raise HparamRegistrationPreflightError(str(exc)) from exc
    ensure_experiment_workspace(recipe, plan_dir, allow_published_plan=True)
    merge_run_manifest(root, manifest_rows)
    if emit_event:
        append_event(
            root,
            "plan_created",
            {
                "step_id": (recipe.get("step") or {}).get("id"),
                "plan_dir": str(plan_dir),
                "run_count": len(manifest_rows),
            },
        )
    return plan


def _hparam_registration_state(plan: Mapping[str, Any]) -> tuple[Path, list[dict[str, Any]]]:
    recipe_value = plan.get("recipe")
    recipe = recipe_value if isinstance(recipe_value, dict) else {}
    root = experiment_root(recipe)
    if root is None:
        raise ValueError("experiment.root is required.")
    manifest_rows = plan_hparam.hparam_manifest_rows(plan)
    plan_registration_rows_state(root, manifest_rows, source="Canonical hparam plan")
    return root, manifest_rows


def plan_tree_sha256(root: Path, *, top_level_entries: frozenset[str] | None = None) -> str:
    """Hash a local plan tree's relative paths, permission modes, types and bytes.

    Includes file sizes and directory entries, but not the root's own name/mode
    or timestamps. top_level_entries restricts traversal to caller-supplied
    top-level names and their descendants; absent selected entries are skipped.
    The default covers all descendants. Rejects a missing or symlink root and
    encountered entries other than regular files/directories with ValueError.
    Reads without locking or taking a snapshot; callers own synchronization and
    the comparison scope. Filesystem errors propagate. A matching digest alone
    does not validate plan semantics or workspace registration.
    """
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"Managed plan is missing or aliased: {root}")
    digest = hashlib.sha256()
    if top_level_entries is None:
        paths = list(root.rglob("*"))
    else:
        paths = []
        for name in top_level_entries:
            entry = root / name
            if not os.path.lexists(entry):
                continue
            paths.append(entry)
            if entry.is_dir() and not entry.is_symlink():
                paths.extend(entry.rglob("*"))
    for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix()):
        info = os.lstat(path)
        relative = path.relative_to(root).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(stat.S_IMODE(info.st_mode).to_bytes(4, "big"))
        if stat.S_ISDIR(info.st_mode):
            digest.update(b"directory")
            continue
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(f"Managed plan contains an unsafe artifact: {path}")
        digest.update(b"file")
        digest.update(info.st_size.to_bytes(8, "big"))
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def iter_registered_hparam_plans(
    workspace: Path,
    step_id: str,
    *,
    selection_metric: Any,
    selection_mode: Any,
    selection_split: Any,
) -> Iterator[tuple[Path, plan_contract.HparamPlan]]:
    """Yield validated local hparam plans in the managed step's registered order.

    Reads the step manifest when iterated, skips recognized blocked bundles and
    plans whose frozen task is not hparam_tune, and validates hparam plans via
    read_hparam_plan with its default workspace checks. Each yielded
    (directory, full plan mapping) must match step_id and all three supplied
    selection values exactly. Missing/corrupt artifacts and mismatches raise
    during iteration rather than being skipped; earlier yields remain possible.
    Does not discover unregistered directories or refresh job/lifecycle state.
    """
    step_manifest = read_step_manifest(workspace, step_id)
    assert step_manifest is not None  # The default reader raises for a missing managed step.
    for registered_plan_dir in step_manifest["plans"]:
        registered_root = Path(str(registered_plan_dir))
        registered_plan_path = registered_root / "plan.json"
        resolved_recipe_path = registered_root / "recipe.resolved.yaml"
        if is_registered_blocked_plan(registered_root, workspace=workspace):
            continue
        if not registered_plan_path.exists():
            raise FileNotFoundError(f"Registered plan is missing plan.json: {registered_plan_path}")
        raw_plan = read_json(registered_plan_path)
        registered_recipe = raw_plan["recipe"] if isinstance(raw_plan.get("recipe"), dict) else {}
        resolved_recipe = read_managed_yaml_mapping(
            resolved_recipe_path.read_text(),
            source=f"Frozen registered recipe {resolved_recipe_path}",
        )
        if registered_recipe.get("task") != resolved_recipe.get("task"):
            raise ValueError(f"Registered plan task differs from recipe.resolved.yaml: {registered_root}")
        if resolved_recipe.get("task") != "hparam_tune":
            continue
        registered_plan = read_hparam_plan(registered_root)
        registered_recipe = registered_plan["recipe"] if isinstance(registered_plan.get("recipe"), dict) else {}
        registered_step = registered_recipe["step"] if isinstance(registered_recipe.get("step"), dict) else {}
        if str(registered_step.get("id") or "") != step_id:
            raise ValueError(f"Registered hparam plan belongs to a different step: {registered_root}")
        registered_evaluation = (
            registered_recipe["evaluation_policy"]
            if isinstance(registered_recipe.get("evaluation_policy"), dict)
            else {}
        )
        if registered_evaluation.get("selection_metric") != selection_metric:
            raise ValueError("Existing ranking selection metric differs from the current recipe.")
        if registered_evaluation.get("selection_mode") != selection_mode:
            raise ValueError("Existing ranking selection mode differs from the current recipe.")
        # The invoking split governs evidence interpretation, so registered aggregation must be homogeneous.
        if registered_evaluation.get("selection_split") != selection_split:
            raise ValueError("Existing ranking selection split differs from the current recipe.")
        yield registered_root, registered_plan


def metric_value(manifest: dict[str, Any], metric: str) -> float | str:
    """Extract a metric without coercing or validating the stored value.

    An existing metrics[metric] entry wins, even if empty or None. Otherwise uses
    best_model_score only when monitor matches metric and the score is not None;
    returns an empty string if neither source applies. Callers own numeric and
    finiteness checks; this lookup does not establish valid evaluation evidence.
    """
    metrics = manifest["metrics"] if isinstance(manifest.get("metrics"), dict) else {}
    if metric in metrics:
        return metrics[metric]
    if manifest.get("monitor") == metric and manifest.get("best_model_score") is not None:
        return manifest["best_model_score"]
    return ""


def fixed_checkpoint_path_from_names(
    manifest: dict[str, Any], checkpoint_dir: str | Path, checkpoint_names: list[str]
) -> str:
    """Resolve checkpoint hints using a supplied inventory, without filesystem I/O.

    checkpoint_names must contain basenames from the caller's inventory. Uses
    best_model_path before checkpoint_path, prefers an inventoried epoch= name,
    and requires agreement with a supplied valid manifest epoch. A best-epoch=
    name is eligible when the fixed name is unavailable and either its epoch
    matches or a completed or skipped-test manifest without an epoch names it;
    otherwise epoch fallback chooses the first matching fixed name in sorted order.
    A completed or skipped-test manifest without an epoch may explicitly bind
    inventoried best.ckpt; callers must freeze its digest before selection.
    Returns an empty string for an absent/empty checkpoint_dir value, invalid
    authored epoch or unresolved hint. The path only reflects the inventory:
    callers own its freshness and checks of file existence, aliases, contents
    and ownership.
    """
    if checkpoint_dir in (None, ""):
        return ""
    checkpoint_dir = Path(str(checkpoint_dir))
    names = {str(name) for name in checkpoint_names}
    raw_epoch = manifest.get("epoch")
    manifest_epoch = epoch_number(raw_epoch)
    if raw_epoch not in (None, "") and manifest_epoch is None:
        return ""
    raw = manifest.get("best_model_path") or manifest.get("checkpoint_path") or ""
    if raw:
        raw_name = Path(str(raw)).name
        if (
            raw_name == "best.ckpt"
            and manifest_epoch is None
            and manifest.get("status") in {"completed", "skipped_test"}
        ):
            return str(checkpoint_dir / raw_name) if raw_name in names else ""
        name = raw_name
        if raw_name.startswith("best-epoch="):
            name = name.removeprefix("best-")
        if (
            name.startswith("epoch=")
            and name in names
            and (manifest_epoch is None or epoch_number_from_checkpoint_name(name) == manifest_epoch)
        ):
            return str(checkpoint_dir / name)
        if (
            raw_name.startswith("best-epoch=")
            and raw_name in names
            and (
                epoch_number_from_checkpoint_name(name) == manifest_epoch
                if manifest_epoch is not None
                else manifest.get("status") in {"completed", "skipped_test"}
            )
        ):
            return str(checkpoint_dir / raw_name)
        if manifest_epoch is None:
            return ""
        for candidate in sorted(names):
            if candidate.startswith("epoch=") and epoch_number_from_checkpoint_name(candidate) == manifest_epoch:
                return str(checkpoint_dir / candidate)
        return ""
    if manifest_epoch is not None:
        for candidate in sorted(names):
            if candidate.startswith("epoch=") and epoch_number_from_checkpoint_name(candidate) == manifest_epoch:
                return str(checkpoint_dir / candidate)
        return ""
    return ""


def epoch_from_checkpoint_name(name: str) -> str:
    if not name.startswith("epoch="):
        return ""
    return name.split("=", 1)[1].split("-", 1)[0].split(".", 1)[0]


def epoch_number(value: Any) -> int | None:
    if isinstance(value, bool) or value in (None, ""):
        return None
    try:
        number = float(str(value))
    except ValueError:
        return None
    if not math.isfinite(number) or not number.is_integer():
        return None
    return int(number)


def epoch_number_from_checkpoint_name(name: str) -> int | None:
    return epoch_number(epoch_from_checkpoint_name(name))


def float_or_none(value: Any) -> float | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return score if math.isfinite(score) else None


def assign_ranks(rows: list[dict[str, Any]], *, key: str, reverse: bool) -> list[dict[str, Any]]:
    """Sort rows by their finite metric value and write 1-based ranks in place.

    Sorting is stable (ties keep their input order). The input list is not
    reordered; the returned list is the sorted view."""
    ranked = sorted(rows, key=lambda row: float(row[key]), reverse=reverse)
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
    return ranked
