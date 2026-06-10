from __future__ import annotations

import copy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
from typing import Any

import yaml

from .hparam import (
    _find_run_manifest,
    _fixed_checkpoint_path,
    _log_has_failure,
    _log_tail,
    _metric_value,
    _now,
    _read_json,
    _read_plan,
    _read_rows,
    _write_rows,
    launch_hparam_trials,
    monitor_hparam_trials,
    stop_hparam_trial,
)
from .manifests import write_json, write_text
from .models import resolve_repo_path
from .plans import build_plan
from .recipes import load_recipe_with_base, load_yaml_file, recipe_name


def init_adaptive_workflow(recipe_path: str | Path, output_dir: str | Path) -> Path:
    root = Path(output_dir)
    recipe_path = Path(recipe_path)
    recipe = load_recipe_with_base(recipe_path)
    _validate_adaptive_recipe(recipe)
    adaptive_dir = root / "adaptive"
    round_dir = adaptive_dir / "rounds" / "round_000"
    round_recipe = _write_round_recipe(recipe, recipe_path, round_dir, 0)
    report = build_plan(recipe_path=round_recipe, output_dir=round_dir)
    if report.exit_code != 0:
        raise RuntimeError(f"Round 000 plan failed with exit code {report.exit_code}.")
    workflow = {
        "schema_version": 1,
        "recipe_path": str(recipe_path),
        "root": str(root),
        "external_optimized": True,
        "objective_metric": _adaptive(recipe).get("objective_metric", "test_auroc"),
        "objective_mode": _adaptive(recipe).get("objective_mode", "max"),
    }
    write_json(adaptive_dir / "workflow.json", workflow)
    _append_event(root, "adaptive_init", {"round": 0, "recipe_path": str(recipe_path), "round_dir": str(round_dir)})
    _append_registry_rows(root, 0, round_dir)
    write_text(adaptive_dir / "README.md", _adaptive_readme(workflow))
    return root


def digest_hparam_run(run_dir: str | Path) -> Path:
    root = Path(run_dir)
    workflow_root, round_dir, round_index = _resolve_workflow_round(root)
    plan = _read_plan(round_dir)
    recipe = plan.get("recipe") if isinstance(plan.get("recipe"), dict) else {}
    monitor_hparam_trials(round_dir)
    status_rows = {row.get("trial_id"): row for row in _read_rows(round_dir / "trial_status.tsv")}
    rows = []
    for trial in plan.get("trials", []):
        trial_id = str(trial.get("trial_id"))
        version = str(trial.get("version") or f"{recipe.get('name')}-{trial_id}")
        manifest_path = _find_run_manifest(round_dir, version, recipe)
        manifest = _read_json(manifest_path) if manifest_path else {}
        status = status_rows.get(trial_id, {})
        row = {
            "round": round_index,
            "trial_id": trial_id,
            "version": version,
            "external_optimized": True,
            "status": status.get("status", ""),
            "pid": status.get("pid", ""),
            "config": trial.get("config", ""),
            "checkpoint_path": _fixed_checkpoint_path(manifest, manifest_path),
            "run_manifest": str(manifest_path or ""),
            "log_path": status.get("log_path", ""),
            "log_failed": _log_has_failure(status.get("log_path")),
            "log_tail": _log_tail(status.get("log_path"), lines=4),
        }
        row.update(_trial_params(trial))
        row.update(_manifest_metrics(manifest))
        rows.append(row)
    out_dir = workflow_root / "adaptive" / "digests"
    out = out_dir / f"round_{round_index:03d}.csv"
    _write_rows(out, rows)
    write_text(out_dir / f"round_{round_index:03d}.md", _digest_markdown(rows, _objective(workflow_root, recipe)))
    _append_event(workflow_root, "digest", {"round": round_index, "path": str(out), "rows": len(rows)})
    _write_incumbent(workflow_root, rows, _objective(workflow_root, recipe), round_index)
    return out


def suggest_next_round(workflow_dir: str | Path) -> Path:
    root = Path(workflow_dir)
    workflow = _workflow(root)
    recipe = load_recipe_with_base(workflow["recipe_path"])
    current_round = _latest_round_index(root)
    next_round = current_round + 1
    digest = _latest_digest(root)
    rows = _read_rows(digest)
    objective = _objective(root, recipe)
    ranked = _rank_rows(rows, objective)
    if not ranked:
        _append_event(root, "suggest_blocked", {"round": next_round, "reason": "no_scored_trials"})
        raise ValueError(f"No digest rows with finite {objective['metric']} are available for suggestion.")
    best = ranked[0]
    suggested = copy.deepcopy(recipe)
    suggested["name"] = f"{recipe_name(recipe)}_adaptive_round_{next_round:03d}"
    suggested.setdefault("search", {})["parameters"] = _suggest_parameters(recipe, ranked)
    suggested["search"]["max_trials"] = int(_adaptive(recipe).get("round_size") or _hparam_count(suggested))
    if suggested.get("base_recipe"):
        suggested["base_recipe"] = str(_resolve_base_recipe(workflow["recipe_path"], suggested["base_recipe"]))
    suggested.setdefault("artifacts", {}).pop("generated_config_dir", None)
    out_dir = root / "adaptive" / "suggestions"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"round_{next_round:03d}.yaml"
    out.write_text(yaml.safe_dump(_strip_internal_recipe_keys(suggested), sort_keys=False))
    rationale = _suggestion_rationale(next_round, objective, best, suggested["search"]["parameters"])
    write_text(out_dir / f"round_{next_round:03d}.md", rationale)
    _append_event(root, "suggest", {"round": next_round, "path": str(out), "best_trial": best.get("trial_id")})
    return out


def adaptive_step(workflow_dir: str | Path, *, execute: bool = False) -> Path:
    root = Path(workflow_dir)
    workflow = _workflow(root)
    recipe = load_recipe_with_base(workflow["recipe_path"])
    current_round = _latest_round_index(root)
    round_dir = _round_dir(root, current_round)
    digest = digest_hparam_run(round_dir)
    if execute:
        _stop_bad_running_trials(root, round_dir, recipe)
    suggestion = suggest_next_round(root)
    _supersede_pending_trials(root, round_dir)
    if execute and not _budget_exhausted(root, recipe):
        next_round = current_round + 1
        next_dir = _round_dir(root, next_round)
        round_recipe = _write_round_recipe(load_recipe_with_base(suggestion), suggestion, next_dir, next_round)
        report = build_plan(recipe_path=round_recipe, output_dir=next_dir)
        if report.exit_code != 0:
            raise RuntimeError(f"Round {next_round:03d} plan failed with exit code {report.exit_code}.")
        _append_registry_rows(root, next_round, next_dir)
        launch_hparam_trials(next_dir, dry_run=False)
        _append_event(root, "launch_round", {"round": next_round, "round_dir": str(next_dir)})
    else:
        _append_event(
            root,
            "adaptive_step_dry_run",
            {"round": current_round, "digest": str(digest), "suggestion": str(suggestion)},
        )
    return suggestion


def adaptive_loop(workflow_dir: str | Path, *, execute: bool = False) -> Path:
    root = Path(workflow_dir)
    recipe = load_recipe_with_base(_workflow(root)["recipe_path"])
    last = root
    while not _budget_exhausted(root, recipe):
        last = adaptive_step(root, execute=execute)
        if not execute:
            break
        time.sleep(float(_adaptive(recipe).get("poll_seconds") or 60))
    _append_event(root, "adaptive_loop_done", {"path": str(last)})
    return Path(last)


def _validate_adaptive_recipe(recipe: dict[str, Any]) -> None:
    adaptive = _adaptive(recipe)
    if not adaptive.get("enabled"):
        raise ValueError("adaptive.enabled must be true for adaptive workflow.")
    objective = str(adaptive.get("objective_metric") or "test_auroc")
    uses_external = objective.startswith("test_") or objective.startswith("external_")
    if uses_external and adaptive.get("test_feedback_for_selection") is not True:
        raise ValueError("adaptive.test_feedback_for_selection=true is required for test/external objectives.")


def _adaptive(recipe: dict[str, Any]) -> dict[str, Any]:
    return recipe.get("adaptive") if isinstance(recipe.get("adaptive"), dict) else {}


def _objective(root: Path, recipe: dict[str, Any]) -> dict[str, str]:
    workflow = _workflow(root) if (root / "adaptive" / "workflow.json").exists() else {}
    adaptive = _adaptive(recipe)
    return {
        "metric": str(workflow.get("objective_metric") or adaptive.get("objective_metric") or "test_auroc"),
        "mode": str(workflow.get("objective_mode") or adaptive.get("objective_mode") or "max"),
    }


def _workflow(root: Path) -> dict[str, Any]:
    path = root / "adaptive" / "workflow.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing adaptive workflow: {path}")
    return json.loads(path.read_text())


def _write_round_recipe(
    recipe: dict[str, Any], source_recipe_path: str | Path, round_dir: Path, round_index: int
) -> Path:
    round_dir.mkdir(parents=True, exist_ok=True)
    copied = _strip_internal_recipe_keys(copy.deepcopy(recipe))
    if copied.get("base_recipe"):
        copied["base_recipe"] = str(_resolve_base_recipe(source_recipe_path, copied["base_recipe"]))
    copied.setdefault("artifacts", {})["generated_config_dir"] = str(round_dir / "configs")
    copied["name"] = f"{recipe_name(recipe)}-round-{round_index:03d}"
    target = round_dir / "round_recipe.yaml"
    target.write_text(yaml.safe_dump(copied, sort_keys=False))
    return target


def _resolve_base_recipe(recipe_path: str | Path, base_recipe: str | Path) -> Path:
    raw = Path(base_recipe).expanduser()
    if raw.is_absolute():
        return raw
    source = Path(recipe_path)
    if source.exists():
        candidate = source.parent / raw
        if candidate.exists():
            return candidate.resolve()
    resolved = resolve_repo_path(raw)
    return resolved.resolve() if resolved and resolved.exists() else raw


def _strip_internal_recipe_keys(recipe: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in recipe.items() if not str(key).startswith("_")}


def _append_event(root: Path, event_type: str, payload: dict[str, Any]) -> None:
    path = root / "adaptive" / "events.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"time": _now(), "event_type": event_type, **payload}
    with path.open("a") as file_obj:
        file_obj.write(json.dumps(row, sort_keys=True) + "\n")


def _append_registry_rows(root: Path, round_index: int, round_dir: Path) -> None:
    path = root / "adaptive" / "trial_registry.tsv"
    plan = _read_plan(round_dir)
    rows = _read_rows(path) if path.exists() else []
    for trial in plan.get("trials", []):
        rows.append(
            {
                "round": round_index,
                "trial_id": trial.get("trial_id"),
                "version": trial.get("version"),
                "config": trial.get("config"),
                "script": str(round_dir / str(trial.get("script"))),
                "round_dir": str(round_dir),
                "registered_at": _now(),
            }
        )
    _write_rows(path, rows)


def _resolve_workflow_round(path: Path) -> tuple[Path, Path, int]:
    if (path / "adaptive" / "workflow.json").exists():
        idx = _latest_round_index(path)
        return path, _round_dir(path, idx), idx
    parts = path.parts
    if "rounds" in parts:
        idx = int(path.name.split("_")[-1])
        workflow_root = Path(*parts[: parts.index("adaptive")]) if "adaptive" in parts else path
        return workflow_root, path, idx
    return path, path, 0


def _latest_round_index(root: Path) -> int:
    rounds = sorted((root / "adaptive" / "rounds").glob("round_*"))
    if not rounds:
        return 0
    return int(rounds[-1].name.split("_")[-1])


def _round_dir(root: Path, index: int) -> Path:
    return root / "adaptive" / "rounds" / f"round_{index:03d}"


def _trial_params(trial: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in trial.items() if key.startswith("runtime.") or key.startswith("yaml:/")}


def _manifest_metrics(manifest: dict[str, Any]) -> dict[str, Any]:
    metrics = manifest.get("metrics") if isinstance(manifest.get("metrics"), dict) else {}
    row = {key: value for key, value in metrics.items() if isinstance(key, str)}
    for key in ("best_model_score", "epoch", "monitor", "monitor_mode", "status"):
        if manifest.get(key) is not None:
            row[key] = manifest.get(key)
    return row


def _digest_markdown(rows: list[dict[str, Any]], objective: dict[str, str]) -> str:
    ranked = _rank_rows(rows, objective)
    lines = [
        "# Adaptive Hparam Digest",
        "",
        f"external_optimized: true",
        f"objective: {objective['metric']} ({objective['mode']})",
        "",
        "## Top trials",
        "",
    ]
    for row in ranked[:5]:
        lines.append(
            f"- {row.get('trial_id')}: {objective['metric']}={row.get(objective['metric'], '')} "
            f"status={row.get('status', '')} checkpoint={row.get('checkpoint_path', '')}"
        )
    return "\n".join(lines) + "\n"


def _write_incumbent(root: Path, rows: list[dict[str, Any]], objective: dict[str, str], round_index: int) -> None:
    ranked = _rank_rows(rows, objective)
    if not ranked:
        return
    best = ranked[0]
    path = root / "adaptive" / "incumbents.tsv"
    incumbents = _read_rows(path) if path.exists() else []
    incumbents.append(
        {
            "round": round_index,
            "trial_id": best.get("trial_id", ""),
            "version": best.get("version", ""),
            "objective_metric": objective["metric"],
            "objective_mode": objective["mode"],
            "objective_score": best.get(objective["metric"], ""),
            "checkpoint_path": best.get("checkpoint_path", ""),
            "external_optimized": True,
            "selected_at": _now(),
        }
    )
    _write_rows(path, incumbents)


def _rank_rows(rows: list[dict[str, Any]], objective: dict[str, str]) -> list[dict[str, Any]]:
    reverse = objective["mode"] == "max"

    def score(row: dict[str, Any]) -> float | None:
        try:
            value = float(row.get(objective["metric"], ""))
        except (TypeError, ValueError):
            return None
        return value if math.isfinite(value) else None

    scored = [(value, row) for row in rows if (value := score(row)) is not None]
    return [row for value, row in sorted(scored, key=lambda item: item[0], reverse=reverse)]


def _latest_digest(root: Path) -> Path:
    digests = sorted((root / "adaptive" / "digests").glob("round_*.csv"))
    if not digests:
        raise FileNotFoundError("No adaptive digest exists. Run hparam-digest first.")
    return digests[-1]


def _suggest_parameters(recipe: dict[str, Any], ranked: list[dict[str, Any]]) -> dict[str, list[Any]]:
    params = (recipe.get("search") or {}).get("parameters") or {}
    best = ranked[0]
    top = ranked[:3]
    suggested: dict[str, list[Any]] = {}
    for key, values in params.items():
        if key not in best:
            suggested[key] = values
            continue
        value = _coerce_like(best[key], values[0] if values else best[key])
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            suggested[key] = _numeric_neighbors(value)
        else:
            seen = [row[key] for row in top if row.get(key) not in (None, "")]
            suggested[key] = list(dict.fromkeys([value, *seen]))[:3]
    return suggested


def _coerce_like(value: Any, example: Any) -> Any:
    if isinstance(example, bool):
        return str(value).lower() in {"1", "true", "yes"}
    if isinstance(example, int) and not isinstance(example, bool):
        return int(float(value))
    if isinstance(example, float):
        return float(value)
    return value


def _numeric_neighbors(value: int | float) -> list[int | float]:
    if isinstance(value, int):
        return sorted(set([max(1, value - 1), value, value + 1]))
    if value == 0:
        return [0.0, 1e-6, 3e-6]
    return sorted(set([float(f"{value * 0.5:.6g}"), float(f"{value:.6g}"), float(f"{value * 1.5:.6g}")]))


def _hparam_count(recipe: dict[str, Any]) -> int:
    params = (recipe.get("search") or {}).get("parameters") or {}
    count = 1
    for choices in params.values():
        count *= len(choices)
    return count


def _suggestion_rationale(
    round_index: int, objective: dict[str, str], best: dict[str, Any], params: dict[str, list[Any]]
) -> str:
    lines = [
        f"# Adaptive Suggestion Round {round_index:03d}",
        "",
        "external_optimized: true",
        f"objective: {objective['metric']} ({objective['mode']})",
        f"best_trial: {best.get('trial_id', '')}",
        f"best_score: {best.get(objective['metric'], '')}",
        "",
        "## Parameters",
        "",
    ]
    lines.extend(f"- {key}: {value}" for key, value in params.items())
    return "\n".join(lines) + "\n"


def _supersede_pending_trials(root: Path, round_dir: Path) -> None:
    for row in _read_rows(round_dir / "trial_status.tsv"):
        if row.get("status") in {"planned", "pending"}:
            _append_event(
                root,
                "supersede_pending_trial",
                {"round_dir": str(round_dir), "trial_id": row.get("trial_id"), "status": row.get("status")},
            )


def _stop_bad_running_trials(root: Path, round_dir: Path, recipe: dict[str, Any]) -> None:
    adaptive = _adaptive(recipe)
    replacement = adaptive.get("replacement") if isinstance(adaptive.get("replacement"), dict) else {}
    if not replacement.get("enabled", True) or not replacement.get("allow_running_stop", False):
        return
    objective = _objective(root, recipe)
    incumbent = _latest_incumbent_score(root)
    margin = float(replacement.get("kill_margin") or 0.0)
    for row in _read_rows(round_dir / "trial_status.tsv"):
        if row.get("status") != "running":
            continue
        should_stop = _log_has_failure(row.get("log_path"))
        manifest = _find_run_manifest(round_dir, row.get("version", ""), recipe)
        data = _read_json(manifest) if manifest else {}
        score = _metric_value(data, objective["metric"])
        if (
            not should_stop
            and incumbent is not None
            and score not in ("", None)
            and _grace_satisfied(row, data, replacement)
        ):
            try:
                value = float(score)
                should_stop = value < incumbent - margin if objective["mode"] == "max" else value > incumbent + margin
            except (TypeError, ValueError):
                should_stop = False
        if should_stop:
            stop_hparam_trial(round_dir, str(row["trial_id"]))
            _append_event(root, "stop_bad_running_trial", {"round_dir": str(round_dir), "trial_id": row["trial_id"]})


def _grace_satisfied(row: dict[str, Any], manifest: dict[str, Any], replacement: dict[str, Any]) -> bool:
    grace_epochs = replacement.get("grace_epochs")
    if grace_epochs is not None:
        try:
            if float(manifest.get("epoch", "")) < float(grace_epochs):
                return False
        except (TypeError, ValueError):
            return False
    grace_minutes = replacement.get("grace_minutes")
    if grace_minutes is not None:
        minutes = _minutes_since(row.get("launched_at", ""))
        if minutes is None or minutes < float(grace_minutes):
            return False
    return True


def _minutes_since(timestamp: str) -> float | None:
    try:
        start = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None
    return (datetime.now(timezone.utc) - start).total_seconds() / 60


def _latest_incumbent_score(root: Path) -> float | None:
    rows = _read_rows(root / "adaptive" / "incumbents.tsv")
    if not rows:
        return None
    try:
        return float(rows[-1]["objective_score"])
    except (KeyError, TypeError, ValueError):
        return None


def _budget_exhausted(root: Path, recipe: dict[str, Any]) -> bool:
    adaptive = _adaptive(recipe)
    max_rounds = int(adaptive.get("max_rounds") or 1)
    max_trials = int(adaptive.get("max_trials_total") or 10**9)
    return (
        _latest_round_index(root) + 1 >= max_rounds
        or len(_read_rows(root / "adaptive" / "trial_registry.tsv")) >= max_trials
    )


def _adaptive_readme(workflow: dict[str, Any]) -> str:
    return (
        "# Adaptive Hparam Workflow\n\n"
        "This workflow is external-optimized and may use test/external feedback for selection.\n\n"
        f"Objective: `{workflow['objective_metric']}` ({workflow['objective_mode']}).\n"
    )
