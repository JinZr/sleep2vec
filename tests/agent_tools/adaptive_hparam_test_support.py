from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys

from agent_tool_test_helpers import run_execution_preflight_fixture, write_finetune_recipe, write_yaml
import pytest
import yaml

from agent_tools import hparam_runtime, managed_scheduler
from agent_tools.experiment_workspace import merge_run_manifest
from agent_tools.models import REPO_ROOT


_RUNTIME_COMMIT = subprocess.run(
    ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, text=True, capture_output=True
).stdout.strip()


@pytest.fixture(autouse=True)
def _stub_execution_snapshot_preflight(monkeypatch):
    monkeypatch.setattr(hparam_runtime, "_validated_execution_snapshot", lambda *_args, **_kwargs: (None, False))
    monkeypatch.setattr(
        managed_scheduler,
        "run_execution_command",
        run_execution_preflight_fixture,
    )


def _run(*args: str) -> subprocess.CompletedProcess:
    runner = Path(__file__).with_name("agent_tools_cli_stub.py")
    return subprocess.run([sys.executable, str(runner), *args], text=True, capture_output=True)


def _read_table(path: Path) -> list[dict[str, str]]:
    delimiter = "\t" if path.suffix == ".tsv" else ","
    with path.open(newline="") as file_obj:
        return list(csv.DictReader(file_obj, delimiter=delimiter))


def _adaptive_recipe(
    tmp_path: Path, *, test_feedback: bool = True, max_rounds: int = 2, relative_base: bool = False
) -> Path:
    base = write_finetune_recipe(tmp_path)
    return write_yaml(
        tmp_path / "adaptive_tune.yaml",
        {
            "name": "unit_adaptive",
            "task": "hparam_tune",
            "variant": "sleep2vec",
            "base_recipe": base.name if relative_base else str(base),
            "execution": {
                "workdir": str(tmp_path / "runtime"),
                "python": sys.executable,
                "runtime_commit": _RUNTIME_COMMIT,
            },
            "search": {
                "method": "grid",
                "max_runs": 1,
                "parameters": {"runtime.lr": [1e-6], "yaml:/model/head/name": ["classification"]},
            },
            "adaptive": {
                "enabled": True,
                "objective_metric": "test_auroc",
                "objective_mode": "max",
                "test_feedback_for_selection": test_feedback,
                "max_rounds": max_rounds,
                "max_runs_total": 4,
                "round_size": 1,
                "poll_seconds": 1,
                "replacement": {
                    "enabled": True,
                    "allow_running_stop": True,
                    "grace_epochs": 1,
                    "grace_minutes": 1,
                    "kill_margin": 0.05,
                },
                "suggest": {"strategy": "best_neighborhood"},
            },
            "evaluation_policy": {
                "selection_metric": "val_ahi_pearson",
                "selection_mode": "max",
                "selection_split": "val",
                "external_test_locked": False,
                "test_after_fit": True,
                "final_eval_split": "test",
                "final_test_unlocked": False,
                "require_manual_unlock_for_final_test": True,
            },
            "decisions": {
                "task": {"value": "hparam_tune", "source": "explicit_recipe"},
                "label_name": {"value": "ahi", "source": "explicit_recipe"},
                "external_test_locked": {"value": False, "source": "explicit_recipe"},
                "train_val_test_policy": {"value": "external optimized adaptive", "source": "explicit_recipe"},
                "overwrite_policy": {"value": False, "source": "explicit_recipe"},
                "final_eval_unlock": {"value": False, "source": "explicit_recipe"},
            },
        },
    )


def _agent_recipe(tmp_path: Path, *, max_rounds: int = 2, explicit_strategy: bool = True) -> Path:
    recipe = _adaptive_recipe(tmp_path, max_rounds=max_rounds)
    payload = yaml.safe_load(recipe.read_text())
    payload["adaptive"]["replacement"] = {"enabled": False}
    payload["adaptive"]["suggest"] = {
        "bounds": {"runtime.lr": [5e-7, 2e-6]},
    }
    if explicit_strategy:
        payload["adaptive"]["suggest"]["strategy"] = "agent_proposal"
    recipe.write_text(yaml.safe_dump(payload))
    return recipe


def _test_selected_adaptive_recipe(
    tmp_path: Path,
    *,
    objective_mode: str = "max",
    strategy: str = "agent_proposal",
    max_rounds: int = 2,
) -> Path:
    recipe = (
        _agent_recipe(tmp_path, max_rounds=max_rounds)
        if strategy == "agent_proposal"
        else _adaptive_recipe(tmp_path, max_rounds=max_rounds)
    )
    payload = yaml.safe_load(recipe.read_text())
    payload["adaptive"]["objective_mode"] = objective_mode
    payload["evaluation_policy"].update(
        {
            "selection_metric": "test_auroc",
            "selection_mode": objective_mode,
            "selection_split": "test",
            "external_test_locked": False,
            "test_after_fit": True,
        }
    )
    payload.setdefault("runtime", {})["ckpt_every_n_epochs"] = 1
    payload["decisions"]["train_val_test_policy"] = {"value": "test", "source": "explicit_recipe"}
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    return recipe


def _write_agent_submission(input_path: Path, *, lr: list[float] | None = None) -> Path:
    proposal_input = json.loads(input_path.read_text())
    proposal_path = Path(proposal_input["expected_proposal_path"])
    proposal_path.parent.mkdir(parents=True, exist_ok=True)
    proposal_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "request_id": proposal_input["request_id"],
                "target_round": proposal_input["input"]["target_round"],
                "parameters": {
                    "runtime.lr": lr or [5e-7],
                    "yaml:/model/head/name": ["classification"],
                },
                "evidence_run_ids": [proposal_input["input"]["digest_rows"][0]["run_id"]],
                "rationale": "The terminal run supports a lower learning rate within the authorized bounds.",
                "proposer": {"agent": "codex", "model": "gpt-5"},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return proposal_path
def _write_fake_manifest(workflow_dir: Path, *, score: float = 0.7) -> None:
    round_dir = workflow_dir / "adaptive" / "rounds" / "round_000"
    launched = _run("hparam-launch", "--plan-dir", str(round_dir))
    assert launched.returncode == 0, launched.stderr
    plan = json.loads((round_dir / "plan.json").read_text())
    run = plan["runs"][0]
    run_dir = Path(run["runtime_dir"])
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "epoch=3.ckpt").write_text("checkpoint")
    (ckpt_dir / "best-epoch=3.ckpt").write_text("alias")
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "version": run["version"],
                "monitor": "val_ahi_pearson",
                "monitor_mode": "max",
                "best_model_score": 0.5,
                "best_model_path": str(ckpt_dir / "best-epoch=3.ckpt"),
                "epoch": 3,
                "status": "finished",
                "metrics": {"val_ahi_pearson": 0.5, "test_auroc": score},
            }
        )
    )


def _write_checkpoint_test_manifest(
    workflow_dir: Path,
    *,
    scores: dict[int, float],
    top_level_score: float,
    extra_checkpoint_metrics: dict[str, dict[int, float]] | None = None,
    extra_top_level_metrics: dict[str, float] | None = None,
    best_model_score: float = 0.5,
) -> tuple[dict, dict[int, Path]]:
    round_dir = workflow_dir / "adaptive" / "rounds" / "round_000"
    launched = _run("hparam-launch", "--plan-dir", str(round_dir))
    assert launched.returncode == 0, launched.stderr
    run = json.loads((round_dir / "plan.json").read_text())["runs"][0]
    checkpoint_dir = Path(run["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True)
    checkpoints = {epoch: checkpoint_dir / f"epoch={epoch}.ckpt" for epoch in scores}
    for checkpoint in checkpoints.values():
        checkpoint.write_text(checkpoint.name)
    (checkpoint_dir / "best-epoch=1.ckpt").write_text("validation-best alias")
    (Path(run["runtime_dir"]) / "run_manifest.json").write_text(
        json.dumps(
            {
                "version": run["version"],
                "monitor": "val_ahi_pearson",
                "monitor_mode": "max",
                "best_model_score": best_model_score,
                "best_model_path": str(checkpoint_dir / "best-epoch=1.ckpt"),
                "epoch": 1,
                "metrics": {
                    "val_ahi_pearson": 0.5,
                    "test_auroc": top_level_score,
                    **(extra_top_level_metrics or {}),
                },
                "test_all_checkpoints_after_fit": True,
                "checkpoint_test_results": [
                    {
                        "checkpoint_path": str(checkpoints[epoch]),
                        "epoch": epoch,
                        "metrics": {
                            "test_auroc": score,
                            **{
                                metric: metric_scores[epoch]
                                for metric, metric_scores in (extra_checkpoint_metrics or {}).items()
                            },
                        },
                    }
                    for epoch, score in scores.items()
                ],
            }
        )
    )
    return run, checkpoints


def _mark_round_terminal(workflow_dir: Path, workspace: Path, *, status: str = "finished") -> None:
    run = json.loads((workflow_dir / "adaptive" / "rounds" / "round_000" / "plan.json").read_text())["runs"][0]
    merge_run_manifest(
        workspace,
        [{"step_id": run["step_id"], "run_id": run["run_id"], "status": status}],
    )



def _write_agent_configuration_submission(input_path: Path) -> Path:
    proposal_input = json.loads(input_path.read_text())
    proposal_path = Path(proposal_input["expected_proposal_path"])
    proposal_path.parent.mkdir(parents=True, exist_ok=True)
    proposal_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "request_id": proposal_input["request_id"],
                "target_round": proposal_input["input"]["target_round"],
                "configurations": [
                    {"runtime.lr": 5e-7, "yaml:/model/head/name": "classification"},
                    {"runtime.lr": 2e-6, "yaml:/model/head/name": "classification"},
                ],
                "evidence_run_ids": [proposal_input["input"]["digest_rows"][0]["run_id"]],
                "rationale": "Probe both authorized ends of the LR interval as exact points.",
                "proposer": {"agent": "codex", "model": "gpt-5"},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return proposal_path
