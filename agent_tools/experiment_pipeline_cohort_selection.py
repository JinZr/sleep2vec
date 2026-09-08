"""Pure candidate-matrix expansion and the frozen target-gate decision.

Layer 0 leaf, pure: it expands the declared cohort axes into phase jobs and
ranks candidates against the frozen gate. No filesystem, scheduler, or
observation access -- the orchestrator in ``experiment_pipeline`` supplies the
evidence it decides on.
"""

from __future__ import annotations

import math
from typing import Any, Literal, Mapping, Sequence, TypedDict


class SelectionEvidence(TypedDict):
    job_id: Any
    job_template_id: Any
    candidate_id: Any
    cohort: Any
    metrics: dict[str, Any]
    result_manifest: str
    result_manifest_sha256: str


class GateContributingEvidence(TypedDict):
    job: Any
    cohort: Any
    metric: Any
    mode: Literal["min", "max"]
    threshold: float
    value: int | float
    result_manifest: str
    result_manifest_sha256: str


class DecisionCandidate(TypedDict):
    candidate_id: str
    source_rank: int
    step_id: Any
    run_id: Any
    checkpoint: Any
    checkpoint_sha256: Any
    config: Any
    config_sha256: Any
    feasible: bool
    failed_gates: list[str]
    selection_evidence: list[GateContributingEvidence]


class CohortDecision(TypedDict):
    pipeline_id: Any
    source_id: Any
    selector: dict[str, Any]
    candidates: list[DecisionCandidate]
    winner: DecisionCandidate | None


def candidate_for_job(job: dict[str, Any], candidates: Mapping[str, Mapping[str, Any]]) -> Mapping[str, Any]:
    key = str(job.get("candidate_id") or job.get("checkpoint_source") or "")
    return candidates[key]


def build_phase_jobs(
    spec: dict[str, Any],
    candidates: Mapping[str, Mapping[str, Any]],
    *,
    role: str,
    winner_id: str | None = None,
) -> list[dict[str, Any]]:
    templates = sorted((job for job in spec["jobs"] if job["role"] == role), key=lambda job: job["id"])
    if role == "selection":
        selected = sorted(candidates.values(), key=_candidate_order)
    else:
        if winner_id is None:
            raise ValueError("Report-only jobs require a frozen cohort-selection winner.")
        selected = [candidates[winner_id]]

    jobs = []
    for candidate in selected:
        for template in templates:
            candidate_id = str(candidate["candidate_id"])
            jobs.append(
                {
                    **template,
                    "id": f"{role}--{candidate_id}--{template['id']}",
                    "job_template_id": template["id"],
                    "candidate_id": candidate_id,
                    "checkpoint_source": candidate["source_id"],
                }
            )
    return jobs


def rank_candidates(
    spec: dict[str, Any],
    candidates: Mapping[str, Mapping[str, Any]],
    evidence: Sequence[SelectionEvidence],
) -> tuple[list[dict[str, Any]], CohortDecision]:
    selection_jobs = {job["id"]: job for job in spec["jobs"] if job["role"] == "selection"}
    evidence_by_key: dict[tuple[str, str], SelectionEvidence] = {}
    for evidence_row in evidence:
        key = (str(evidence_row.get("candidate_id") or ""), str(evidence_row.get("job_template_id") or ""))
        if key in evidence_by_key:
            raise ValueError(f"Duplicate cohort-selection result: {key[0]} / {key[1]}")
        evidence_by_key[key] = evidence_row

    expected = {(candidate_id, job_id) for candidate_id in candidates for job_id in selection_jobs}
    if set(evidence_by_key) != expected:
        raise ValueError("Cohort-selection results do not form the complete frozen candidate-by-cohort matrix.")

    gates = sorted(spec["selector"]["gates"], key=lambda gate: (gate["job"], gate["metric"]))
    ranking: list[dict[str, Any]] = []
    decision_candidates: list[DecisionCandidate] = []
    for candidate in sorted(candidates.values(), key=_candidate_order):
        candidate_id = str(candidate["candidate_id"])
        values = {}
        failed = []
        contributing: list[GateContributingEvidence] = []
        for gate in gates:
            result = evidence_by_key[(candidate_id, gate["job"])]
            metrics = result.get("metrics")
            value = metrics.get(gate["metric"]) if isinstance(metrics, dict) else None
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(
                    f"Cohort-selection metric must be finite: {candidate_id} / {gate['job']} / {gate['metric']}"
                )
            passed = value <= gate["threshold"] if gate["mode"] == "min" else value >= gate["threshold"]
            gate_id = f"{gate['job']}:{gate['metric']}"
            values[gate_id] = value
            if not passed:
                failed.append(gate_id)
            contributing.append(
                {
                    "job": gate["job"],
                    "cohort": selection_jobs[gate["job"]]["cohort"],
                    "metric": gate["metric"],
                    "mode": gate["mode"],
                    "threshold": gate["threshold"],
                    "value": value,
                    "result_manifest": result["result_manifest"],
                    "result_manifest_sha256": result["result_manifest_sha256"],
                }
            )
        ranking.append(
            {
                "candidate_id": candidate_id,
                "source_rank": int(candidate["source_rank"]),
                "step_id": candidate["step_id"],
                "run_id": candidate["run_id"],
                "internal_metric": candidate["selection_metric"],
                "internal_score": candidate["score"],
                "feasible": not failed,
                "failed_gates": ";".join(failed),
                **{f"gate.{name}": value for name, value in sorted(values.items())},
            }
        )
        decision_candidates.append(
            {
                "candidate_id": candidate_id,
                "source_rank": int(candidate["source_rank"]),
                "step_id": candidate["step_id"],
                "run_id": candidate["run_id"],
                "checkpoint": candidate["checkpoint"],
                "checkpoint_sha256": candidate["checkpoint_sha256"],
                "config": candidate["config"],
                "config_sha256": candidate["config_sha256"],
                "feasible": not failed,
                "failed_gates": failed,
                "selection_evidence": contributing,
            }
        )

    feasible = [candidate for candidate in decision_candidates if candidate["feasible"]]
    winner = min(feasible, key=_candidate_order) if feasible else None
    winner_id = str(winner["candidate_id"]) if winner is not None else None
    for row in ranking:
        row["winner"] = row["candidate_id"] == winner_id
    decision: CohortDecision = {
        "pipeline_id": spec["pipeline"]["id"],
        "source_id": next(iter(spec["checkpoint_sources"])),
        "selector": spec["selector"],
        "candidates": decision_candidates,
        "winner": winner,
    }
    return ranking, decision


def _candidate_order(candidate: Mapping[str, Any]) -> tuple[int, str]:
    return int(candidate["source_rank"]), str(candidate["run_id"])
