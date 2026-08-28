from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path
import re
from typing import Any

from . import experiment_io as exp_io

RESEARCH_LOG_NAME = "RESEARCH_LOG.md"
RESEARCH_LOG_KINDS = {"action", "observation", "interpretation", "decision", "conclusion"}
RESEARCH_LOG_AUTHORITIES = {"human", "policy", "canonical_decision"}
RESEARCH_LOG_ENTRY_FIELDS = {
    "id",
    "recorded_at",
    "kind",
    "title",
    "actor",
    "source",
    "evidence",
    "body",
    "occurred_at",
    "authority",
    "scope",
    "supersedes",
}
RESEARCH_LOG_PREAMBLE = """# Research Log

This append-only log records meaningful research actions, observations, interpretations, decisions, and conclusions.
`run_manifest.tsv` remains the sole authority for execution lifecycle state.

"""
RESEARCH_LOG_MARKER_RE = re.compile(
    r'^<!-- agent-tools-research-entry id="([a-z0-9][a-z0-9._-]{0,127})" sha256="([0-9a-f]{64})" -->\n',
    re.MULTILINE,
)
RESEARCH_LOG_ENTRY_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _research_log_single_line(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Research log entry {field} must be a non-empty string.")
    if value != value.strip() or "\n" in value or "\r" in value:
        raise ValueError(f"Research log entry {field} must be a trimmed single-line string.")
    return value


def _research_log_timestamp(value: Any, field: str) -> str:
    text = _research_log_single_line(value, field)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"Research log entry {field} must be an ISO timestamp.") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError(f"Research log entry {field} must be in UTC.")
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _normalized_research_log_entry(
    entry: dict[str, Any],
    *,
    experiment_id: str,
    managed_rows: list[dict[str, Any]],
    root: Path,
    remote: str | None,
    read_step_manifest,
    managed_run_key,
) -> dict[str, Any]:
    unexpected = sorted(set(entry) - RESEARCH_LOG_ENTRY_FIELDS)
    if unexpected:
        raise ValueError(f"Unexpected research log entry fields: {', '.join(unexpected)}")
    missing = sorted(
        field
        for field in ("id", "recorded_at", "kind", "title", "actor", "source", "evidence", "body")
        if field not in entry
    )
    if missing:
        raise ValueError(f"Research log entry is missing required fields: {', '.join(missing)}")

    entry_id = _research_log_single_line(entry["id"], "id")
    if not RESEARCH_LOG_ENTRY_ID_RE.fullmatch(entry_id):
        raise ValueError("Research log entry id must match [a-z0-9][a-z0-9._-]{0,127}.")
    kind = _research_log_single_line(entry["kind"], "kind")
    if kind not in RESEARCH_LOG_KINDS:
        raise ValueError(f"Research log entry kind must be one of: {', '.join(sorted(RESEARCH_LOG_KINDS))}.")

    authority = entry.get("authority")
    if authority is not None:
        authority = _research_log_single_line(authority, "authority")
        if authority not in RESEARCH_LOG_AUTHORITIES:
            raise ValueError(
                f"Research log entry authority must be one of: {', '.join(sorted(RESEARCH_LOG_AUTHORITIES))}."
            )
    if kind == "decision" and authority is None:
        raise ValueError("Research log decision entries require authority.")

    body = entry["body"]
    if not isinstance(body, str) or not body.strip():
        raise ValueError("Research log entry body must be a non-empty string.")
    body = body.replace("\r\n", "\n").replace("\r", "\n").strip()

    evidence = entry["evidence"]
    if not isinstance(evidence, list) or not evidence:
        raise ValueError("Research log entry evidence must be a non-empty list.")
    normalized_evidence = []
    for index, item in enumerate(evidence):
        if not isinstance(item, dict):
            raise ValueError(f"Research log entry evidence[{index}] must be a mapping.")
        unexpected_evidence = sorted(set(item) - {"label", "locator", "sha256"})
        if unexpected_evidence:
            raise ValueError(
                f"Unexpected research log entry evidence[{index}] fields: {', '.join(unexpected_evidence)}"
            )
        missing_evidence = sorted(field for field in ("label", "locator") if field not in item)
        if missing_evidence:
            raise ValueError(
                f"Research log entry evidence[{index}] is missing required fields: {', '.join(missing_evidence)}"
            )
        normalized_item = {
            "label": _research_log_single_line(item["label"], f"evidence[{index}].label"),
            "locator": _research_log_single_line(item["locator"], f"evidence[{index}].locator"),
        }
        if "sha256" in item:
            digest = _research_log_single_line(item["sha256"], f"evidence[{index}].sha256")
            if not SHA256_RE.fullmatch(digest):
                raise ValueError(f"Research log entry evidence[{index}].sha256 must be a lowercase SHA-256.")
            normalized_item["sha256"] = digest
        normalized_evidence.append(normalized_item)

    scope = entry.get("scope")
    normalized_scope: dict[str, Any] = {}
    if "scope" in entry:
        if not isinstance(scope, dict) or not scope:
            raise ValueError("Research log entry scope must be a non-empty mapping.")
        unexpected_scope = sorted(set(scope) - {"step_id", "run_ids"})
        if unexpected_scope:
            raise ValueError(f"Unexpected research log entry scope fields: {', '.join(unexpected_scope)}")
        if "step_id" not in scope:
            raise ValueError("Research log entry scope.run_ids requires scope.step_id.")
        step_id = _research_log_single_line(scope["step_id"], "scope.step_id")
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", step_id):
            raise ValueError(
                "Research log entry scope.step_id must use lowercase letters, digits, hyphens, and underscores."
            )
        step_manifest = read_step_manifest(root, step_id, remote=remote)
        if step_manifest["experiment_id"] != experiment_id:
            raise ValueError("Research log entry scope.step_id belongs to a different experiment.")
        normalized_scope["step_id"] = step_id
        if "run_ids" in scope:
            run_ids = scope["run_ids"]
            if not isinstance(run_ids, list) or not run_ids:
                raise ValueError("Research log entry scope.run_ids must be a non-empty list.")
            normalized_run_ids = [
                _research_log_single_line(run_id, f"scope.run_ids[{index}]") for index, run_id in enumerate(run_ids)
            ]
            if len(normalized_run_ids) != len(set(normalized_run_ids)):
                raise ValueError("Research log entry scope.run_ids must not contain duplicates.")
            managed_keys = {managed_run_key(row) for row in managed_rows}
            missing_runs = [run_id for run_id in normalized_run_ids if (step_id, run_id) not in managed_keys]
            if missing_runs:
                raise ValueError(f"Research log entry scope references unknown managed runs: {', '.join(missing_runs)}")
            normalized_scope["run_ids"] = normalized_run_ids

    supersedes = entry.get("supersedes")
    normalized_supersedes = []
    if supersedes is not None:
        if not isinstance(supersedes, list):
            raise ValueError("Research log entry supersedes must be a list.")
        normalized_supersedes = [
            _research_log_single_line(superseded_id, f"supersedes[{index}]")
            for index, superseded_id in enumerate(supersedes)
        ]
        if len(normalized_supersedes) != len(set(normalized_supersedes)):
            raise ValueError("Research log entry supersedes must not contain duplicates.")
        if any(not RESEARCH_LOG_ENTRY_ID_RE.fullmatch(superseded_id) for superseded_id in normalized_supersedes):
            raise ValueError("Research log entry supersedes values must be valid research log entry ids.")
        if entry_id in normalized_supersedes:
            raise ValueError("Research log entry cannot supersede itself.")

    normalized = {
        "id": entry_id,
        "recorded_at": _research_log_timestamp(entry["recorded_at"], "recorded_at"),
        "kind": kind,
        "title": _research_log_single_line(entry["title"], "title"),
        "actor": _research_log_single_line(entry["actor"], "actor"),
        "source": _research_log_single_line(entry["source"], "source"),
        "evidence": normalized_evidence,
        "body": body,
    }
    if "occurred_at" in entry:
        normalized["occurred_at"] = _research_log_timestamp(entry["occurred_at"], "occurred_at")
    if authority is not None:
        normalized["authority"] = authority
    if normalized_scope:
        normalized["scope"] = normalized_scope
    if normalized_supersedes:
        normalized["supersedes"] = normalized_supersedes
    return normalized


def _research_log_blocks(text: str, path: Path) -> dict[str, str]:
    markers = list(RESEARCH_LOG_MARKER_RE.finditer(text))
    if not markers:
        if text != RESEARCH_LOG_PREAMBLE:
            raise ValueError(f"Managed research log has an invalid preamble or unmarked content: {path}")
        return {}
    if text[: markers[0].start()] != RESEARCH_LOG_PREAMBLE:
        raise ValueError(f"Managed research log has an invalid preamble: {path}")

    entries = {}
    for index, marker in enumerate(markers):
        entry_id, expected_digest = marker.groups()
        if entry_id in entries:
            raise ValueError(f"Managed research log has a duplicate entry id: {entry_id}")
        block_end = markers[index + 1].start() if index + 1 < len(markers) else len(text)
        block = text[marker.end() : block_end]
        actual_digest = hashlib.sha256(block.encode()).hexdigest()
        if actual_digest != expected_digest:
            raise ValueError(f"Managed research log entry digest differs from its content: {entry_id}")
        if not block.startswith("## ") or f"- Entry ID: `{entry_id}`\n" not in block:
            raise ValueError(f"Managed research log entry has invalid Markdown structure: {entry_id}")
        entries[entry_id] = actual_digest
    return entries


def _research_log_block(entry: dict[str, Any], experiment_id: str) -> str:
    lines = [
        f"## {entry['title']}",
        "",
        f"- Entry ID: `{entry['id']}`",
        f"- Kind: `{entry['kind']}`",
        f"- Recorded at: `{entry['recorded_at']}`",
    ]
    if "occurred_at" in entry:
        lines.append(f"- Occurred at: `{entry['occurred_at']}`")
    lines.extend(
        [
            f"- Actor: {entry['actor']}",
            f"- Source: {entry['source']}",
            f"- Experiment: `{experiment_id}`",
        ]
    )
    scope = entry.get("scope") or {}
    if "step_id" in scope:
        lines.append(f"- Step: `{scope['step_id']}`")
    if "run_ids" in scope:
        lines.append("- Runs:")
        lines.extend(f"  - `{run_id}`" for run_id in scope["run_ids"])
    if "authority" in entry:
        lines.append(f"- Authority: `{entry['authority']}`")
    if "supersedes" in entry:
        lines.append("- Supersedes: " + ", ".join(f"`{entry_id}`" for entry_id in entry["supersedes"]))
    lines.extend(["", "### Evidence", ""])
    for item in entry["evidence"]:
        lines.append(f"- Label: {item['label']}")
        lines.append(f"  Locator: {item['locator']}")
        if "sha256" in item:
            lines.append(f"  - SHA-256: `{item['sha256']}`")
    lines.extend(["", "### Record", "", entry["body"], ""])
    return "\n".join(lines) + "\n"


def append_research_log(
    root: str | Path,
    entry: dict[str, Any],
    *,
    experiment_id: str,
    managed_rows: list[dict[str, Any]],
    remote: str | None,
    read_step_manifest,
    managed_run_key,
    io=exp_io,
) -> tuple[Path, str, bool]:
    root = Path(root)
    path = root / RESEARCH_LOG_NAME
    managed_paths = [path, path.with_name(f".{path.name}.cas.lock")]
    if remote:
        managed_paths.append(Path(f"{path}.lock"))
    scope = entry.get("scope") if isinstance(entry, dict) else None
    if isinstance(scope, dict) and isinstance(scope.get("step_id"), str):
        managed_paths.append(root / "steps" / scope["step_id"] / "step.yaml")
    io.validate_managed_output_paths(root, managed_paths, remote=remote)
    normalized = _normalized_research_log_entry(
        entry,
        experiment_id=experiment_id,
        managed_rows=managed_rows,
        root=root,
        remote=remote,
        read_step_manifest=read_step_manifest,
        managed_run_key=managed_run_key,
    )
    block = _research_log_block(normalized, experiment_id)
    digest = hashlib.sha256(block.encode()).hexdigest()
    marker = f'<!-- agent-tools-research-entry id="{normalized["id"]}" sha256="{digest}" -->\n'

    for _attempt in range(3):
        exists = io.path_exists_at(path, remote=remote)
        current = io.read_text_at(path, remote=remote) if exists else RESEARCH_LOG_PREAMBLE
        entries = _research_log_blocks(current, path)
        existing_digest = entries.get(normalized["id"])
        if existing_digest is not None:
            if existing_digest == digest:
                return path, normalized["id"], False
            raise ValueError(f"Research log entry id already exists with different content: {normalized['id']}")
        missing_superseded = [
            superseded_id for superseded_id in normalized.get("supersedes", []) if superseded_id not in entries
        ]
        if missing_superseded:
            raise ValueError(f"Research log entry supersedes unknown entry ids: {', '.join(missing_superseded)}")
        replacement = current + marker + block
        _research_log_blocks(replacement, path)
        expected_sha256 = hashlib.sha256(current.encode()).hexdigest() if exists else None
        if io.conditional_atomic_replace_text_at(
            path,
            replacement,
            expected_sha256,
            managed_root=root,
            remote=remote,
        ):
            return path, normalized["id"], True
    raise RuntimeError(f"Managed research log changed during three append attempts: {path}")
