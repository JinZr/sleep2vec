from __future__ import annotations

import json
from pathlib import Path
import pickle
import sys
from typing import Any

from ..models import REPO_ROOT, repo_relative, resolve_repo_path


def preset_summary(preset_path: str | Path, *, local_path_base: str | Path | None = None) -> dict[str, Any]:
    resolved = resolve_repo_path(preset_path, relative_to=local_path_base)
    if resolved is None:
        raise FileNotFoundError("Preset path is required.")
    warnings: list[str] = []
    blocking_issues: list[str] = []
    if not resolved.exists():
        return {
            "preset_path": repo_relative(resolved),
            "samples": 0,
            "warnings": [],
            "blocking_issues": [f"Preset not found: {resolved}"],
        }
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    try:
        with resolved.open("rb") as file_obj:
            samples = pickle.load(file_obj)
    except Exception as exc:  # pragma: no cover - defensive around old pickles
        return {
            "preset_path": repo_relative(resolved),
            "samples": 0,
            "warnings": [],
            "blocking_issues": [f"Failed to load preset: {exc}"],
        }

    items = samples if isinstance(samples, list) else []
    metadata_keys = sorted({key for item in items for key in getattr(item, "metadata", {}).keys()})
    payload_keys = sorted({key for item in items for key in getattr(item, "payload", {}).keys()})
    available_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    for item in items:
        for channel in getattr(item, "payload", {}).get("available_channels", []) or []:
            available_counts[str(channel)] = available_counts.get(str(channel), 0) + 1
        source = getattr(item, "metadata", {}).get("source")
        if source not in (None, ""):
            source_counts[str(source)] = source_counts.get(str(source), 0) + 1

    starts = [getattr(item, "start", None) for item in items if getattr(item, "start", None) is not None]
    ends = [getattr(item, "end", None) for item in items if getattr(item, "end", None) is not None]
    manifest_path = resolved.with_name(f"{resolved.name}.manifest.json")
    manifest = None
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
        except json.JSONDecodeError as exc:
            warnings.append(f"Sidecar manifest is not valid JSON: {exc}")

    return {
        "preset_path": repo_relative(resolved),
        "samples": len(items),
        "id_examples": [getattr(item, "id", None) for item in items[:5]],
        "path_examples": [getattr(item, "path", None) for item in items[:5]],
        "start_end": {
            "min_start": min(starts) if starts else None,
            "max_end": max(ends) if ends else None,
        },
        "metadata_keys": metadata_keys,
        "payload_keys": payload_keys,
        "available_channels_counts": available_counts,
        "source_counts": source_counts,
        "sidecar_manifest": manifest,
        "warnings": warnings,
        "blocking_issues": blocking_issues,
    }
