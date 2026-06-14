from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .models import REPO_ROOT, SUPPORTED_VARIANTS, task_requires_variant

REQUIRED_HEADINGS = (
    "## When to use",
    "## Required inputs",
    "## First information-gathering commands",
    "## Decision checklist",
    "## Stop-and-consult gates",
    "## Canonical commands",
    "## Expected artifacts",
    "## Validation gates",
    "## Common failure modes",
    "## Relevant owners and index pages",
)


def load_manifest(path: Path | None = None) -> dict[str, Any]:
    manifest_path = path or REPO_ROOT / "skills" / "manifest.yaml"
    data = yaml.safe_load(manifest_path.read_text())
    if not isinstance(data, dict):
        raise ValueError("skills/manifest.yaml must be a mapping.")
    return data


def list_skills() -> list[dict[str, Any]]:
    manifest = load_manifest()
    skills = manifest.get("skills") or {}
    return [
        {
            "name": name,
            "path": item.get("path"),
            "task_types": item.get("task_types", []),
            "owners": item.get("owners", []),
            "relevant_index": item.get("relevant_index", []),
        }
        for name, item in sorted(skills.items())
    ]


def validate_skills() -> dict[str, Any]:
    issues: list[str] = []
    manifest_path = REPO_ROOT / "skills" / "manifest.yaml"
    if not manifest_path.exists():
        return {"ok": False, "issues": ["skills/manifest.yaml is missing."]}
    manifest = load_manifest(manifest_path)
    agents_text = (REPO_ROOT / "AGENTS.md").read_text() if (REPO_ROOT / "AGENTS.md").exists() else ""
    skills = manifest.get("skills") or {}
    for name, item in skills.items():
        skill_path = REPO_ROOT / str(item.get("path", ""))
        if not skill_path.exists():
            issues.append(f"{name}: skill path missing: {item.get('path')}")
            continue
        text = skill_path.read_text()
        for heading in REQUIRED_HEADINGS:
            if heading not in text:
                issues.append(f"{name}: missing heading {heading}")
        for owner in item.get("owners", []):
            if owner not in agents_text:
                issues.append(f"{name}: owner not present in AGENTS.md: {owner}")
        for index_path in item.get("relevant_index", []):
            if not (REPO_ROOT / index_path).exists():
                issues.append(f"{name}: relevant index path missing: {index_path}")
    for example in (REPO_ROOT / "skills").glob("*/examples/*.yaml"):
        data = yaml.safe_load(example.read_text())
        if not isinstance(data, dict):
            issues.append(f"{example}: example must be a mapping")
            continue
        for key in ("name", "task"):
            if key not in data:
                issues.append(f"{example}: missing {key}")
        task = data.get("task")
        variant = data.get("variant")
        if task_requires_variant(task):
            if variant not in SUPPORTED_VARIANTS:
                issues.append(f"{example}: unsupported variant {variant}; expected one of {SUPPORTED_VARIANTS}")
        elif variant not in (None, ""):
            issues.append(
                f"{example}: task={task} must omit variant or set it to null; "
                "sleep2stat is a task, not a model variant."
            )
    return {"ok": not issues, "issues": issues, "skills": list_skills()}
