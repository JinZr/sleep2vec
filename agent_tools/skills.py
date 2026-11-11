"""Validation of the checked-in ``skills/`` playbooks against their manifest.

Behind the ``skills`` command. ``REQUIRED_HEADINGS`` defines the structural
headings every playbook must carry. The validator checks heading presence and
manifest agreement, but does not validate section body content or execute
anything a playbook describes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypedDict

import yaml

from .models import REPO_ROOT, SUPPORTED_VARIANTS, YAML_SAFE_LOADER, task_requires_variant

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


class SkillSummary(TypedDict):
    name: Any
    path: Any
    task_types: Any


class SkillValidationStatus(TypedDict):
    ok: bool
    issues: list[str]


class SkillValidationResult(SkillValidationStatus, total=False):
    skills: list[SkillSummary]


def load_manifest(path: Path | None = None) -> dict[str, Any]:
    manifest_path = path or REPO_ROOT / "skills" / "manifest.yaml"
    data = yaml.load(manifest_path.read_text(), Loader=YAML_SAFE_LOADER)
    if not isinstance(data, dict):
        raise ValueError("skills/manifest.yaml must be a mapping.")
    return data


def list_skills() -> list[SkillSummary]:
    manifest = load_manifest()
    skills = manifest.get("skills") or {}
    return [
        {
            "name": name,
            "path": item.get("path"),
            "task_types": item.get("task_types", []),
        }
        for name, item in sorted(skills.items())
    ]


def validate_skills() -> SkillValidationResult:
    issues: list[str] = []
    manifest_path = REPO_ROOT / "skills" / "manifest.yaml"
    if not manifest_path.exists():
        return {"ok": False, "issues": ["skills/manifest.yaml is missing."]}
    manifest = load_manifest(manifest_path)
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
    for example in (REPO_ROOT / "skills").glob("*/examples/*.yaml"):
        data = yaml.load(example.read_text(), Loader=YAML_SAFE_LOADER)
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
