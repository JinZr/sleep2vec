"""Repository orientation: Git state, Python details, index existence, and fixed paths.

Layer 0 leaf behind ``repo-summary``. Read-only and best-effort by design: a
missing, broken, or non-git checkout reports ``available: False`` instead of
failing, because this is the first command an agent runs to orient itself and
it must never be the thing that blocks.
"""

from __future__ import annotations

import subprocess
import sys
from typing import TypedDict

from .models import REPO_ROOT


class RepoGitSummary(TypedDict):
    available: bool
    branch: str
    commit: str
    dirty: bool


class RepoIndexSummary(TypedDict):
    path: str
    exists: bool


class RepoImportantPaths(TypedDict):
    agents_md: str
    skills_manifest: str
    configs: str
    tests: str


class RepoPythonSummary(TypedDict):
    executable: str
    version: str


class RepoSummary(TypedDict):
    repo_root: str
    git: RepoGitSummary
    codex_index: RepoIndexSummary
    important_paths: RepoImportantPaths
    python: RepoPythonSummary


def _git(args: list[str]) -> tuple[bool, str]:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return False, ""
    return result.returncode == 0, result.stdout.strip()


def repo_summary() -> RepoSummary:
    git_available = (REPO_ROOT / ".git").exists()
    branch = ""
    commit = ""
    dirty = False
    if git_available:
        ok, branch = _git(["branch", "--show-current"])
        git_available = git_available and ok
        ok, commit = _git(["rev-parse", "HEAD"])
        git_available = git_available and ok
        ok, status = _git(["status", "--short"])
        dirty = bool(status) if ok else False

    index_path = REPO_ROOT / "doc" / "codex_index"
    return {
        "repo_root": str(REPO_ROOT),
        "git": {
            "available": git_available,
            "branch": branch,
            "commit": commit,
            "dirty": dirty,
        },
        "codex_index": {
            "path": str(index_path.relative_to(REPO_ROOT)),
            "exists": index_path.exists(),
        },
        "important_paths": {
            "agents_md": "AGENTS.md",
            "skills_manifest": "skills/manifest.yaml",
            "configs": "configs",
            "tests": "tests",
        },
        "python": {
            "executable": sys.executable,
            "version": sys.version.split()[0],
        },
    }
