from __future__ import annotations

import subprocess
import sys
from typing import Any

from .models import REPO_ROOT


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


def repo_summary() -> dict[str, Any]:
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
