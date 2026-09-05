"""Guard tests for utils/type_check.py, the mypy ledger ratchet.

The script decides whether grandfathered modules may leave the ledger, so every
way it can reach a wrong conclusion has to be pinned down. Its failure mode is
not a crash: it is confidently telling a contributor to delete valid ledger
entries, or silently accepting a new one.

Each test below corresponds to a way that actually went wrong in review:

* an incomplete mypy run undercounts, and an undercount looks exactly like a
  shrunken ledger;
* a *complete* run with no errors is the state the ratchet is aiming at, and
  must not be mistaken for a broken one;
* the neutralized pass is expensive, so it must run once, not once per entry.

mypy is never invoked here -- run_mypy is replaced with canned output captured
from real runs, so these stay fast and hermetic.
"""

from __future__ import annotations

import ast
from pathlib import Path
import subprocess

import pytest

from utils import type_check

# Captured verbatim from mypy 1.18.2; the three ways the neutralized pass ends.
COMPLETED_WITH_ERRORS = (
    'agent_tools/plans.py:269: error: Item "None" of "dict | None" has no attribute "get"  [union-attr]\n'
    "agent_tools/slurm.py:31: error: Incompatible types in assignment  [assignment]\n"
    "Found 908 errors in 43 files (checked 63 source files)\n"
)
COMPLETED_WITHOUT_ERRORS = "Success: no issues found in 63 source files\n"
ABORTED_EARLY = (
    "numpy/__init__.pyi:737: error: Type statement is only supported in Python 3.12 and greater  [syntax]\n"
    'agent_tools/plans.py:269: error: Item "None" of "dict | None" has no attribute "get"  [union-attr]\n'
    "Found 3 errors in 3 files (errors prevented further checking)\n"
)

LEDGERED = """
[tool.mypy]
python_version = "3.10"
files = ["agent_tools"]

[[tool.mypy.overrides]]
module = ["yaml.*", "pandas.*"]
ignore_missing_imports = true

[[tool.mypy.overrides]]
ignore_errors = true
module = [
    "agent_tools.plans",
    "agent_tools.slurm",
]
"""


def _completed(stdout: str, returncode: int) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=["mypy"], returncode=returncode, stdout=stdout, stderr="")


def _canned(stdout: str, returncode: int, calls: list[tuple[str, ...]] | None = None):
    def run_mypy(*arguments: str) -> subprocess.CompletedProcess[str]:
        if calls is not None:
            calls.append(arguments)
        return _completed(stdout, returncode)

    return run_mypy


def test_ledger_counts_only_ignore_errors_overrides():
    # The yaml/pandas override sits in the same [[tool.mypy.overrides]] array
    # but grandfathers nothing; counting it would inflate the ledger.
    assert type_check.ledger_of(LEDGERED) == {"agent_tools.plans", "agent_tools.slurm"}


def test_missing_import_allowlist_is_limited_to_known_third_party_dependencies():
    document = type_check.tomllib.loads((Path(type_check.__file__).parents[1] / "pyproject.toml").read_text())
    allowlists = [
        override["module"]
        for override in document["tool"]["mypy"]["overrides"]
        if override.get("ignore_missing_imports")
    ]

    assert allowlists == [["yaml.*", "pandas.*", "wandb.*", "torch.*"]]
    assert all(not module.startswith("agent_tools") for module in allowlists[0])


def test_ledger_of_reports_a_revision_without_mypy_config():
    # None is the bootstrap signal, distinct from an empty ledger.
    assert type_check.ledger_of("[tool.black]\nline-length = 120\n") is None
    assert type_check.ledger_of('[tool.mypy]\nfiles = ["agent_tools"]\n') == set()


def test_completed_run_with_errors_maps_files_to_modules(monkeypatch):
    monkeypatch.setattr(type_check, "run_mypy", _canned(COMPLETED_WITH_ERRORS, 1))

    assert type_check.modules_with_errors(LEDGERED) == {"agent_tools.plans", "agent_tools.slurm"}


def test_completed_run_without_errors_is_not_an_environment_failure(monkeypatch):
    # The state the ratchet aims at: the last grandfathered module is fixed, so
    # mypy exits 0 with a summary that names no error count. Reading that as a
    # broken run would stall the cleanup at the finish line.
    monkeypatch.setattr(type_check, "run_mypy", _canned(COMPLETED_WITHOUT_ERRORS, 0))

    assert type_check.modules_with_errors(LEDGERED) == set()


def test_aborted_run_refuses_to_judge_instead_of_undercounting(monkeypatch):
    # Seen on Python 3.11+: mypy stops on numpy's PEP 695 stubs and reports 3
    # errors instead of 908. Trusting that would call almost every entry stale.
    monkeypatch.setattr(type_check, "run_mypy", _canned(ABORTED_EARLY, 2))

    with pytest.raises(SystemExit) as excinfo:
        type_check.modules_with_errors(LEDGERED)

    assert "do not delete ledger entries" in str(excinfo.value)


def test_staleness_check_runs_mypy_once_per_invocation(tmp_path, monkeypatch):
    # Not a micro-optimization: evaluating the failing set inside the
    # comprehension's condition ran one full --no-incremental pass per ledger
    # entry and took the CI job from 37s to 4m40s.
    calls: list[tuple[str, ...]] = []
    monkeypatch.setattr(type_check, "run_mypy", _canned(COMPLETED_WITH_ERRORS, 0, calls))
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text(LEDGERED, encoding="utf-8")

    assert type_check.main([]) == 0
    # One bare run for the mypy check, one neutralized run for staleness.
    assert len(calls) == 2, calls
    assert calls[0] == ()
    assert "--no-incremental" in calls[1]


def test_every_entry_is_reported_stale_once_the_ledger_is_paid_off(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(type_check, "run_mypy", _canned(COMPLETED_WITHOUT_ERRORS, 0))
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text(LEDGERED, encoding="utf-8")

    assert type_check.main([]) == 1

    output = capsys.readouterr().out
    assert "Stale mypy ledger entries" in output
    assert '"agent_tools.plans",' in output
    assert '"agent_tools.slurm",' in output


def test_live_ledger_passes(tmp_path, monkeypatch):
    monkeypatch.setattr(type_check, "run_mypy", _canned(COMPLETED_WITH_ERRORS, 0))
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text(LEDGERED, encoding="utf-8")

    assert type_check.main([]) == 0


def test_mypy_failure_short_circuits_before_the_ledger(tmp_path, monkeypatch):
    # A plain type error must fail on its own terms, without the ledger check
    # second-guessing it.
    calls: list[tuple[str, ...]] = []
    monkeypatch.setattr(type_check, "run_mypy", _canned(COMPLETED_WITH_ERRORS, 1, calls))
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text(LEDGERED, encoding="utf-8")

    assert type_check.main([]) == 1
    assert calls == [()]


def test_additions_are_rejected(tmp_path, monkeypatch, capsys):
    # The scenario the base-revision diff exists for: hide a new type error by
    # grandfathering its module. mypy passes and the entry looks perfectly live.
    monkeypatch.setattr(type_check, "run_mypy", _canned(COMPLETED_WITH_ERRORS, 0))
    monkeypatch.setattr(type_check, "base_document", lambda revision: LEDGERED)
    monkeypatch.chdir(tmp_path)
    grown = LEDGERED.replace('    "agent_tools.slurm",', '    "agent_tools.slurm",\n    "agent_tools.repo",')
    (tmp_path / "pyproject.toml").write_text(grown, encoding="utf-8")

    assert type_check.main(["--base", "deadbeef"]) == 1

    output = capsys.readouterr().out
    assert "may only shrink" in output
    assert '"agent_tools.repo",' in output


def test_bootstrap_revision_without_a_ledger_is_not_an_addition(tmp_path, monkeypatch):
    # The commit that introduces the ledger has no previous one to diff against.
    monkeypatch.setattr(type_check, "run_mypy", _canned(COMPLETED_WITH_ERRORS, 0))
    monkeypatch.setattr(type_check, "base_document", lambda revision: "[tool.black]\nline-length = 120\n")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text(LEDGERED, encoding="utf-8")

    assert type_check.main(["--base", "deadbeef"]) == 0


@pytest.mark.parametrize("revision", ["", "0" * 40])
def test_missing_base_revision_skips_the_additions_check(revision):
    # workflow_dispatch has no base sha, and a new branch push reports all-zero.
    assert type_check.base_document(revision) is None


def test_toml_reader_works_on_this_interpreter():
    assert type_check.tomllib.__name__ in {"tomli", "tomllib"}
    assert type_check.tomllib.loads('[tool.mypy]\nfiles = ["agent_tools"]\n')["tool"]["mypy"]


def test_tomli_is_imported_only_under_a_version_guard():
    # mypy declares tomli only below 3.11, so an unconditional import crashes
    # the script on a 3.11+ machine before any check runs -- and since
    # utils/style_check.sh always invokes it, the documented local check fails
    # on a clean repository. Every CI matrix pins 3.10, so running the tests
    # cannot catch that: inspect the source instead of the live import.
    tree = ast.parse(Path(type_check.__file__).read_text(encoding="utf-8"))
    guarded = {
        node
        for branch in ast.walk(tree)
        if isinstance(branch, ast.If)
        for node in ast.walk(branch)
        if isinstance(node, (ast.Import, ast.ImportFrom))
    }

    unguarded = [
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import) and node not in guarded
        for alias in node.names
        if alias.name == "tomli"
    ]

    assert unguarded == [], "tomli must be imported only in the sys.version_info < (3, 11) branch"


def test_ledger_toggle_must_be_unambiguous(monkeypatch):
    # The neutralized config is produced by flipping exactly one line; a second
    # ignore_errors block would silently leave part of the ledger in force.
    monkeypatch.setattr(type_check, "run_mypy", _canned(COMPLETED_WITH_ERRORS, 1))

    with pytest.raises(SystemExit) as excinfo:
        type_check.modules_with_errors(LEDGERED + '\n[[tool.mypy.overrides]]\nignore_errors = true\nmodule = ["x"]\n')

    assert "exactly one ignore_errors toggle" in str(excinfo.value)
