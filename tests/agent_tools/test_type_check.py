"""Guard the unsuppressed, single-pass mypy wrapper and its shrink-only Any ledger."""

from __future__ import annotations

import ast
from pathlib import Path
import subprocess
import sys

import pytest

from utils import type_check


@pytest.mark.parametrize(
    "suppression",
    [
        "ignore_errors = true",
        '\n[[tool.mypy.overrides]]\nmodule = "agent_tools.plans"\nignore_errors = true',
        '\n[[tool.mypy.overrides]]\nmodule = "*"\nignore_errors = true',
    ],
)
def test_error_suppression_is_rejected_before_mypy(tmp_path, monkeypatch, capsys, suppression):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text('[tool.mypy]\nfiles = ["agent_tools"]\n' + suppression)
    monkeypatch.setattr(type_check.subprocess, "run", lambda *args, **kwargs: pytest.fail("Suppression reached mypy"))

    assert type_check.main([]) == 1
    output = capsys.readouterr()
    assert "ignore_errors" in output.out + output.err


@pytest.mark.parametrize("returncode", [0, 1, 2])
def test_runs_mypy_once_with_explicit_config_and_preserves_exit_status(tmp_path, monkeypatch, returncode):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text(
        '[tool.mypy]\nfiles = ["agent_tools"]\nignore_errors = false\n'
        '[[tool.mypy.overrides]]\nmodule = "yaml.*"\nignore_missing_imports = true\nignore_errors = false\n'
    )
    (tmp_path / "agent_tools").mkdir()
    monkeypatch.setattr(type_check, "ANY_LEDGER", {"Any": 0, "dict[str, Any]": 0})
    calls = []

    def run(args, **kwargs):
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args, returncode)

    monkeypatch.setattr(type_check.subprocess, "run", run)

    assert type_check.main([]) == returncode
    assert calls == [([sys.executable, "-m", "mypy", "--config-file", str(type_check.PYPROJECT)], {})]


SPELLINGS = """\
from typing import Any, Mapping
import typing

counted: Any
attribute: typing.Any
payload: dict[str, Any]
read_only: Mapping[str, Any]
narrow: dict[str, int]
text = "Any"  # Any in a string or a comment is not a spelling.
"""


@pytest.mark.parametrize(
    ("ledger", "returncode", "report"),
    [
        ({"Any": 4, "dict[str, Any]": 1}, 0, "Any ledger: 4 Any, 1 dict[str, Any], as recorded."),
        ({"Any": 3, "dict[str, Any]": 1}, 1, "Any: found 4, ledger 3. The ledger only shrinks"),
        ({"Any": 4, "dict[str, Any]": 2}, 1, "Lower ANY_LEDGER['dict[str, Any]'] to 1 in this commit."),
    ],
)
def test_any_ledger_must_match_the_explicit_spellings_and_mypy_still_runs(
    tmp_path, monkeypatch, capsys, ledger, returncode, report
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text('[tool.mypy]\nfiles = ["agent_tools"]\n')
    (tmp_path / "agent_tools" / "nested").mkdir(parents=True)
    (tmp_path / "agent_tools" / "nested" / "module.py").write_text(SPELLINGS)
    monkeypatch.setattr(type_check, "ANY_LEDGER", ledger)
    calls = []

    def run(args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(type_check.subprocess, "run", run)

    assert type_check.main([]) == returncode
    assert calls == [[sys.executable, "-m", "mypy", "--config-file", str(type_check.PYPROJECT)]]
    assert report in capsys.readouterr().out


def test_removed_base_option_is_rejected_before_mypy(monkeypatch, capsys):
    monkeypatch.setattr(
        type_check.subprocess, "run", lambda *args, **kwargs: pytest.fail("Obsolete option reached mypy")
    )

    with pytest.raises(SystemExit) as excinfo:
        type_check.main(["--base", "deadbeef"])

    assert excinfo.value.code == 2
    assert "unrecognized arguments: --base deadbeef" in capsys.readouterr().err


def test_missing_import_allowlist_is_limited_to_known_third_party_dependencies():
    document = type_check.tomllib.loads((Path(type_check.__file__).parents[1] / "pyproject.toml").read_text())
    allowlists = [
        override["module"]
        for override in document["tool"]["mypy"]["overrides"]
        if override.get("ignore_missing_imports")
    ]

    assert allowlists == [["yaml.*", "pandas.*", "wandb.*", "torch.*"]]
    assert all(not module.startswith("agent_tools") for module in allowlists[0])


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
