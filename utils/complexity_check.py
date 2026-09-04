#!/usr/bin/env python3
"""Hold agent_tools to a mccabe complexity ceiling and keep its suppression ledger honest.

agent_tools is control flow, where branch count is the thing worth capping; the
model and training packages are legitimately long-bodied and are out of scope by
not being named here, rather than by a ``per-file-ignores`` entry that would
leave them unchecked forever. Running flake8 alone would not keep the ceiling
meaningful, so this runs three checks:

1. **The ceiling.** ``C901`` over ``agent_tools`` at ``MAX_COMPLEXITY``.
2. **The suppression ledger.** Functions already above the ceiling carry
   ``# noqa: C901``. Left alone, that list decays two ways: a new over-ceiling
   function can be waved through with one comment, and a suppression outlives
   the complexity it was hiding. Caught by re-running with noqa disabled and
   requiring the live violations and the annotations to be the same lines, and
   by holding the total to ``SUPPRESSION_CEILING``.
3. **The embedded programs.** ``python_program_sources/*.py.src`` are fragments
   assembled by ``python_programs.source()`` and executed with ``python -c``, so
   flake8's directory walk never sees them and the fragments do not lint
   standalone (names resolve only once concatenated). Each registered program is
   assembled and checked at the same ceiling.

Both ``utils/style_check.sh`` and the ``style_check`` workflow call this, so a
clean local run means a clean CI run. It owns the ceiling value: nothing else
spells the number.

Growth still needs a human to raise ``SUPPRESSION_CEILING`` or extend
``PROGRAM_LEDGER``, which is a reviewed diff rather than a silent comment --
weaker than the base-revision diff in ``utils/type_check.py``, and deliberately
so, since that ratchet needs CI to pass a base sha and this check runs in the
style job, which has no such plumbing.
"""

from __future__ import annotations

from pathlib import Path
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_tools import python_programs  # noqa: E402  (needs the repo root on sys.path)

MAX_COMPLEXITY = 25
PACKAGE = Path("agent_tools")
#: Functions already above the ceiling, carrying ``# noqa: C901``. Shrink it
#: when you simplify one of them; do not grow it. A new function over the
#: ceiling is a design signal, not a lint to suppress.
SUPPRESSION_CEILING = 24
#: Assembled programs already above the ceiling, by registered program name.
#: Same contract as the annotations above, spelled here because a fragment's
#: line numbers do not survive concatenation.
PROGRAM_LEDGER = {
    "experiment_io.conditional_atomic_replace_text",
    "experiment_io.validate_managed_output_paths",
}

SUPPRESSION = re.compile(r"#\s*noqa:\s*C901\b")
VIOLATION = re.compile(r"^(?P<path>.+?):(?P<line>\d+):\d+: C901 '(?P<name>.+?)' is too complex \((?P<score>\d+)\)$")


def run_flake8(*arguments: str) -> str:
    result = subprocess.run(
        [sys.executable, "-m", "flake8", "--select=C901", f"--max-complexity={MAX_COMPLEXITY}", *arguments],
        capture_output=True,
        text=True,
    )
    if result.returncode not in (0, 1):
        print(result.stdout or result.stderr, file=sys.stderr)
        sys.exit(f"flake8 failed to run (exit {result.returncode}).")
    return result.stdout


def violations(output: str) -> dict[tuple[str, int], tuple[str, int]]:
    """``(path, line) -> (function, score)`` for every C901 in ``output``."""
    found = {}
    for line in output.splitlines():
        match = VIOLATION.match(line)
        if match:
            found[(match["path"], int(match["line"]))] = (match["name"], int(match["score"]))
    return found


def annotations() -> set[tuple[str, int]]:
    """Every ``# noqa: C901`` in the package, as ``(path, line)``."""
    return {
        (str(path), number)
        for path in sorted(PACKAGE.rglob("*.py"))
        for number, text in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
        if SUPPRESSION.search(text)
    }


def check_ceiling() -> dict[tuple[str, int], tuple[str, int]]:
    """Reports, and returns, the functions flake8 itself still fails on."""
    reported = violations(output := run_flake8(str(PACKAGE)))
    if reported:
        print("Functions over the complexity ceiling:")
        print(output, end="")
        print("Split the function, or -- if it is genuinely irreducible -- add # noqa: C901 to")
        print("its def line and raise SUPPRESSION_CEILING in the same commit, so growing the")
        print("ledger stays a reviewed decision rather than a silent one.")
    else:
        print(f"ceiling: no agent_tools function over {MAX_COMPLEXITY} without a suppression.")
    return reported


def check_ledger(reported: dict[tuple[str, int], tuple[str, int]]) -> bool:
    live = violations(run_flake8("--disable-noqa", str(PACKAGE)))
    suppressed = annotations()
    ok = True

    stale = sorted(suppressed - set(live))
    if stale:
        print(f"\nStale suppressions -- these lines are under {MAX_COMPLEXITY} now, so delete the # noqa: C901:")
        for path, number in stale:
            print(f"    {path}:{number}")
        ok = False

    # Over the ceiling, not annotated, and yet flake8's own run stayed quiet:
    # something other than a # noqa is silencing C901 here, which is exactly the
    # file-level blindness this check exists to rule out. Violations flake8 did
    # report are check_ceiling's to explain, not a second finding.
    unannotated = sorted(set(live) - suppressed - set(reported))
    if unannotated:
        print("\nSilently exempt from C901 -- no # noqa here, yet flake8 did not report it:")
        for path, number in unannotated:
            print(f"    {path}:{number}  {live[(path, number)][0]}")
        ok = False

    if len(live) > SUPPRESSION_CEILING:
        print(f"\nThe suppression ledger may only shrink: {len(live)} entries, ceiling {SUPPRESSION_CEILING}.")
        print("Simplify the function instead of grandfathering it.")
        ok = False
    elif len(live) < SUPPRESSION_CEILING:
        print(f"\nThe ledger shrank to {len(live)}. Lower SUPPRESSION_CEILING in {__file__} to hold the gain.")
        ok = False
    elif ok:
        print(f"suppression ledger: {len(live)} functions, none stale.")
    return ok


def check_programs() -> bool:
    """The assembled ``python -c`` programs, which flake8's walk cannot reach."""
    with tempfile.TemporaryDirectory() as tmp:
        names = {}
        for name in python_programs.registered_programs():
            path = Path(tmp) / f"{name.replace('.', '_')}.py"
            path.write_text(python_programs.source(name), encoding="utf-8")
            names[str(path)] = name
        found = violations(run_flake8(tmp))

    over = {names[path] for path, _ in found}
    ok = True
    if new := sorted(over - PROGRAM_LEDGER):
        print(f"\nAssembled programs over the ceiling: {', '.join(new)}.")
        print("Split the program, or add it to PROGRAM_LEDGER with the reason it cannot be.")
        ok = False
    if fixed := sorted(PROGRAM_LEDGER - over):
        print(f"\nStale PROGRAM_LEDGER entries -- these are under {MAX_COMPLEXITY} now: {', '.join(fixed)}.")
        ok = False
    if ok:
        print(f"embedded programs: {len(names)} assembled, {len(over)} grandfathered, none stale.")
    return ok


def main() -> int:
    print(f"== agent_tools complexity (ceiling {MAX_COMPLEXITY}) ==")
    # Every check runs: a contributor fixing one report should see the rest in
    # the same pass rather than one per push.
    reported = check_ceiling()
    results = [not reported, check_ledger(reported), check_programs()]
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
