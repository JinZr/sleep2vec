#!/usr/bin/env python3
"""Hold agent_tools to a mccabe complexity ceiling and keep its suppression ledger honest.

agent_tools is control flow, where branch count is the thing worth capping; the
model and training packages are legitimately long-bodied and are out of scope by
not being named here, rather than by a ``per-file-ignores`` entry that would
leave them unchecked forever. Running flake8 alone would not keep the ceiling
meaningful, so this runs three checks:

1. **The ceiling.** ``C901`` over ``agent_tools`` at ``MAX_COMPLEXITY``.
2. **The suppression ledger.** Functions already above the ceiling carry
   ``# noqa: C901``. Left alone, that list decays three ways: a new over-ceiling
   function can be waved through with one comment, a suppression outlives the
   complexity it was hiding, and one function can be simplified while another
   goes over in the same commit -- a swap a bare count cannot see. Caught by
   re-running with noqa disabled and requiring the live violations to be exactly
   ``SUPPRESSION_LEDGER``'s functions, on exactly the annotated lines.
3. **The embedded programs.** ``python_program_sources/*.py.src`` are fragments
   assembled by ``python_programs.source()`` and executed with ``python -c``, so
   flake8's directory walk never sees them and the fragments do not lint
   standalone (names resolve only once concatenated). Each registered program is
   assembled and checked at the same ceiling.

Both ``utils/style_check.sh`` and the ``style_check`` workflow call this, so a
clean local run means a clean CI run. It owns the ceiling value: nothing else
spells the number.

Every probe runs ``--isolated``. Inheriting ``.flake8`` would let one
``per-file-ignores`` entry or ``exclude`` pattern blind the gate and the ledger
audit at once, since both read the same configured flake8 -- the file-level
blindness this check exists to rule out, arriving through the config instead.

Growth still needs a human to extend ``SUPPRESSION_LEDGER`` or
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
#: Functions already above the ceiling, each carrying ``# noqa: C901`` on its
#: def line. Identities rather than a count, so simplifying one function while
#: another goes over does not net out to a passing check. Trailing numbers are
#: the score when grandfathered, for context only -- nothing reads them, so a
#: function may be improved without a ledger edit until it drops under the
#: ceiling. Delete entries as you fix them; a new one is a design signal, not a
#: lint to suppress.
SUPPRESSION_LEDGER = {
    ("agent_tools/adapters/embedding_extraction.py", "EmbeddingExtractionAdapter.task_issues"),  # 30
    ("agent_tools/adaptive_hparam.py", "_adaptive_step"),  # 48
    ("agent_tools/adaptive_hparam.py", "_init_adaptive_workflow_locked"),  # 26
    ("agent_tools/domain/index_csv.py", "index_summary"),  # 33
    ("agent_tools/experiment_io.py", "append_managed_text_at"),  # 28
    ("agent_tools/experiment_io.py", "conditional_atomic_replace_text_at"),  # 47
    ("agent_tools/experiment_io.py", "read_managed_output_texts_at"),  # 26
    ("agent_tools/experiment_io.py", "validate_managed_output_paths"),  # 39
    ("agent_tools/experiment_pipeline.py", "_run_attempts"),  # 27
    ("agent_tools/experiment_pipeline.py", "_validate_frozen_pipeline"),  # 29
    ("agent_tools/experiment_pipeline.py", "_validate_spec"),  # 57
    ("agent_tools/experiment_sources.py", "_remote_checkpoint_rows"),  # 28
    ("agent_tools/experiment_tracking.py", "experiment_status_snapshot"),  # 36
    ("agent_tools/experiments.py", "_managed_workspace"),  # 26
    ("agent_tools/hparam_selection.py", "resolve_hparam_candidates"),  # 47
    ("agent_tools/managed_scheduler.py", "_launch_managed_runs"),  # 47
    ("agent_tools/managed_scheduler.py", "_launch_slurm_runs"),  # 28
    ("agent_tools/managed_scheduler.py", "observe_slurm_run"),  # 44
    ("agent_tools/plans.py", "_build_plan"),  # 31
    ("agent_tools/plans.py", "evaluate_recipe"),  # 40
    ("agent_tools/research_log.py", "_normalized_research_log_entry"),  # 35
    ("agent_tools/run_artifacts.py", "read_hparam_plan"),  # 31
    ("agent_tools/run_artifacts.py", "read_registered_plan"),  # 68
    ("agent_tools/run_evidence.py", "status_row"),  # 29
}
#: Assembled blocks already above the ceiling: ``(program, block) -> scores``.
#: Spelled here rather than as a ``# noqa`` in the fragment, because a
#: fragment's line numbers do not survive concatenation -- and because a noqa
#: there would be the whole gate, there being no second check behind it.
#: Keyed per block, not per program, so a second over-ceiling block appearing
#: in a grandfathered assembly is still reported. mccabe labels an unnamed block
#: with its line, which moves whenever a fragment above it grows, so the key
#: drops that number and the score carries what identity remains: they must
#: match exactly, since a block that got simpler is a gain to record and one
#: that got worse is the thing this check is for. Two same-kind blocks of the
#: same score in one program are therefore indistinguishable -- swapping one for
#: the other reads as unchanged. Pinning the location would mean mapping
#: assembled line numbers back through the fragment offsets ``source()`` owns,
#: and that swap leaves the debt exactly as the ledger describes it.
PROGRAM_LEDGER = {
    ("experiment_io.conditional_atomic_replace_text", "TryExcept"): (46,),
    ("experiment_io.validate_managed_output_paths", "Loop"): (30,),
}

SUPPRESSION = re.compile(r"#\s*noqa:\s*C901\b")
VIOLATION = re.compile(r"^(?P<path>.+?):(?P<line>\d+):\d+: C901 '(?P<name>.+?)' is too complex \((?P<score>\d+)\)$")


def run_flake8(*arguments: str) -> str:
    result = subprocess.run(
        # --isolated: read no .flake8, so no per-file-ignores or exclude can
        # blind this probe and the ledger audit that reads the same flake8.
        [
            sys.executable,
            "-m",
            "flake8",
            "--isolated",
            "--select=C901",
            f"--max-complexity={MAX_COMPLEXITY}",
            *arguments,
        ],
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

    identities = {(path, name) for (path, _), (name, _) in live.items()}
    if added := sorted(identities - SUPPRESSION_LEDGER):
        print("\nThe suppression ledger may only shrink, and these functions are new to it:")
        for path, name in added:
            print(f'    ("{path}", "{name}"),')
        print("Simplify the function instead of grandfathering it.")
        ok = False
    if fixed := sorted(SUPPRESSION_LEDGER - identities):
        print(f"\nThese are under {MAX_COMPLEXITY} now. Delete their SUPPRESSION_LEDGER entries too:")
        for path, name in fixed:
            print(f'    ("{path}", "{name}"),')
        ok = False

    if ok:
        print(f"suppression ledger: {len(SUPPRESSION_LEDGER)} functions, none stale.")
    return ok


def block(label: str) -> str:
    """An mccabe label without the line number it carries for unnamed blocks."""
    head, _, tail = label.rpartition(" ")
    return head if tail.isdigit() else label


def check_programs() -> bool:
    """The assembled ``python -c`` programs, which flake8's walk cannot reach."""
    with tempfile.TemporaryDirectory() as tmp:
        names = {}
        for name in python_programs.registered_programs():
            path = Path(tmp) / f"{name.replace('.', '_')}.py"
            path.write_text(python_programs.source(name), encoding="utf-8")
            names[str(path)] = name
        # noqa disabled: PROGRAM_LEDGER is the only record for these programs,
        # so an annotation in a fragment would not be a suppression to audit --
        # it would be the gate itself, switched off.
        found = violations(run_flake8("--disable-noqa", tmp))

    over: dict[tuple[str, str], list[int]] = {}
    for (path, _), (label, score) in found.items():
        over.setdefault((names[path], block(label)), []).append(score)
    measured = {key: tuple(sorted(scores)) for key, scores in over.items()}

    ok = True
    for key in sorted(measured.keys() | PROGRAM_LEDGER.keys()):
        expected, actual = PROGRAM_LEDGER.get(key), measured.get(key)
        if expected == actual:
            continue
        program, name = key
        if expected is None:
            print(f"\n{program}: {name} is over the ceiling at {actual}.")
            print("Split it, or add it to PROGRAM_LEDGER with the reason it cannot be split.")
        elif actual is None:
            print(f"\n{program}: {name} is under {MAX_COMPLEXITY} now. Delete its PROGRAM_LEDGER entry.")
        else:
            print(f"\n{program}: {name} was {expected}, is now {actual}. Update its PROGRAM_LEDGER entry.")
        ok = False

    if ok:
        print(f"embedded programs: {len(names)} assembled, {len(PROGRAM_LEDGER)} grandfathered, none stale.")
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
