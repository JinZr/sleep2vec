# Verification Gates

Prefer targeted pytest files for contract changes, `python utils/check_configs.py`
for config changes, and short smoke checks for runtime wiring. Follow the
relevant owner gate in [AGENTS.md](../../AGENTS.md#testing-guidelines) and the
user-authorized scope; these evidence rules add no full-suite or remote benchmark requirement.

For test failures, retain the original command, Git revision and relevant local
changes, interpreter/environment, worker count, failing node IDs, and complete
failure reason. For consultation assertions, inspect the issue fields, messages,
and evidence rather than inferring a cause from `exit_code` alone.

Use bounded, focused reruns to distinguish serial and parallel behavior. A
targeted or serial pass does not erase a prior full-suite failure or establish
its root cause. Inspect the failed wait, synchronization, and process cleanup
before treating a larger timeout as a fix; do not repeatedly rerun the full
suite merely to obtain a green result.

Report CI coverage from the [current workflow](../../.github/workflows/unit_tests.yml)
and the actual run, including relevant ignored, deselected, or skipped tests.
A green job is evidence only for what it executed, not for the entire local suite.
Keep static validation, mock/fake-transport tests, and real remote or cluster
execution distinct when describing results.
