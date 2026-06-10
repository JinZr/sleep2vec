# Agent Consultation Policies

These policies define when an agent must stop and ask for user confirmation before running or planning an experiment.

The goal is to prevent silent guessing for high-impact choices such as label selection, split usage, checkpoint selection, external-test evaluation, preset regeneration, and hyper-parameter search space.

Agent tools should classify checks as:

- PASS: safe to continue.
- WARN: safe to continue, but the user should be informed.
- NEEDS_USER_INPUT: stop and ask the user.
- FAIL: invalid or unsafe; cannot continue without fixing the repo/config.

`NEEDS_USER_INPUT` should exit with code 2.
`FAIL` should exit with code 1.
`PASS` and warning-only results should exit with code 0.
