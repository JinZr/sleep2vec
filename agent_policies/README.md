# Agent Consultation Policies

`consultation_policy.yaml` defines the high-impact fields that require explicit user intent before an agent may plan or run an experiment. Runtime defaults are owned by their runtime/rendering code rather than by a second policy file.

The goal is to prevent silent guessing for high-impact choices such as experiment ownership, step purpose, label selection, split usage, checkpoint selection, external-test evaluation, preset regeneration, and hyper-parameter search space.

New runnable plans also require a managed experiment workspace and semantic run names. Monitoring does not authorize launching, and stopping requires a recorded reason.

The runtime `DecisionReport` classifies checks as:

- PASS: safe to continue.
- WARN: safe to continue, but the user should be informed.
- NEEDS_USER_INPUT: stop and ask the user.
- FAIL: invalid or unsafe; cannot continue without fixing the repo/config.

`NEEDS_USER_INPUT` should exit with code 2.
`FAIL` should exit with code 1.
`PASS` and warning-only results should exit with code 0.
