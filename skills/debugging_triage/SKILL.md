# Skill: debugging_triage

## When to use

Use for `debug` tasks that collect failure context before changing code or rerunning long jobs.

## Required inputs

Start with the failed command, its exit status when known, stdout/stderr, and
relevant artifacts. Include a recipe, config path, or run directory only when
it belongs to the failure. For test failures, retain the execution identity and
original failure evidence described in [verification gates](../_shared/verification_gates.md).

## First information-gathering commands

- Use `python -m agent_tools repo-summary --json` when repository context is missing.
- Use `python -m agent_tools config-summary --config <config> --json` for relevant config failures, not as a prerequisite for unrelated CLI or test failures.

## Decision checklist

Separate a reported consultation blocker from an input/parser error, internal
exception, timeout, or unknown operation outcome. Read the actual issue's
field, message, and evidence; `FAIL` alone does not identify the cause. Follow
the [consultation result contract](../../doc/agent_contracts/task_recipe.md#consultation-and-diagnostics)
before interpreting reports or exit codes. Then locate the config, data,
checkpoint, runtime, environment, or test owner supported by that evidence.

## Stop-and-consult gates

The agent must stop and ask the user before continuing if any high-impact decision is missing, ambiguous, conflicting, or marked as `ASK_USER`.

An exception or missing report is not itself a missing scientific decision.
Do not invent user questions or treat diagnosis as authorization to launch,
change the environment, or repair frozen artifacts.

## Canonical commands

Use relevant read-only summaries and the existing error output first, then
targeted tests or scoped smoke checks. Do not automatically repeat `doctor`
or state-creating `plan` for a clearer receipt. For a slow or disconnected
operation, follow the [takeover evidence rules](../../doc/agent_contracts/experiment_workspace.md#takeover-and-continue-execution)
before deciding whether another attempt is appropriate.

## Expected artifacts

An evidence-backed diagnosis that distinguishes confirmed failures, possible
causes, and remaining unknowns. Include questions only for unresolved decisions
that actually require user input; a context bundle is optional diagnostic evidence.

## Validation gates

Run the smallest in-scope check that reproduces or explains the failure, using
the shared [verification gates](../_shared/verification_gates.md). A successful
retry is evidence about that retry, not proof of the original failure's cause.

## Common failure modes

Missing paths, stale generated artifacts, unsupported backend, and environment dependency gaps.
For parallel-only test failures, inspect the exact failed wait or assertion;
do not assume every consultation failure is a subprocess timeout.

## Relevant owners and index pages

Owner: `agent-tooling-maintainer`. Index: shared [Codex navigation](../../doc/codex_index/README.md).
