# Context Bundle Contract

`agent_tools context` writes a diagnostic-only bundle containing:

- `context.json`: machine-readable task context, inputs, questions, command recommendations, validation gates, warnings, and blockers.
- `context.md`: human-readable summary.
- `commands.sh` and `validation.sh`: executable-format diagnostic previews written only when no blocking issue exists; they do not authorize execution.
- `commands.blocked.sh`: non-runnable blocker stub because runnable commands require a managed recipe-backed plan.
- `questions.json` and `questions.md`: structured blocking issues, including questions when consultation is required.

`context.json` includes `task`, `status`, `can_generate_commands`, `consultation_required`, `questions`, `repo`, `skill`, `owners`, `relevant_docs`, `inputs`, `config_summary`, `index_summary`, `preset_summary`, `expected_artifacts`, `recommended_commands`, `validation_commands`, `warnings`, and `blocking_issues`.
