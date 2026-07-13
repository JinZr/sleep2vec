# Context Bundle Contract

`agent_tools context` writes a diagnostic-only bundle containing:

- `context.json`: machine-readable task context, status, decisions, questions, commands, validation gates, warnings, and blockers.
- `context.md`: human-readable summary.
- `commands.blocked.sh`: non-runnable blocker stub because runnable commands require a managed recipe-backed plan.
- `questions.json` and `questions.md`: structured questions when consultation is required.

`context.json` includes `status`, `can_generate_commands`, `consultation_required`, `questions`, `config_summary`, `repo`, `skill`, `owners`, `relevant_docs`, `index_summary`, `preset_summary`, `expected_artifacts`, and recommended commands.
