# Context Bundle Contract

`agent_tools context` writes a bundle containing:

- `context.json`: machine-readable task context, status, decisions, questions, commands, validation gates, warnings, and blockers.
- `context.md`: human-readable summary.
- `commands.sh`: executable command plan only when consultation gates pass.
- `commands.blocked.sh`: non-runnable blocker stub when user input is required.
- `questions.json` and `questions.md`: structured questions when consultation is required.

`context.json` includes `status`, `can_generate_commands`, `consultation_required`, `questions`, `config_summary`, `repo`, `skill`, `owners`, `relevant_docs`, `index_summary`, `preset_summary`, `expected_artifacts`, and recommended commands.
