# Agent Skills

The Codex index maps code ownership and reusable implementations.

The skills in this folder define repeatable task procedures: what information to gather, which command to run, what to validate, which artifacts to expect, and which subagent owner should review the work.

Use `python -m agent_tools skills --list` to list available skills.
Use `python -m agent_tools skills --validate` to validate this folder.
Use `python -m agent_tools context ...` to gather task-specific facts. Runnable commands require a recipe-backed `agent_tools plan`.
