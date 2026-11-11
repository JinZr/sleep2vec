"""sleep2vec-specific config, sidecar, preset, and index summaries.

Import submodules explicitly (``from .domain.finetune_summary import ...``); never
aggregate through this package (``from .domain import ...``). Re-exporting the
``index_csv`` consumer here would make any ``agent_tools.domain.*`` import trigger
``configs`` before the target leaf finishes defining, a partial-import cycle.
"""
