"""Layer 1: the task-adapter protocol and registry.

Import ``TaskAdapter`` from ``adapters.base`` and the registry helpers from
``adapters.registry``. The package re-exports nothing, so importing one adapter
module does not load the registry and, through it, every other adapter.
Per-task adapters are registered by ``registry`` in a deliberate order; callers
should use the registry helpers rather than importing adapter implementations
directly.
"""
