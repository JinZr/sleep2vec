from __future__ import annotations

import importlib
from typing import Any, Callable


def resolve_callable(reference: str, builtins: dict[str, Callable[..., Any]]) -> Callable[..., Any]:
    if reference in builtins:
        return builtins[reference]
    if ":" not in reference:
        raise ValueError(f"Expected built-in key or module:function import path, got {reference!r}.")
    module_name, function_name = reference.split(":", 1)
    if not module_name or not function_name:
        raise ValueError(f"Invalid import path: {reference!r}. Expected module:function.")
    module = importlib.import_module(module_name)
    target = getattr(module, function_name)
    if not callable(target):
        raise ValueError(f"Import path {reference!r} did not resolve to a callable.")
    return target
