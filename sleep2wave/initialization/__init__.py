from __future__ import annotations

import importlib

_SLEEP2VEC2_EXPORTS = {
    "Sleep2Vec2InitializationReport",
    "load_sleep2vec2_initialization",
}


def __getattr__(name):
    if name in _SLEEP2VEC2_EXPORTS:
        module = importlib.import_module("sleep2wave.initialization.sleep2vec2")
        return getattr(module, name)
    raise AttributeError(name)


__all__ = [
    "Sleep2Vec2InitializationReport",
    "load_sleep2vec2_initialization",
]
