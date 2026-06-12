from __future__ import annotations

from typing import Callable

from sleep2stat.config import AnalyzerConfig, ReducerConfig

AnalyzerBuilder = Callable[[AnalyzerConfig], object]
ReducerBuilder = Callable[[ReducerConfig], object]

ANALYZER_REGISTRY: dict[str, AnalyzerBuilder] = {}
REDUCER_REGISTRY: dict[str, ReducerBuilder] = {}


def register_analyzer(name: str):
    def decorator(cls):
        if name in ANALYZER_REGISTRY:
            raise ValueError(f"Analyzer {name!r} is already registered.")
        ANALYZER_REGISTRY[name] = cls
        return cls

    return decorator


def register_reducer(name: str):
    def decorator(cls):
        if name in REDUCER_REGISTRY:
            raise ValueError(f"Reducer {name!r} is already registered.")
        REDUCER_REGISTRY[name] = cls
        return cls

    return decorator


def create_analyzer(config: AnalyzerConfig):
    try:
        builder = ANALYZER_REGISTRY[config.type]
    except KeyError as exc:
        raise ValueError(f"Unknown sleep2stat analyzer type: {config.type!r}.") from exc
    return builder(config)


def create_reducer(config: ReducerConfig):
    try:
        builder = REDUCER_REGISTRY[config.type]
    except KeyError as exc:
        raise ValueError(f"Unknown sleep2stat reducer type: {config.type!r}.") from exc
    return builder(config)
