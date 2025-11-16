import typing as t

import torch.nn as nn

HeadFactory = t.Callable[..., nn.Module]

HEAD_REGISTRY: dict[str, HeadFactory] = {}


def register_head(name: str):
    def decorator(factory: HeadFactory):
        if name in HEAD_REGISTRY:
            raise ValueError(f"Head '{name}' already registered.")
        HEAD_REGISTRY[name] = factory
        setattr(factory, "registry_name", name)
        return factory

    return decorator


def create_head(name: str, **kwargs) -> nn.Module:
    if name not in HEAD_REGISTRY:
        raise KeyError(
            f"Unknown head '{name}'. Available: {sorted(HEAD_REGISTRY.keys())}"
        )
    return HEAD_REGISTRY[name](**kwargs)


def available_heads() -> list[str]:
    return sorted(HEAD_REGISTRY.keys())
