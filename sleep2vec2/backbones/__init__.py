"""Backbone architectures for the sleep2vec2 recipe.

Unlike the original sleep2vec recipe, sleep2vec2 uses a local, encoder-only
RoFormer implementation (no Hugging Face `transformers` dependency).

Importing this package registers all available backbones in
``sleep2vec2.registry``.
"""

# Import backbones for registration side-effects.
from . import roformer as _roformer  # noqa: F401

__all__ = []
