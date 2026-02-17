"""sleep2vec package initializer.

This file ensures the repository can be imported as a regular package, even
when CLI entrypoints like ``sleep2vec/finetune.py`` are executed directly.
"""

import sys as _sys

# Keep absolute imports (from sleep2vec.*) working within this mirrored package.
_sys.modules.setdefault("sleep2vec", _sys.modules[__name__])

__all__ = []
