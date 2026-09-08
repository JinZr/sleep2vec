"""Frozen top-level re-export of ``domain.index_csv``.

``plan_context``, ``cli``, and external importers reach the index summary
through this path. The shim exists only to keep that spelling stable and should
be dropped once they import the domain module directly.
"""

from __future__ import annotations

from .domain.index_csv import IndexSummary, index_summary

__all__ = ["IndexSummary", "index_summary"]
