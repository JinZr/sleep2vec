"""Config-shape providers: config families claimed by the config rather than by the task.

Layer 1. ``configs`` walks two tables, in order, before falling back to the
generic finetune summary body, so a config is summarized under the family it
actually belongs to even when the caller names a different task or variant:

1. ``CONFIG_SHAPE_SUMMARIES`` runs first: a probe on the raw loaded mapping
   paired with a summary that takes only the config path. A shape claim here
   wins over any variant a provider below would force.
2. ``CONFIG_SUMMARY_PROVIDERS``: a provider either forces a variant outright or
   probes the raw loaded mapping, and its summary takes the finetune-family
   validation keywords.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, NamedTuple

from ..domain.sex_age_summary import _looks_like_sex_age_baseline_config_data, sex_age_baseline_config_summary
from ..models import ConfigSummary, SexAgeConfigSummary
from .sleep2stat import looks_like_sleep2stat_config_data, sleep2stat_config_summary

CONFIG_SHAPE_SUMMARIES: tuple[tuple[Callable[[dict[str, Any]], bool], Callable[[str | Path], ConfigSummary]], ...] = (
    (looks_like_sleep2stat_config_data, sleep2stat_config_summary),
)


class ConfigSummaryProvider(NamedTuple):
    #: Variant name that forces this provider even when the config shape does
    #: not match (None disables forcing).
    force_variant: str | None
    #: Config-shape probe on the raw loaded mapping.
    matches: Callable[[dict[str, Any]], bool]
    #: Produce the structured summary for a resolved config path.
    summarize: Callable[..., SexAgeConfigSummary]


CONFIG_SUMMARY_PROVIDERS: tuple[ConfigSummaryProvider, ...] = (
    ConfigSummaryProvider(
        force_variant="sex_age_baseline",
        matches=_looks_like_sex_age_baseline_config_data,
        summarize=sex_age_baseline_config_summary,
    ),
)
