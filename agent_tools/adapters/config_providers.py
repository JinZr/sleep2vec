from __future__ import annotations

from typing import Any, Callable, NamedTuple

from ..domain.sex_age_summary import _looks_like_sex_age_baseline_config_data, sex_age_baseline_config_summary


class ConfigSummaryProvider(NamedTuple):
    #: Variant name that forces this provider even when the config shape does
    #: not match (None disables forcing).
    force_variant: str | None
    #: Config-shape probe on the raw loaded mapping.
    matches: Callable[[dict[str, Any]], bool]
    #: Produce the structured summary for a resolved config path.
    summarize: Callable[..., dict[str, Any]]


CONFIG_SUMMARY_PROVIDERS: tuple[ConfigSummaryProvider, ...] = (
    ConfigSummaryProvider(
        force_variant="sex_age_baseline",
        matches=_looks_like_sex_age_baseline_config_data,
        summarize=sex_age_baseline_config_summary,
    ),
)
