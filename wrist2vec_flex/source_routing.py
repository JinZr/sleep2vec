from __future__ import annotations

import typing as t

EFFECTIVE_CHANNEL_SEPARATOR = "::"


def normalize_channel_source_names(
    channel_names: t.Sequence[str],
    channel_source_names: t.Mapping[str, t.Sequence[str]] | None = None,
) -> dict[str, list[str]]:
    raw = {str(name): [str(src) for src in sources] for name, sources in dict(channel_source_names or {}).items()}
    normalized: dict[str, list[str]] = {}
    for channel_name in channel_names:
        sources = list(raw.get(str(channel_name)) or [])
        normalized[str(channel_name)] = sources if sources else [str(channel_name)]
    return normalized


def make_effective_channel_name(logical_name: str, source_name: str) -> str:
    return f"{logical_name}{EFFECTIVE_CHANNEL_SEPARATOR}{source_name}"


def parse_effective_channel_name(name: str) -> tuple[str, str | None]:
    if EFFECTIVE_CHANNEL_SEPARATOR not in name:
        return str(name), None
    logical_name, source_name = str(name).split(EFFECTIVE_CHANNEL_SEPARATOR, 1)
    return logical_name, source_name


def uses_explicit_channel_sources(channel_source_names: t.Mapping[str, t.Sequence[str]] | None) -> bool:
    for channel_name, sources in dict(channel_source_names or {}).items():
        if list(sources) != [str(channel_name)]:
            return True
    return False


def build_effective_channel_mappings(
    channel_names: t.Sequence[str],
    channel_source_names: t.Mapping[str, t.Sequence[str]] | None = None,
) -> tuple[list[str], dict[str, str], dict[str, str]]:
    normalized = normalize_channel_source_names(channel_names, channel_source_names)
    effective_names: list[str] = []
    effective_to_logical: dict[str, str] = {}
    effective_to_source: dict[str, str] = {}

    for channel_name in channel_names:
        sources = list(normalized[str(channel_name)])
        if sources == [str(channel_name)]:
            effective_names.append(str(channel_name))
            effective_to_logical[str(channel_name)] = str(channel_name)
            effective_to_source[str(channel_name)] = str(channel_name)
            continue

        for source_name in sources:
            effective_name = make_effective_channel_name(str(channel_name), str(source_name))
            effective_names.append(effective_name)
            effective_to_logical[effective_name] = str(channel_name)
            effective_to_source[effective_name] = str(source_name)

    return effective_names, effective_to_logical, effective_to_source


__all__ = [
    "EFFECTIVE_CHANNEL_SEPARATOR",
    "build_effective_channel_mappings",
    "make_effective_channel_name",
    "normalize_channel_source_names",
    "parse_effective_channel_name",
    "uses_explicit_channel_sources",
]
