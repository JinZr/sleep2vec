from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Callable

from hypnodata.config import CandidateSpec, SignalSpec
from hypnodata.edf import EdfInventory, EdfSignalInfo


@dataclass(frozen=True)
class ChannelSelection:
    canonical_channel: str
    kind: str
    available: bool
    required: bool
    raw_file: str | None
    raw_label: str | None
    raw_index: int | None
    raw_sfreq: float | None
    target_sfreq: float | None
    raw_unit: str | None
    target_unit: str | None
    raw_n_samples: int | None
    selection_reason: str
    warning: str | None = None


class ChannelResolutionError(ValueError):
    pass


def resolve_channels(
    signals: dict[str, SignalSpec],
    inventories: dict[str, EdfInventory],
    scorer: Callable[[str, SignalSpec, CandidateSpec, EdfSignalInfo], float | None] | None = None,
) -> tuple[dict[str, ChannelSelection], list[str]]:
    selections: dict[str, ChannelSelection] = {}
    warnings: list[str] = []
    for canonical, spec in signals.items():
        selection = _resolve_one(canonical, spec, inventories, scorer=scorer)
        if selection.warning:
            warnings.append(selection.warning)
        selections[canonical] = selection
    return selections, warnings


def _resolve_one(
    canonical: str,
    spec: SignalSpec,
    inventories: dict[str, EdfInventory],
    *,
    scorer: Callable[[str, SignalSpec, CandidateSpec, EdfSignalInfo], float | None] | None,
) -> ChannelSelection:
    matches = []
    for file_key, inventory in sorted(inventories.items()):
        for signal in inventory.signals:
            for candidate_idx, candidate in enumerate(spec.candidates):
                match_type = ""
                if candidate.label is not None and signal.raw_label == candidate.label:
                    match_type = "label"
                elif candidate.regex is not None and re.search(candidate.regex, signal.raw_label):
                    match_type = "regex"
                if match_type:
                    adapter_score = 0.0 if scorer is None else scorer(canonical, spec, candidate, signal)
                    score = float(candidate.priority) + float(adapter_score or 0.0)
                    matches.append(
                        (
                            score,
                            candidate.priority,
                            adapter_score,
                            file_key,
                            signal.raw_index,
                            candidate_idx,
                            signal,
                            match_type,
                        )
                    )

    if not matches:
        if spec.required and spec.candidates:
            raise ChannelResolutionError(f"Missing required channel {canonical!r}.")
        return ChannelSelection(
            canonical_channel=canonical,
            kind=spec.kind,
            available=False,
            required=spec.required,
            raw_file=None,
            raw_label=None,
            raw_index=None,
            raw_sfreq=None,
            target_sfreq=spec.target_sfreq,
            raw_unit=None,
            target_unit=spec.target_unit,
            raw_n_samples=None,
            selection_reason="missing required annotation" if spec.required else "missing optional channel",
        )

    best_score = max(item[0] for item in matches)
    best = [item for item in matches if item[0] == best_score]
    best.sort(key=lambda item: (item[3], item[4], item[5], item[6].raw_label))
    if len(best) > 1 and spec.required:
        labels = [item[6].raw_label for item in best]
        raise ChannelResolutionError(f"Ambiguous required channel {canonical!r} at score {best_score:g}: {labels}")
    _, priority, adapter_score, file_key, _, _, signal, match_type = best[0]
    warning = None
    if len(best) > 1:
        warning = (
            f"Ambiguous optional channel {canonical!r} at score {best_score:g}; "
            f"selected {signal.raw_label!r} from {file_key!r} deterministically."
        )
    return _selection_from_signal(canonical, spec, signal, priority, adapter_score, match_type, warning)


def _selection_from_signal(
    canonical: str,
    spec: SignalSpec,
    signal: EdfSignalInfo,
    priority: int,
    adapter_score: float | None,
    match_type: str,
    warning: str | None,
) -> ChannelSelection:
    reason = f"{match_type}:{signal.raw_label}; priority={priority}"
    if adapter_score not in (None, 0, 0.0):
        reason = f"{reason}; adapter_score={float(adapter_score):g}"
    return ChannelSelection(
        canonical_channel=canonical,
        kind=spec.kind,
        available=True,
        required=spec.required,
        raw_file=str(signal.path),
        raw_label=signal.raw_label,
        raw_index=signal.raw_index,
        raw_sfreq=signal.sfreq,
        target_sfreq=spec.target_sfreq or signal.sfreq,
        raw_unit=signal.unit,
        target_unit=spec.target_unit,
        raw_n_samples=signal.n_samples,
        selection_reason=reason,
        warning=warning,
    )
