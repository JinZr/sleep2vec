from __future__ import annotations

from dataclasses import dataclass

from hypnodata.config import SignalSpec
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
) -> tuple[dict[str, ChannelSelection], list[str]]:
    selections: dict[str, ChannelSelection] = {}
    warnings: list[str] = []
    for canonical, spec in signals.items():
        selection = _resolve_one(canonical, spec, inventories)
        if selection.warning:
            warnings.append(selection.warning)
        selections[canonical] = selection
    return selections, warnings


def _resolve_one(
    canonical: str,
    spec: SignalSpec,
    inventories: dict[str, EdfInventory],
) -> ChannelSelection:
    matches = []
    for candidate_idx, candidate_label in enumerate(spec.candidates):
        for file_key, inventory in sorted(inventories.items()):
            for signal in inventory.signals:
                if signal.raw_label == candidate_label:
                    matches.append((candidate_idx, file_key, signal.raw_index, signal.raw_label, signal))

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

    best_idx = min(item[0] for item in matches)
    best = [item for item in matches if item[0] == best_idx]
    best.sort(key=lambda item: (item[1], item[2], item[3]))
    if len(best) > 1 and spec.required:
        labels = [item[4].raw_label for item in best]
        raise ChannelResolutionError(
            f"Ambiguous required channel {canonical!r} for candidate {spec.candidates[best_idx]!r}: {labels}"
        )
    _, file_key, _, _, signal = best[0]
    warning = None
    if len(best) > 1:
        warning = (
            f"Ambiguous optional channel {canonical!r} for candidate {spec.candidates[best_idx]!r}; "
            f"selected {signal.raw_label!r} from {file_key!r} deterministically."
        )
    return _selection_from_signal(canonical, spec, signal, warning)


def _selection_from_signal(
    canonical: str,
    spec: SignalSpec,
    signal: EdfSignalInfo,
    warning: str | None,
) -> ChannelSelection:
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
        selection_reason=f"label:{signal.raw_label}",
        warning=warning,
    )
