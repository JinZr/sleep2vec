from pathlib import Path

import pytest

from hypnodata.channels import ChannelResolutionError, resolve_channels
from hypnodata.config import SignalSpec
from hypnodata.edf import EdfInventory, EdfSignalInfo


def _signal(label: str, idx: int, path: Path | None = None) -> EdfSignalInfo:
    return EdfSignalInfo(
        path=path or Path("record.edf"),
        raw_label=label,
        raw_index=idx,
        sfreq=10.0,
        unit="uV",
        n_samples=100,
        duration=10.0,
    )


def _spec(required: bool = True, candidates=None) -> SignalSpec:
    return SignalSpec(
        name="eeg",
        kind="eeg",
        required=required,
        target_sfreq=10.0,
        target_unit="uV",
        candidates=["EEG C3"] if candidates is None else candidates,
    )


def test_resolve_channels_prefers_first_matching_candidate():
    inventory = EdfInventory(
        path=Path("record.edf"),
        signals=[_signal("EEG C4", 0), _signal("EEG C3", 1)],
        duration=10.0,
        warnings=[],
    )
    spec = _spec(candidates=["EEG C3", "EEG C4"])

    selections, warnings = resolve_channels({"eeg": spec}, {"edf": inventory})

    assert warnings == []
    assert selections["eeg"].raw_label == "EEG C3"
    assert selections["eeg"].selection_reason == "label:EEG C3"


def test_required_ambiguity_fails():
    inventory = EdfInventory(
        path=Path("record.edf"),
        signals=[_signal("EEG C3", 0), _signal("EEG C3", 1)],
        duration=10.0,
        warnings=[],
    )
    spec = _spec(candidates=["EEG C3"])

    with pytest.raises(ChannelResolutionError, match="Ambiguous required channel"):
        resolve_channels({"eeg": spec}, {"edf": inventory})


def test_optional_ambiguity_warns_and_selects_deterministically():
    inventory = EdfInventory(
        path=Path("record.edf"),
        signals=[_signal("EEG C3", 1), _signal("EEG C3", 0)],
        duration=10.0,
        warnings=[],
    )
    spec = _spec(required=False, candidates=["EEG C3"])

    selections, warnings = resolve_channels({"eeg": spec}, {"edf": inventory})

    assert selections["eeg"].raw_label == "EEG C3"
    assert selections["eeg"].raw_index == 0
    assert warnings
    assert "Ambiguous optional channel" in warnings[0]


def test_optional_missing_sets_unavailable_selection():
    inventory = EdfInventory(path=Path("record.edf"), signals=[_signal("ECG", 0)], duration=10.0, warnings=[])

    selections, warnings = resolve_channels({"eeg": _spec(required=False)}, {"edf": inventory})

    assert warnings == []
    assert selections["eeg"].available is False
    assert selections["eeg"].selection_reason == "missing optional channel"


def test_required_annotation_without_candidates_stays_required():
    inventory = EdfInventory(path=Path("record.edf"), signals=[_signal("EEG C3", 0)], duration=10.0, warnings=[])
    spec = SignalSpec(
        name="stage5",
        kind="stage",
        required=True,
        target_sfreq=None,
        target_unit=None,
        candidates=[],
        epoch_sec=5,
    )

    selections, warnings = resolve_channels({"stage5": spec}, {"edf": inventory})

    assert warnings == []
    assert selections["stage5"].available is False
    assert selections["stage5"].required is True
    assert selections["stage5"].selection_reason == "missing required annotation"
