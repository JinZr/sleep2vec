from pathlib import Path

import pytest

from hypnodata.channels import ChannelResolutionError, resolve_channels
from hypnodata.config import CandidateSpec, SignalSpec
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
        candidates=[CandidateSpec(label="EEG C3", priority=1)] if candidates is None else candidates,
    )


def test_resolve_channels_prefers_higher_priority_exact_alias():
    inventory = EdfInventory(
        path=Path("record.edf"),
        signals=[_signal("EEG C4", 0), _signal("EEG C3", 1)],
        duration=10.0,
        warnings=[],
    )
    spec = _spec(candidates=[CandidateSpec(label="EEG C4", priority=1), CandidateSpec(label="EEG C3", priority=5)])

    selections, warnings = resolve_channels({"eeg": spec}, {"edf": inventory})

    assert warnings == []
    assert selections["eeg"].raw_label == "EEG C3"
    assert selections["eeg"].selection_reason == "label:EEG C3; priority=5"


def test_resolve_channels_supports_regex_candidate():
    inventory = EdfInventory(
        path=Path("record.edf"),
        signals=[_signal("C3-M2", 0)],
        duration=10.0,
        warnings=[],
    )
    spec = _spec(candidates=[CandidateSpec(regex=r"^C3", priority=2)])

    selections, _ = resolve_channels({"eeg": spec}, {"edf": inventory})

    assert selections["eeg"].raw_label == "C3-M2"
    assert selections["eeg"].selection_reason == "regex:C3-M2; priority=2"


def test_required_ambiguity_fails():
    inventory = EdfInventory(
        path=Path("record.edf"),
        signals=[_signal("EEG C3", 0), _signal("EEG C4", 1)],
        duration=10.0,
        warnings=[],
    )
    spec = _spec(candidates=[CandidateSpec(regex=r"^EEG", priority=1)])

    with pytest.raises(ChannelResolutionError, match="Ambiguous required channel"):
        resolve_channels({"eeg": spec}, {"edf": inventory})


def test_optional_ambiguity_warns_and_selects_deterministically():
    inventory = EdfInventory(
        path=Path("record.edf"),
        signals=[_signal("EEG C4", 1), _signal("EEG C3", 0)],
        duration=10.0,
        warnings=[],
    )
    spec = _spec(required=False, candidates=[CandidateSpec(regex=r"^EEG", priority=1)])

    selections, warnings = resolve_channels({"eeg": spec}, {"edf": inventory})

    assert selections["eeg"].raw_label == "EEG C3"
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
