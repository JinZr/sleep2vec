from __future__ import annotations

from dataclasses import dataclass

EPOCH_SEC = 30


@dataclass(frozen=True)
class ModalitySpec:
    name: str
    sample_rate_hz: int
    frames_per_epoch: int
    frequency_group: str


CANONICAL_MODALITIES = (
    "eeg",
    "eog",
    "emg",
    "ecg",
    "airflow",
    "belt",
    "spo2",
    "ibi",
    "resp",
)

_HIGH_FREQUENCY_MODALITIES = {"eeg", "eog", "emg", "ecg"}
_LOW_FREQUENCY_MODALITIES = {"airflow", "belt", "spo2", "ibi", "resp"}


def _build_spec(name: str) -> ModalitySpec:
    if name in _HIGH_FREQUENCY_MODALITIES:
        sample_rate_hz = 128
        frequency_group = "high_frequency"
    elif name in _LOW_FREQUENCY_MODALITIES:
        sample_rate_hz = 4
        frequency_group = "low_frequency"
    else:
        raise ValueError(f"Unknown sleep2wave modality: {name}")
    return ModalitySpec(
        name=name,
        sample_rate_hz=sample_rate_hz,
        frames_per_epoch=EPOCH_SEC * sample_rate_hz,
        frequency_group=frequency_group,
    )


MODALITY_SPECS = {name: _build_spec(name) for name in CANONICAL_MODALITIES}

MODALITY_ALIASES = {
    "eeg_original": "eeg",
    "eog_original": "eog",
    "emg_original": "emg",
    "ecg_original": "ecg",
    "resp_nasal_original": "airflow",
    "resp_original": "belt",
    "heartbeat": "ibi",
    "breath": "resp",
}


def normalize_modality_name(name: str) -> str:
    if not isinstance(name, str) or not name:
        raise ValueError("Modality name must be a non-empty string.")
    if name in MODALITY_SPECS:
        return name
    if name in MODALITY_ALIASES:
        return MODALITY_ALIASES[name]
    raise ValueError(f"Unknown sleep2wave modality: {name}")


def get_modality_spec(name: str) -> ModalitySpec:
    return MODALITY_SPECS[normalize_modality_name(name)]


def validate_modality_sequence(modalities: list[str] | tuple[str, ...], *, allow_aliases: bool = False) -> list[str]:
    if not isinstance(modalities, (list, tuple)) or not modalities:
        raise ValueError("Modality sequence must be a non-empty list.")

    normalized: list[str] = []
    seen: set[str] = set()
    for name in modalities:
        if allow_aliases:
            canonical = normalize_modality_name(name)
        else:
            if name not in MODALITY_SPECS:
                raise ValueError(f"sleep2wave configs must use canonical modality names. Got: {name}")
            canonical = name
        if canonical in seen:
            raise ValueError(f"Duplicate sleep2wave modality: {canonical}")
        seen.add(canonical)
        normalized.append(canonical)
    return normalized


__all__ = [
    "CANONICAL_MODALITIES",
    "EPOCH_SEC",
    "MODALITY_ALIASES",
    "MODALITY_SPECS",
    "ModalitySpec",
    "get_modality_spec",
    "normalize_modality_name",
    "validate_modality_sequence",
]
