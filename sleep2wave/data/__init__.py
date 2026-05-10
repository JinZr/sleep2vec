import importlib

from sleep2wave.data.modalities import (
    CANONICAL_MODALITIES,
    MODALITY_ALIASES,
    MODALITY_SPECS,
    ModalitySpec,
    get_modality_spec,
    normalize_modality_name,
    validate_modality_sequence,
)

_GENERATIVE_EXPORTS = {
    "SLEEP2WAVE_SCHEMA_VERSION",
    "Sleep2WaveGenerativeDataset",
    "build_sample_indices_from_frame",
    "build_sample_indices_from_index",
}


def __getattr__(name):
    if name in _GENERATIVE_EXPORTS:
        generative_dataset = importlib.import_module("sleep2wave.data.generative_dataset")
        return getattr(generative_dataset, name)
    raise AttributeError(name)


__all__ = [
    "CANONICAL_MODALITIES",
    "MODALITY_ALIASES",
    "MODALITY_SPECS",
    "SLEEP2WAVE_SCHEMA_VERSION",
    "ModalitySpec",
    "Sleep2WaveGenerativeDataset",
    "build_sample_indices_from_frame",
    "build_sample_indices_from_index",
    "get_modality_spec",
    "normalize_modality_name",
    "validate_modality_sequence",
]
