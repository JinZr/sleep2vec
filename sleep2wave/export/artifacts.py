from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import typing as t

import numpy as np
import torch

from sleep2wave.common import dump_cli_args_yaml
from sleep2wave.export.manifest import write_manifest
from sleep2wave.inference.uncertainty import ModalityUncertainty


def _to_numpy(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    return value.detach().cpu().numpy()


def _jsonable(value: t.Any) -> t.Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def write_generation_artifacts(
    output_dir: str | Path,
    *,
    generated: dict[str, torch.Tensor],
    uncertainty: dict[str, ModalityUncertainty],
    masks: dict[str, dict[str, torch.Tensor]],
    metadata_rows: t.Sequence[dict[str, t.Any]],
    manifest: dict[str, t.Any],
    config_path: str | Path,
    args: argparse.Namespace,
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    generated_payload = {f"generated/{modality}": _to_numpy(values) for modality, values in generated.items()}
    np.savez(output_path / "generated.npz", **generated_payload)

    uncertainty_payload: dict[str, np.ndarray] = {}
    for modality, values in uncertainty.items():
        uncertainty_payload[f"mean/{modality}"] = _to_numpy(values.mean)
        uncertainty_payload[f"std/{modality}"] = _to_numpy(values.std)
        uncertainty_payload[f"sample_count/{modality}"] = _to_numpy(values.sample_count)
        uncertainty_payload[f"high_uncertainty_mask/{modality}"] = _to_numpy(values.high_uncertainty_mask)
    np.savez(output_path / "uncertainty.npz", **uncertainty_payload)

    masks_payload: dict[str, np.ndarray] = {}
    for family, modality_masks in masks.items():
        for modality, values in modality_masks.items():
            masks_payload[f"{family}/{modality}"] = _to_numpy(values)
    np.savez(output_path / "masks.npz", **masks_payload)

    with (output_path / "metadata.jsonl").open("w") as f:
        for row in metadata_rows:
            f.write(json.dumps({key: _jsonable(value) for key, value in row.items()}, sort_keys=True) + "\n")

    shutil.copy2(config_path, output_path / "config.yaml")
    dump_cli_args_yaml(args, output_path / "cli_args.yaml")
    write_manifest(output_path / "manifest.json", manifest)
    return output_path


__all__ = ["write_generation_artifacts"]
