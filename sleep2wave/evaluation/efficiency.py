from __future__ import annotations

import typing as t

import numpy as np


def _first_epoch_count(generated_npz: np.lib.npyio.NpzFile) -> int:
    for key in generated_npz.files:
        if not key.startswith("generated/"):
            continue
        values = generated_npz[key]
        if values.ndim >= 2:
            return int(values.shape[1])
    return 0


def summarize_generation_efficiency(
    manifest: dict[str, t.Any],
    generated_npz: np.lib.npyio.NpzFile,
    uncertainty_npz: np.lib.npyio.NpzFile,
) -> dict[str, dict[str, t.Any]]:
    sampler = manifest.get("sampler", {})
    if not isinstance(sampler, dict):
        sampler = {}
    generated_modalities = manifest.get("generated_modalities", manifest.get("target_modalities", []))
    generated_elements = {
        key.removeprefix("generated/"): int(np.prod(generated_npz[key].shape))
        for key in generated_npz.files
        if key.startswith("generated/")
    }
    sample_counts = {
        key.removeprefix("sample_count/"): int(np.asarray(uncertainty_npz[key]).reshape(-1)[0])
        for key in uncertainty_npz.files
        if key.startswith("sample_count/")
    }
    metrics: dict[str, t.Any] = {
        "sampler_name": sampler.get("name"),
        "sampler_steps": sampler.get("steps"),
        "num_samples": sampler.get("num_samples"),
        "generated_modalities": list(generated_modalities) if isinstance(generated_modalities, list) else [],
        "epoch_count": _first_epoch_count(generated_npz),
        "generated_elements": generated_elements,
        "sample_counts": sample_counts,
    }
    if "elapsed_seconds" in manifest:
        metrics["elapsed_seconds"] = manifest["elapsed_seconds"]
        epoch_count = metrics["epoch_count"]
        if isinstance(epoch_count, int) and epoch_count > 0:
            metrics["seconds_per_epoch"] = float(manifest["elapsed_seconds"]) / epoch_count
    return {"all": metrics}


__all__ = ["summarize_generation_efficiency"]
