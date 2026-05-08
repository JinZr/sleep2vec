from __future__ import annotations

import json
from pathlib import Path
import typing as t

GENERATED_SIGNAL_PROVENANCE = "generated_decision_support_not_acquired_clinical_channels"


def build_generation_manifest(
    *,
    task_type: str,
    condition_modalities: t.Sequence[str],
    target_modalities: t.Sequence[str],
    diffusion_ckpt: str | Path,
    autoencoder_ckpt: str | Path,
    sampler: dict[str, t.Any],
    autoencoder_type: str,
    latent_dim: int,
    latent_frames_per_epoch: t.Mapping[str, int],
    patches_per_epoch: int,
    channel_specific: bool,
    output_files: t.Sequence[str],
) -> dict[str, t.Any]:
    return {
        "schema_version": 2,
        "artifact_type": "sleep2wave_generation",
        "signal_provenance": GENERATED_SIGNAL_PROVENANCE,
        "clinical_use": "decision_support_only",
        "task_type": task_type,
        "condition_modalities": list(condition_modalities),
        "target_modalities": list(target_modalities),
        "generated_modalities": list(target_modalities),
        "diffusion_checkpoint": str(diffusion_ckpt),
        "autoencoder_checkpoint": str(autoencoder_ckpt),
        "sampler": dict(sampler),
        "autoencoder_type": autoencoder_type,
        "latent_dim": int(latent_dim),
        "latent_frames_per_epoch": dict(latent_frames_per_epoch),
        "patches_per_epoch": int(patches_per_epoch),
        "channel_specific": bool(channel_specific),
        "output_files": list(output_files),
    }


def write_manifest(path: str | Path, manifest: dict[str, t.Any]) -> Path:
    output = Path(path)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n")
    return output


__all__ = ["GENERATED_SIGNAL_PROVENANCE", "build_generation_manifest", "write_manifest"]
