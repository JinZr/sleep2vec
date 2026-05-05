from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

torch = pytest.importorskip("torch")

from sleep2wave.export.artifacts import write_generation_artifacts
from sleep2wave.export.manifest import GENERATED_SIGNAL_PROVENANCE, build_generation_manifest
from sleep2wave.inference.uncertainty import compute_uncertainty


def test_export_writes_generation_artifacts(tmp_path: Path):
    config_path = tmp_path / "config_source.yaml"
    config_path.write_text(yaml.safe_dump({"recipe": "sleep2wave", "stage": "inference"}))
    generated = {"eeg": torch.randn(3, 2, 1, 4)}
    uncertainty = compute_uncertainty(generated)
    masks = {
        "availability": {"eeg": torch.ones(2, dtype=torch.bool)},
        "quality": {"eeg": torch.ones(2)},
        "corruption": {"eeg": torch.zeros(2, 1, 4, dtype=torch.bool)},
        "condition": {"eeg": torch.zeros(2, dtype=torch.bool)},
        "target": {"eeg": torch.ones(2, dtype=torch.bool)},
    }
    manifest = build_generation_manifest(
        task_type="translation",
        condition_modalities=["ecg"],
        target_modalities=["eeg"],
        diffusion_ckpt="diffusion.ckpt",
        autoencoder_ckpt="autoencoder.ckpt",
        sampler={"name": "ddim", "steps": 2, "eta": 0.0, "num_samples": 3},
        output_files=["generated.npz"],
    )
    args = argparse.Namespace(config=config_path, diffusion_ckpt="diffusion.ckpt")

    output_dir = write_generation_artifacts(
        tmp_path / "out",
        generated=generated,
        uncertainty=uncertainty,
        masks=masks,
        metadata_rows=[{"subject_id": "s1", "night_id": "n1"}],
        manifest=manifest,
        config_path=config_path,
        args=args,
    )

    assert (output_dir / "manifest.json").exists()
    assert (output_dir / "config.yaml").exists()
    assert (output_dir / "cli_args.yaml").exists()
    assert (output_dir / "metadata.jsonl").exists()

    generated_npz = np.load(output_dir / "generated.npz")
    uncertainty_npz = np.load(output_dir / "uncertainty.npz")
    masks_npz = np.load(output_dir / "masks.npz")
    manifest_payload = json.loads((output_dir / "manifest.json").read_text())

    assert generated_npz["generated/eeg"].shape == (3, 2, 1, 4)
    assert uncertainty_npz["mean/eeg"].shape == (2, 1, 4)
    assert uncertainty_npz["std/eeg"].shape == (2, 1, 4)
    assert uncertainty_npz["sample_count/eeg"].tolist() == [3]
    assert masks_npz["target/eeg"].tolist() == [True, True]
    assert manifest_payload["signal_provenance"] == GENERATED_SIGNAL_PROVENANCE
