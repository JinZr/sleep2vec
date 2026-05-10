from __future__ import annotations

import argparse
from pathlib import Path
import pickle
import re
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sleep2wave generation once per subject/night.")
    parser.add_argument("--config", type=Path, required=True, help="Sleep2Wave inference YAML config.")
    parser.add_argument("--diffusion-ckpt", type=Path, required=True, help="Sleep2Wave diffusion checkpoint path.")
    parser.add_argument("--autoencoder-ckpt", type=Path, default=None, help="Optional autoencoder checkpoint override.")
    data_group = parser.add_mutually_exclusive_group()
    data_group.add_argument("--preset-path", type=Path, default=None, help="Generation preset pickle path.")
    data_group.add_argument("--index", type=Path, default=None, help="Generation index CSV path.")
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["restoration", "imputation", "translation", "partial_full"],
        help="Generation task type.",
    )
    parser.add_argument(
        "--condition-modalities",
        nargs="+",
        required=True,
        help="Observed modalities used as conditions.",
    )
    parser.add_argument("--target-modalities", nargs="+", required=True, help="Modalities to generate.")
    parser.add_argument("--corruption-name", type=str, default=None, help="Optional inference corruption name.")
    parser.add_argument("--corruption-kwargs", type=str, default=None, help="JSON object of corruption parameters.")
    parser.add_argument("--condition-mask-npz", type=Path, default=None, help="Optional external condition mask NPZ.")
    parser.add_argument("--num-samples", type=int, default=None, help="Number of generated samples per window.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Batch generation output directory.")
    parser.add_argument("--stride-epochs", type=int, default=1, help="Sliding-window stride in 30-second epochs.")
    parser.add_argument(
        "--overlap-fusion",
        type=str,
        default="mean",
        choices=["mean", "median", "uncertainty_weighted"],
        help="Method used to fuse overlapping generated windows.",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Generation dataloader batch size.")
    parser.add_argument("--num-workers", type=int, default=0, help="Generation dataloader workers.")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device used for generation.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for generation sampling.")
    return parser.parse_args(argv)


def _slug(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "unknown"


def _group_samples(
    *,
    backend: str,
    preset_path: Path | str | None,
    index: Path | str | None,
    kaldi_data_root: Path | str | None,
    kaldi_manifest: Path | str | None,
    context_epochs: int,
    stride_epochs: int,
):
    from sleep2wave.data.generative_dataset import (
        _load_kaldi_samples,
        build_sample_indices_from_index,
        normalize_sample_index,
    )

    if backend == "kaldi":
        if index is not None:
            raise ValueError("Batch generation with data.backend=kaldi does not support --index.")
        if preset_path is not None:
            with Path(preset_path).open("rb") as f:
                samples = [normalize_sample_index(item) for item in pickle.load(f)]
        else:
            samples = _load_kaldi_samples(
                kaldi_data_root,
                kaldi_manifest,
                split="test",
            )
    elif (preset_path is None) == (index is None):
        raise ValueError("Batch generation requires exactly one preset path or index path.")
    elif preset_path is not None:
        with Path(preset_path).open("rb") as f:
            samples = [normalize_sample_index(item) for item in pickle.load(f)]
    else:
        samples = build_sample_indices_from_index(
            index,
            context_epochs=context_epochs,
            stride_epochs=stride_epochs,
        )
    grouped = {}
    for sample in samples:
        subject_id = sample.payload.get("subject_id", sample.metadata.get("subject_id", "unknown"))
        night_id = sample.payload.get("night_id", sample.metadata.get("night_id", "unknown"))
        grouped.setdefault((subject_id, night_id), []).append(sample)
    return grouped


def run_batch_generation(args: argparse.Namespace) -> list[Path]:
    from sleep2wave.generate import run_generation
    from sleep2wave.generative.config import load_sleep2wave_config

    config = load_sleep2wave_config(args.config)
    if config.stage != "inference" or config.data is None:
        raise ValueError("sleep2wave.generate_batch requires a stage=inference config.")
    grouped = _group_samples(
        backend=config.data.backend,
        preset_path=args.preset_path or config.data.preset_path,
        index=args.index or config.data.index,
        kaldi_data_root=config.data.kaldi_data_root,
        kaldi_manifest=config.data.kaldi_manifest,
        context_epochs=config.data.context_epochs,
        stride_epochs=args.stride_epochs,
    )
    preset_dir = args.output_dir / "_batch_presets"
    preset_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for (subject_id, night_id), samples in grouped.items():
        group_name = f"subject-{_slug(subject_id)}_night-{_slug(night_id)}"
        group_preset = preset_dir / f"{group_name}.pkl"
        with group_preset.open("wb") as f:
            pickle.dump(samples, f)
        generation_args = argparse.Namespace(
            config=args.config,
            diffusion_ckpt=args.diffusion_ckpt,
            autoencoder_ckpt=args.autoencoder_ckpt,
            preset_path=group_preset,
            index=None,
            task=args.task,
            condition_modalities=args.condition_modalities,
            target_modalities=args.target_modalities,
            corruption_name=getattr(args, "corruption_name", None),
            corruption_kwargs=getattr(args, "corruption_kwargs", None),
            condition_mask_npz=getattr(args, "condition_mask_npz", None),
            num_samples=args.num_samples,
            output_dir=args.output_dir / group_name,
            stride_epochs=args.stride_epochs,
            overlap_fusion=args.overlap_fusion,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=args.device,
            seed=args.seed,
        )
        outputs.append(run_generation(generation_args))
    return outputs


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_batch_generation(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "parse_args", "run_batch_generation"]
