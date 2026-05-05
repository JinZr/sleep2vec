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
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--diffusion-ckpt", type=Path, required=True)
    parser.add_argument("--autoencoder-ckpt", type=Path, default=None)
    data_group = parser.add_mutually_exclusive_group()
    data_group.add_argument("--preset-path", type=Path, default=None)
    data_group.add_argument("--index", type=Path, default=None)
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["restoration", "imputation", "translation", "partial_full"],
    )
    parser.add_argument("--condition-modalities", nargs="+", required=True)
    parser.add_argument("--target-modalities", nargs="+", required=True)
    parser.add_argument("--corruption-name", type=str, default=None)
    parser.add_argument("--corruption-kwargs", type=str, default=None)
    parser.add_argument("--condition-mask-npz", type=Path, default=None)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stride-epochs", type=int, default=1)
    parser.add_argument(
        "--overlap-fusion",
        type=str,
        default="mean",
        choices=["mean", "median", "uncertainty_weighted"],
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def _slug(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "unknown"


def _group_samples(
    *,
    preset_path: Path | str | None,
    index: Path | str | None,
    context_epochs: int,
    stride_epochs: int,
):
    from sleep2wave.data.generative_dataset import build_sample_indices_from_index, normalize_sample_index

    if (preset_path is None) == (index is None):
        raise ValueError("Batch generation requires exactly one preset path or index path.")
    if preset_path is not None:
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
        preset_path=args.preset_path or config.data.preset_path,
        index=args.index or config.data.index,
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
