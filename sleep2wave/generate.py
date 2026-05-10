from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import typing as t

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sleep2wave.autoencoders.model import Sleep2WaveAutoencoder
from sleep2wave.diffusion.model import Sleep2WaveDiffusionTransformer
from sleep2wave.generative.config import SamplerConfig, Sleep2WaveConfig


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate sleep2wave PSG channels from incomplete recordings.")
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
    parser.add_argument("--output-dir", type=Path, default=None, help="Output artifact directory.")
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


def _load_diffusion_model(
    checkpoint_path: Path,
    config: Sleep2WaveConfig,
    *,
    device: torch.device,
) -> Sleep2WaveDiffusionTransformer:
    if config.diffusion is None:
        raise ValueError("diffusion block is required for generation.")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Diffusion checkpoint not found: {checkpoint_path}")

    model = Sleep2WaveDiffusionTransformer.from_config(config.diffusion, modalities=config.modalities.all)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state_dict, dict):
        raise ValueError("Diffusion checkpoint must contain a state_dict mapping.")

    target_keys = set(model.state_dict().keys())
    if any(key.startswith("model.") for key in state_dict):
        filtered_state = {
            key[len("model.") :]: value
            for key, value in state_dict.items()
            if key.startswith("model.") and key[len("model.") :] in target_keys
        }
    else:
        filtered_state = {key: value for key, value in state_dict.items() if key in target_keys}
    if not filtered_state:
        raise ValueError("Diffusion checkpoint does not contain sleep2wave diffusion model weights.")
    model.load_state_dict(filtered_state, strict=True)
    model.to(device)
    model.eval()
    return model


def _to_device(mapping: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in mapping.items()}


def _decode_generated_latents(
    autoencoder: Sleep2WaveAutoencoder,
    generated_latents: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    decoded: dict[str, torch.Tensor] = {}
    for modality, latents in generated_latents.items():
        if latents.dim() != 6:
            raise ValueError(
                f"Generated latents for '{modality}' must have shape [num_samples, B, E, C, L, D], "
                f"got {tuple(latents.shape)}."
            )
        num_samples, batch_size, context_epochs, channels, latent_frames, latent_dim = latents.shape
        flat = latents.reshape(num_samples * batch_size, context_epochs, channels, latent_frames, latent_dim)
        decoded_flat = autoencoder.decode_latents({modality: flat})[modality]
        decoded[modality] = decoded_flat.reshape(num_samples, batch_size, context_epochs, *decoded_flat.shape[2:])
    return decoded


def _resolve_generation_data_source(
    args: argparse.Namespace, data_config
) -> tuple[str, Path | str | None, Path | str | None, Path | str | None, Path | str | None]:
    if data_config.backend == "kaldi":
        if args.index is not None:
            raise ValueError("Generation with data.backend=kaldi does not support --index.")
        backend = "kaldi"
        preset_path = args.preset_path
        index = None
        kaldi_data_root = data_config.kaldi_data_root
        kaldi_manifest = data_config.kaldi_manifest
    elif args.preset_path is not None or args.index is not None:
        backend = "npz"
        preset_path = args.preset_path
        index = args.index
        kaldi_data_root = None
        kaldi_manifest = None
    else:
        backend = data_config.backend
        preset_path = data_config.preset_path
        index = data_config.index
        kaldi_data_root = data_config.kaldi_data_root
        kaldi_manifest = data_config.kaldi_manifest
    if backend == "npz" and (preset_path is None) == (index is None):
        raise ValueError("Generation requires exactly one preset path or index path.")
    if backend == "kaldi" and (kaldi_data_root is None or kaldi_manifest is None):
        raise ValueError("Generation with data.backend=kaldi requires kaldi_data_root and kaldi_manifest.")
    return backend, preset_path, index, kaldi_data_root, kaldi_manifest


def _parse_corruption_kwargs(raw: str | None) -> dict[str, t.Any]:
    if raw is None:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("--corruption-kwargs must be a JSON object.") from exc
    if not isinstance(parsed, dict):
        raise ValueError("--corruption-kwargs must be a JSON object.")
    for key, value in parsed.items():
        if not isinstance(key, str) or not key:
            raise ValueError("--corruption-kwargs keys must be non-empty strings.")
        if isinstance(value, (dict, list)):
            raise ValueError(f"--corruption-kwargs field '{key}' must be a scalar value.")
    return parsed


def _resolve_inference_corruption_specs(
    *,
    config: Sleep2WaveConfig,
    args: argparse.Namespace,
    task,
) -> dict[str, tuple[str, dict[str, t.Any]]]:
    corruption_name = getattr(args, "corruption_name", None)
    corruption_kwargs_raw = getattr(args, "corruption_kwargs", None)
    if task.task_type in {"restoration", "imputation"}:
        corruption_modalities = task.target_modalities
    else:
        corruption_modalities = task.condition_modalities
    if corruption_kwargs_raw is not None and corruption_name is None:
        raise ValueError("--corruption-kwargs requires --corruption-name.")
    if corruption_name is not None:
        from sleep2wave.data.corruptions import CORRUPTION_REGISTRY

        if corruption_name not in CORRUPTION_REGISTRY:
            raise ValueError(f"--corruption-name must be one of {sorted(CORRUPTION_REGISTRY)}. Got: {corruption_name}")
        kwargs = _parse_corruption_kwargs(corruption_kwargs_raw)
        return {modality: (corruption_name, kwargs) for modality in corruption_modalities}

    if config.inference is None:
        return {}
    policy = config.inference.corruptions.for_task(task.task_type)
    if policy is None:
        return {}
    specs: dict[str, tuple[str, dict[str, t.Any]]] = {}
    for modality in corruption_modalities:
        spec = policy.for_modality(modality)
        if spec is not None:
            modality_offset = config.modalities.all.index(modality)
            seed = int(getattr(args, "seed", 0)) * len(config.modalities.all) + modality_offset
            choice = spec.select(seed=seed)
            specs[modality] = (choice.name, dict(choice.kwargs))
    return specs


def _activate_requested_generation_targets(
    availability_mask: dict[str, torch.Tensor],
    task,
) -> dict[str, torch.Tensor]:
    if task.task_type not in {"translation", "partial_full"}:
        return availability_mask
    adjusted = dict(availability_mask)
    for modality in task.target_modalities:
        adjusted[modality] = torch.ones_like(adjusted[modality], dtype=torch.bool)
    return adjusted


def _batch_metadata_rows(batch: dict[str, t.Any]) -> list[dict[str, t.Any]]:
    rows: list[dict[str, t.Any]] = []
    batch_size = batch["epoch_index"].shape[0]
    for idx in range(batch_size):
        row = {key: values[idx] for key, values in batch["metadata"].items()}
        row["start_epoch"] = int(batch["epoch_index"][idx, 0].item())
        row["end_epoch"] = int(batch["epoch_index"][idx, -1].item()) + 1
        rows.append(row)
    return rows


def _collect_generation_windows(
    *,
    config: Sleep2WaveConfig,
    args: argparse.Namespace,
    model: Sleep2WaveDiffusionTransformer,
    autoencoder: Sleep2WaveAutoencoder,
    sampler_config: SamplerConfig,
    task,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, dict[str, torch.Tensor]], list[int], list[dict[str, t.Any]]]:
    from sleep2wave.data.generative_dataset import Sleep2WaveGenerativeDataset
    from sleep2wave.data.modalities import CANONICAL_MODALITIES
    from sleep2wave.diffusion.samplers import build_sampler
    from sleep2wave.diffusion.task_masks import build_patch_condition_availability
    from sleep2wave.inference.sliding_window import validate_single_night

    backend, preset_path, index, kaldi_data_root, kaldi_manifest = _resolve_generation_data_source(args, config.data)
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.stride_epochs <= 0:
        raise ValueError("--stride-epochs must be positive.")
    corruption_specs = _resolve_inference_corruption_specs(config=config, args=args, task=task)
    dataset_split = None
    if backend == "kaldi" and preset_path is None:
        dataset_split = "test"

    dataset = Sleep2WaveGenerativeDataset(
        backend=backend,
        preset_path=preset_path,
        index=index,
        kaldi_data_root=kaldi_data_root,
        kaldi_manifest=kaldi_manifest,
        split=dataset_split,
        context_epochs=config.data.context_epochs,
        stride_epochs=args.stride_epochs,
        condition_modalities=task.condition_modalities,
        target_modalities=task.target_modalities,
        task_type=task.task_type,
        corruption_specs=corruption_specs,
        condition_mask_npz=getattr(args, "condition_mask_npz", None),
        seed=args.seed,
    )
    loader = dataset.dataloader(batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    sampler = build_sampler(
        sampler_config,
        diffusion_steps=config.diffusion.diffusion_steps,
        beta_schedule=config.diffusion.beta_schedule,
    )

    generated_windows: dict[str, list[torch.Tensor]] = {modality: [] for modality in task.target_modalities}
    mask_windows: dict[str, dict[str, list[torch.Tensor]]] = {
        "availability": {modality: [] for modality in CANONICAL_MODALITIES},
        "quality": {modality: [] for modality in CANONICAL_MODALITIES},
        "corruption": {modality: [] for modality in CANONICAL_MODALITIES},
    }
    metadata_rows: list[dict[str, t.Any]] = []
    start_epochs: list[int] = []

    torch.manual_seed(args.seed)
    with torch.no_grad():
        for batch in loader:
            observed_signals = _to_device(batch["observed_signals"], device)
            observed_latents = autoencoder(observed_signals).latents
            condition_latents = {modality: observed_latents[modality] for modality in task.condition_modalities}
            availability_mask = _to_device(batch["availability_mask"], device)
            sampler_availability_mask = _activate_requested_generation_targets(availability_mask, task)
            quality_mask = _to_device(batch["quality_mask"], device)
            channel_mask = _to_device(batch["channel_mask"], device)
            corruption_mask = _to_device(batch["corruption_mask"], device)
            condition_availability = build_patch_condition_availability(
                sampler_availability_mask,
                corruption_mask,
                task,
                patches_per_epoch=config.diffusion.patches_per_epoch,
            )
            output = sampler.sample(
                model,
                condition_latents=condition_latents,
                task=task,
                availability_mask=sampler_availability_mask,
                quality_mask=quality_mask,
                night_position=batch["night_position"].to(device),
                condition_availability_mask=condition_availability,
                channel_mask=channel_mask,
            )
            decoded = _decode_generated_latents(autoencoder, output.generated_latents)
            for modality, values in decoded.items():
                generated_windows[modality].append(values.cpu())
            for modality in CANONICAL_MODALITIES:
                mask_windows["availability"][modality].append(batch["availability_mask"][modality].cpu())
                mask_windows["quality"][modality].append(batch["quality_mask"][modality].cpu())
                mask_windows["corruption"][modality].append(batch["corruption_mask"][modality].cpu())
            rows = _batch_metadata_rows(batch)
            metadata_rows.extend(rows)
            start_epochs.extend(row["start_epoch"] for row in rows)

    validate_single_night(metadata_rows)
    generated = {modality: torch.cat(values, dim=1) for modality, values in generated_windows.items()}
    masks = {
        family: {modality: torch.cat(values, dim=0) for modality, values in modality_values.items()}
        for family, modality_values in mask_windows.items()
    }
    return generated, masks, start_epochs, metadata_rows


def _fuse_generated(
    windows: dict[str, torch.Tensor],
    start_epochs: list[int],
    *,
    mode: str,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    from sleep2wave.inference.sliding_window import fuse_overlapping_windows

    fused: dict[str, torch.Tensor] = {}
    epoch_index: torch.Tensor | None = None
    for modality, values in windows.items():
        result = fuse_overlapping_windows(values, start_epochs, mode=mode)
        fused[modality] = result.values
        if epoch_index is None:
            epoch_index = result.epoch_index
        elif not torch.equal(epoch_index, result.epoch_index):
            raise ValueError("Generated modalities produced inconsistent epoch indices.")
    if epoch_index is None:
        raise ValueError("No generated windows were produced.")
    return fused, epoch_index


def _fuse_masks(
    windows: dict[str, dict[str, torch.Tensor]],
    start_epochs: list[int],
    *,
    epoch_index: torch.Tensor,
    condition_modalities: t.Sequence[str],
    target_modalities: t.Sequence[str],
) -> dict[str, dict[str, torch.Tensor]]:
    from sleep2wave.data.modalities import CANONICAL_MODALITIES
    from sleep2wave.inference.sliding_window import fuse_mask_windows

    fused: dict[str, dict[str, torch.Tensor]] = {"availability": {}, "quality": {}, "corruption": {}}
    for modality in CANONICAL_MODALITIES:
        fused["availability"][modality] = fuse_mask_windows(
            windows["availability"][modality],
            start_epochs,
            mode="any",
        ).values
        fused["quality"][modality] = fuse_mask_windows(
            windows["quality"][modality],
            start_epochs,
            mode="mean",
        ).values
        fused["corruption"][modality] = fuse_mask_windows(
            windows["corruption"][modality],
            start_epochs,
            mode="any",
        ).values

    total_epochs = int(epoch_index.numel())
    fused["condition"] = {
        modality: torch.full((total_epochs,), modality in condition_modalities, dtype=torch.bool)
        for modality in CANONICAL_MODALITIES
    }
    fused["target"] = {
        modality: torch.full((total_epochs,), modality in target_modalities, dtype=torch.bool)
        for modality in CANONICAL_MODALITIES
    }
    return fused


def run_generation(args: argparse.Namespace) -> Path:
    from sleep2wave.autoencoders.checkpoints import load_sleep2wave_autoencoder_checkpoint
    from sleep2wave.data.modalities import validate_modality_sequence
    from sleep2wave.diffusion.tasks import build_generation_task
    from sleep2wave.export.artifacts import write_generation_artifacts
    from sleep2wave.export.manifest import build_generation_manifest
    from sleep2wave.generative.config import load_sleep2wave_config
    from sleep2wave.inference.uncertainty import compute_uncertainty

    config = load_sleep2wave_config(args.config)
    if config.stage != "inference":
        raise ValueError("sleep2wave.generate requires stage=inference config.")
    if config.diffusion is None or config.sampler is None or config.export is None:
        raise ValueError("diffusion, sampler, and export blocks are required for generation.")
    if config.data.context_epochs != config.diffusion.context_epochs:
        raise ValueError("data.context_epochs must match diffusion.context_epochs for generation.")

    condition_modalities = validate_modality_sequence(args.condition_modalities, allow_aliases=False)
    target_modalities = validate_modality_sequence(args.target_modalities, allow_aliases=False)
    task = build_generation_task(
        args.task,
        condition_modalities=condition_modalities,
        target_modalities=target_modalities,
        auxiliary_restoration_token=(
            config.diffusion.auxiliary_restoration_token if args.task in {"restoration", "imputation"} else False
        ),
    )
    sampler_config = config.sampler
    if args.num_samples is not None:
        if args.num_samples <= 0:
            raise ValueError("--num-samples must be positive.")
        sampler_config = replace(sampler_config, num_samples=args.num_samples)

    autoencoder_ckpt = args.autoencoder_ckpt or config.diffusion.autoencoder_checkpoint
    if autoencoder_ckpt is None:
        raise ValueError("An autoencoder checkpoint is required for waveform generation.")
    output_dir = args.output_dir or config.export.output_dir
    device = torch.device(args.device)

    autoencoder = load_sleep2wave_autoencoder_checkpoint(
        autoencoder_ckpt,
        latent_dim=config.diffusion.latent_dim,
        latent_frames_per_epoch=config.diffusion.latent_frames_per_epoch,
        modalities=config.modalities.all,
        device=device,
    )
    diffusion_model = _load_diffusion_model(args.diffusion_ckpt, config, device=device)
    generated_windows, mask_windows, start_epochs, metadata_rows = _collect_generation_windows(
        config=config,
        args=args,
        model=diffusion_model,
        autoencoder=autoencoder,
        sampler_config=sampler_config,
        task=task,
        device=device,
    )
    generated, epoch_index = _fuse_generated(
        generated_windows,
        start_epochs,
        mode=args.overlap_fusion,
    )
    masks = _fuse_masks(
        mask_windows,
        start_epochs,
        epoch_index=epoch_index,
        condition_modalities=task.condition_modalities,
        target_modalities=task.target_modalities,
    )
    uncertainty = compute_uncertainty(generated)
    manifest = build_generation_manifest(
        task_type=task.task_type,
        condition_modalities=task.condition_modalities,
        target_modalities=task.target_modalities,
        diffusion_ckpt=args.diffusion_ckpt,
        autoencoder_ckpt=autoencoder_ckpt,
        sampler={
            "name": sampler_config.name,
            "steps": sampler_config.steps,
            "eta": sampler_config.eta,
            "num_samples": sampler_config.num_samples,
            "overlap_fusion": args.overlap_fusion,
        },
        autoencoder_type="temporal_conv",
        latent_dim=config.diffusion.latent_dim,
        latent_frames_per_epoch=config.diffusion.latent_frames_per_epoch,
        patches_per_epoch=config.diffusion.patches_per_epoch,
        channel_specific=True,
        output_files=[
            "generated.npz",
            "uncertainty.npz",
            "masks.npz",
            "metadata.jsonl",
            "config.yaml",
            "cli_args.yaml",
        ],
    )
    for row in metadata_rows:
        row["generated_epoch_index_start"] = int(epoch_index[0].item())
        row["generated_epoch_index_end"] = int(epoch_index[-1].item()) + 1
    return write_generation_artifacts(
        output_dir,
        generated=generated,
        uncertainty=uncertainty,
        masks=masks,
        metadata_rows=metadata_rows,
        manifest=manifest,
        config_path=args.config,
        args=args,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_generation(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
