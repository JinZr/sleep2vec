from __future__ import annotations

import logging
from pathlib import Path
import re
import typing as t

import torch

_EPOCH_RE = re.compile(r"epoch[=\-](\d+)")


def _parse_epoch(path: Path) -> int | None:
    match = _EPOCH_RE.search(path.name)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _load_state_dict(path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    else:
        state = ckpt

    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint {path} does not contain a state_dict.")
    return state


def select_checkpoints(
    ckpt_dir: Path,
    *,
    end_ckpt: Path | None,
    num_ckpts: int,
) -> list[Path]:
    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

    ckpts = sorted(ckpt_dir.glob("*.ckpt"))
    if not ckpts:
        raise ValueError(f"No .ckpt files found under {ckpt_dir}")

    if end_ckpt is not None:
        end_ckpt = Path(end_ckpt)
        if not end_ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found: {end_ckpt}")

    epoch_pairs: list[tuple[int, Path]] = []
    for path in ckpts:
        epoch = _parse_epoch(path)
        if epoch is not None:
            epoch_pairs.append((epoch, path))

    if epoch_pairs:
        epoch_pairs.sort(key=lambda item: item[0])
        if end_ckpt is not None:
            end_epoch = _parse_epoch(end_ckpt)
            if end_epoch is not None:
                epoch_pairs = [item for item in epoch_pairs if item[0] <= end_epoch]
            else:
                try:
                    end_idx = [p for _, p in epoch_pairs].index(end_ckpt)
                    epoch_pairs = epoch_pairs[: end_idx + 1]
                except ValueError:
                    pass
        selected = [p for _, p in epoch_pairs][-num_ckpts:]
        if len(selected) == num_ckpts:
            return selected

    ckpts_sorted = sorted(ckpts, key=lambda p: p.stat().st_mtime)
    if end_ckpt is not None:
        end_mtime = end_ckpt.stat().st_mtime
        ckpts_sorted = [p for p in ckpts_sorted if p.stat().st_mtime <= end_mtime + 1e-6]
    selected = ckpts_sorted[-num_ckpts:]
    if len(selected) < num_ckpts:
        raise ValueError(f"Not enough checkpoints to average: requested {num_ckpts}, found {len(selected)}")
    return selected


def average_checkpoints(
    filenames: t.Sequence[Path],
    *,
    device: torch.device | str = torch.device("cpu"),
) -> dict[str, torch.Tensor]:
    if not filenames:
        raise ValueError("No checkpoints provided for averaging.")

    device = torch.device(device)
    state = _load_state_dict(Path(filenames[0]), device=device)
    avg: dict[str, torch.Tensor] = {}

    for key, value in state.items():
        if torch.is_tensor(value):
            avg[key] = value.clone()

    for path in filenames[1:]:
        state_dict = _load_state_dict(Path(path), device=device)
        for key, value in avg.items():
            if key not in state_dict:
                raise KeyError(f"Missing key '{key}' in checkpoint {path}")
            avg[key] += state_dict[key]

    n = len(filenames)
    for key, value in avg.items():
        if value.is_floating_point():
            value.div_(n)
        else:
            value //= n

    logging.info("Averaged %d checkpoints", n)
    return avg


__all__ = ["average_checkpoints", "select_checkpoints"]
