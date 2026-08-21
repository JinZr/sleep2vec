import os

import torch.distributed as dist


def is_rank_zero_process() -> bool:
    """Return whether this process should behave as rank zero.

    This helper is intentionally scoped to single-node launches. Multi-node
    correctness is out of scope; callers that need cluster-global rank semantics
    should use `torch.distributed.get_rank()` after distributed initialization.
    """
    for env_name in ("RANK", "SLURM_PROCID", "LOCAL_RANK", "SLURM_LOCALID"):
        rank = os.environ.get(env_name)
        if rank in (None, ""):
            continue
        try:
            return int(rank) == 0
        except ValueError:
            continue
    return True


def is_torch_distributed_ready() -> bool:
    """Return whether torch.distributed is both available and initialized."""
    return dist.is_available() and dist.is_initialized()


def get_rank_world_size() -> tuple[int, int]:
    """Return `(rank, world_size)` with `(0, 1)` fallback when dist is unavailable."""
    if is_torch_distributed_ready():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1
