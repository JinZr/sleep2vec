import os

import torch.distributed as dist


def is_rank_zero_process() -> bool:
    """Return whether this process should behave as rank zero.

    This helper is intentionally scoped to single-node launches. It only inspects
    process-level `RANK` / `LOCAL_RANK` environment variables and treats missing
    or invalid values as rank zero. Multi-node correctness is out of scope for
    this helper; callers that need cluster-global rank semantics should use
    `torch.distributed.get_rank()` after distributed initialization instead.
    """
    for env_name in ("RANK", "LOCAL_RANK"):
        rank = os.environ.get(env_name)
        if rank in (None, ""):
            continue
        try:
            return int(rank) == 0
        except ValueError:
            return True
    return True


def is_torch_distributed_ready() -> bool:
    """Return whether torch.distributed is both available and initialized."""
    return dist.is_available() and dist.is_initialized()


def get_rank_world_size() -> tuple[int, int]:
    """Return `(rank, world_size)` with `(0, 1)` fallback when dist is unavailable."""
    if is_torch_distributed_ready():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1
