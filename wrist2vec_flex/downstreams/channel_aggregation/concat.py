import torch

from .base import ChannelAggregator


class ConcatChannelAggregator(ChannelAggregator):
    name = "concat"

    def forward(self, feats, *, channel_mask=None):
        has_L, feats = self._validate_shapes(feats)
        if channel_mask is not None:
            mask = self._normalize_channel_mask(
                channel_mask,
                batch_size=feats[0].size(0),
                device=feats[0].device,
            )
            if not mask.all():
                raise ValueError("concat channel aggregation requires all channels to be present.")
        fused = torch.cat(feats, dim=-1)
        return self._restore_rank(fused, has_L), has_L


__all__ = ["ConcatChannelAggregator"]
