import torch
import torch.nn as nn


class LayerMix(nn.Module):
    """Learned softmax weighting over transformer layers."""

    def __init__(self, num_layers: int, n_mods: int = 1, shared_across_modalities: bool = False):
        super().__init__()
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}.")
        if n_mods <= 0:
            raise ValueError(f"n_mods must be positive, got {n_mods}.")
        self.num_layers = int(num_layers)
        self.shared_across_modalities = bool(shared_across_modalities)
        weight_mods = 1 if self.shared_across_modalities else n_mods
        self.weight = nn.Parameter(torch.zeros(weight_mods, self.num_layers))

    def _weights(self, mod_idx: int) -> torch.Tensor:
        if self.shared_across_modalities:
            idx = 0
        else:
            if mod_idx < 0 or mod_idx >= self.weight.size(0):
                raise ValueError(f"mod_idx {mod_idx} out of range for {self.weight.size(0)} modalities.")
            idx = mod_idx
        return torch.softmax(self.weight[idx], dim=0)

    def mix(self, layer_stack: torch.Tensor, mod_idx: int = 0) -> torch.Tensor:
        if layer_stack.size(0) != self.num_layers:
            raise ValueError(f"Expected {self.num_layers} layers for mixing, got {layer_stack.size(0)}.")
        weights = self._weights(mod_idx).view(self.num_layers, *([1] * (layer_stack.dim() - 1)))
        return (weights * layer_stack).sum(dim=0)


__all__ = ["LayerMix"]
