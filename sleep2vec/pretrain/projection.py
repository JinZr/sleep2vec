import torch
import torch.nn as nn
import torch.nn.functional as F


class SimCLRProjectionHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 2048, out_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim, bias=False)
        self.bn2 = nn.BatchNorm1d(out_dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if h.dim() == 2:
            x = self.fc1(h)
            x = self.bn1(x)
            x = F.relu(x, inplace=True)
            x = self.fc2(x)
            x = self.bn2(x)
            z = F.normalize(x, dim=-1)
            return z

        elif h.dim() == 3:
            B, T, H = h.shape
            x = h.view(B * T, H)
            x = self.fc1(x)
            x = self.bn1(x)
            x = F.relu(x, inplace=True)
            x = self.fc2(x)
            x = self.bn2(x)
            z = F.normalize(x, dim=-1)
            return z.view(B, T, -1)

        else:
            raise ValueError(f"SimCLRProjectionHead expects 2D/3D, got shape {h.shape}")
