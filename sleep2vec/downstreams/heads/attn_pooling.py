import torch
import torch.nn as nn


class AttnPooling(nn.Module):
    def __init__(self, d, heads=1, temp=1.0, dropout=0.0):
        super().__init__()
        self.W = nn.Linear(d, d, bias=True)
        self.q = nn.Parameter(torch.randn(heads, d) * 0.02)
        self.heads = heads
        self.temp = temp
        self.drop = nn.Dropout(dropout)

    def forward(self, H, mask):
        Ht = torch.tanh(self.W(H))
        scores = torch.einsum("bld,hd->blh", Ht, self.q) / self.temp
        scores = scores.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        A = torch.softmax(scores, dim=1)
        A = self.drop(A)
        Z = torch.einsum("blh,bld->bhd", A, H)
        Z = Z.reshape(H.size(0), -1)
        return Z, A


__all__ = ["AttnPooling"]
