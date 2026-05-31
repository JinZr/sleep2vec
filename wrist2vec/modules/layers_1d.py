import typing as t

import torch
import torch.nn as nn


class ResBlock1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        downsample: t.Union[nn.Module, None] = None,
        norm_layer: t.Callable[[int], nn.Module] = nn.BatchNorm1d,
        single_conv: bool = False,
    ) -> None:
        super().__init__()
        layers = [
            nn.Conv1d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(),
        ]
        if not single_conv:
            layers.extend(
                [
                    nn.Conv1d(out_channels, out_channels, 3, padding=1, bias=False),
                    norm_layer(out_channels),
                ]
            )

        self.stem = nn.Sequential(*layers)
        self.act = nn.ReLU()
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        return self.act(self.stem(x) + identity)
