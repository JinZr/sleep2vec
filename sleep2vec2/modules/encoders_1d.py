import typing as t

import torch.nn as nn

from .layers_1d import ResBlock1d


class UNetEncoderHigh(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        res_single_conv: bool = False,
        norm_layer: t.Callable[[int], nn.Module] = nn.BatchNorm1d,
        device: str = "cuda",
    ):
        super().__init__()
        self.device = device
        self._norm_layer = norm_layer
        self.inplanes = 32
        self.res_single_conv = res_single_conv
        self.feature_dim = feature_dim

        assert self.feature_dim % 8 == 0, "elf.feature_dim % 8 != 0"

        self.high_sr_signal_entrance = nn.Sequential(
            nn.Conv1d(1, self.inplanes, 7, stride=2, padding=3, bias=False),
            norm_layer(self.inplanes),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        self.feature_transform = nn.Sequential(
            self._make_layer(64, blocks=1, stride=2),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            self._make_layer(128, blocks=1, stride=2),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            self._make_layer(256, blocks=1, stride=2),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            self._make_layer(512, blocks=1, stride=3),
            self._make_layer(self.feature_dim, blocks=1, stride=5),
        )

        self.total_params = sum(p.numel() for p in self.parameters())
        print(f"Total parameters: {self.total_params}")
        self.trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Trainable parameters: {self.trainable_params}")

    def _make_layer(self, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes, 1, stride=stride, bias=False),
                self._norm_layer(planes),
            )
        layers = [
            ResBlock1d(
                self.inplanes,
                planes,
                stride,
                downsample,
                single_conv=self.res_single_conv,
            )
        ]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(
                ResBlock1d(
                    planes,
                    planes,
                    1,
                    norm_layer=self._norm_layer,
                    single_conv=self.res_single_conv,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.to(self.device)
        B, L, D = x.shape
        x = x.view(B * L, 1, D)
        assert D == 3840
        x = self.high_sr_signal_entrance(x)
        x = self.feature_transform(x)
        x = x.view(B, L, self.feature_dim)
        return x


class UNetEncoderLow(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        res_single_conv: bool = False,
        norm_layer: t.Callable[[int], nn.Module] = nn.BatchNorm1d,
        device: str = "cuda",
    ):
        super().__init__()
        self.device = device
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.res_single_conv = res_single_conv
        self.feature_dim = feature_dim

        assert self.feature_dim % 8 == 0, "self.feature_dim % 8 != 0"

        self.low_sr_signal_entrance = nn.Sequential(
            nn.Conv1d(1, self.inplanes, 7, stride=2, padding=3, bias=False),
            norm_layer(self.inplanes),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        self.feature_transform = nn.Sequential(
            self._make_layer(64, blocks=1, stride=2),
            self._make_layer(256, blocks=1, stride=3),
            self._make_layer(512, blocks=1, stride=5),
        )

        self.total_params = sum(p.numel() for p in self.parameters())
        print(f"Total parameters: {self.total_params}")
        self.trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Trainable parameters: {self.trainable_params}")

    def _make_layer(self, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes, 1, stride=stride, bias=False),
                self._norm_layer(planes),
            )
        layers = [
            ResBlock1d(
                self.inplanes,
                planes,
                stride,
                downsample,
                single_conv=self.res_single_conv,
            )
        ]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(
                ResBlock1d(
                    planes,
                    planes,
                    1,
                    norm_layer=self._norm_layer,
                    single_conv=self.res_single_conv,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.to(self.device)
        B, L, D = x.shape
        x = x.view(B * L, 1, D)
        assert D == 120
        x = self.low_sr_signal_entrance(x)
        x = self.feature_transform(x)
        x = x.view(B, L, self.feature_dim)
        return x
