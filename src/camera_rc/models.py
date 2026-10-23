"""Small, randomly initialized baselines; no network or weight downloads.

These deliberately modest networks are training/evaluation baselines, not
vehicle controllers or reproductions of a published autonomous-driving model.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        # GroupNorm works with the batch size of one often needed on a laptop.
        groups = min(8, out_channels)
        while out_channels % groups:
            groups -= 1
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
        )


class TinyUNet(nn.Module):
    def __init__(self, out_channels: int, width: int, positive: bool = False):
        super().__init__()
        self.positive = positive
        self.enc1 = ConvBlock(3, width)
        self.enc2 = ConvBlock(width, width * 2)
        self.enc3 = ConvBlock(width * 2, width * 4)
        self.dec2 = ConvBlock(width * 6, width * 2)
        self.dec1 = ConvBlock(width * 3, width)
        self.head = nn.Conv2d(width, out_channels, 1)

    def forward(self, image: torch.Tensor, state: torch.Tensor | None = None) -> torch.Tensor:
        x1 = self.enc1(image * 2.0 - 1.0)
        x2 = self.enc2(F.avg_pool2d(x1, 2))
        x3 = self.enc3(F.avg_pool2d(x2, 2))
        x = self.dec2(torch.cat((F.interpolate(x3, size=x2.shape[-2:], mode="bilinear", align_corners=False), x2), dim=1))
        x = self.dec1(torch.cat((F.interpolate(x, size=x1.shape[-2:], mode="bilinear", align_corners=False), x1), dim=1))
        output = self.head(x)
        return F.softplus(output) + 1e-3 if self.positive else output


class TinyTrajectory(nn.Module):
    """Predict K future (forward metres, left metres, speed m/s) values.

    State is the *current* measured (speed m/s, yaw rate rad/s). No future
    state or future image is consumed. Forecast intervals come from the data.
    """

    def __init__(self, horizon: int, width: int):
        super().__init__()
        self.horizon = horizon
        self.encoder = nn.Sequential(
            ConvBlock(3, width), nn.AvgPool2d(2),
            ConvBlock(width, width * 2), nn.AvgPool2d(2),
            ConvBlock(width * 2, width * 4), nn.AdaptiveAvgPool2d((2, 2)),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(width * 16 + 2, width * 8), nn.SiLU(), nn.Linear(width * 8, horizon * 3))

    def forward(self, image: torch.Tensor, state: torch.Tensor | None = None) -> torch.Tensor:
        if state is None or state.shape != (image.shape[0], 2):
            raise ValueError("Trajectory state must have shape [batch, 2]: speed_mps, yaw_rate_rps")
        features = self.encoder(image * 2.0 - 1.0).flatten(1)
        output = self.head(torch.cat((features, state), dim=1)).reshape(-1, self.horizon, 3)
        return torch.cat((output[..., :2], F.softplus(output[..., 2:3])), dim=-1)


def build_model(config: dict) -> nn.Module:
    task = config.get("task")
    options = config.get("model", {})
    width = options.get("width", 16)
    if isinstance(width, bool) or not isinstance(width, int) or width < 4:
        raise ValueError("model.width must be an integer >= 4")
    if task == "segmentation":
        classes = options.get("num_classes", 3)
        if isinstance(classes, bool) or not isinstance(classes, int) or classes < 2 or classes > 255:
            raise ValueError("model.num_classes must be an integer from 2 to 255 (255 is the ignore label)")
        return TinyUNet(classes, width)
    if task == "depth":
        return TinyUNet(1, width, positive=True)
    if task == "trajectory":
        horizon = options.get("horizon", 5)
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
            raise ValueError("model.horizon must be a positive integer")
        return TinyTrajectory(horizon, width)
    raise ValueError("task must be segmentation, depth, or trajectory")
