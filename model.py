import torch
import numpy as np
from torch import nn
from torch.nn import functional as F


class Net(nn.Module):
    def __init__(self, in_channels: int, n_classes: int) -> None:
        super().__init__()
        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, 1, 1),
            nn.Conv2d(16, 32, 3, 1, 1),
        )
        # downsample to 1/2 and upscale
        self.encode2 = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(32, 32, 3, 1, 1),
        )
        # downsample to 1/4 and upscale
        self.encode3 = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(32, 32, 3, 1, 1),
        )
        # aux (N,C,H,W), C=point_count
        self.aux = nn.Sequential(
            nn.Conv2d(32 * 3, n_classes, 3, 1, 1),
        )

    def forward(self, x: torch.Tensor):
        input_size = x.shape[2:4]
        ncol = x.shape[3]
        enc1 = self.conv0(x)
        enc2 = self.encode2(enc1)
        enc3 = self.encode3(enc2)
        enc2 = F.interpolate(enc2, size=input_size)
        enc3 = F.interpolate(enc3, size=input_size)
        aux = self.aux(torch.cat([enc1, enc2, enc3], 1))
        aux = torch.flatten(aux, 2, 3)  # (N,C,HxW)
        aux = torch.max(aux, dim=-1).values  # (N,C), int
        # points = []
        # for i in aux:
        #     points.append([[torch.div(p, ncol), p % ncol, 1] for p in i])
        # returns (N,20,3), 20 flatten points, assumes it exists
        # return torch.tensor(points, dtype=torch.float, requires_grad=True)
        return aux.reshape(aux.shape[0], 20, 2)
