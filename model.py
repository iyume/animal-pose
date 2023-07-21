import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class Net(nn.Module):
    version = "v0.1"

    def __init__(self, in_channels: int, n_classes: int) -> None:
        super().__init__()
        # input 400x400
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 11, 4, 5),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
        )
        # 100x100
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 48, 3, 2, 1),
            nn.ReLU(),
        )
        # 50x50
        self.conv3 = nn.Sequential(
            nn.Conv2d(48, 64, 3, 2, 1),
            nn.ReLU(),
        )
        # 25x25
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 2, 1),
            nn.ReLU(),
        )
        # 12x12
        self.fc1 = nn.Linear(13 * 13 * 64, 4000)
        self.fc2 = nn.Linear(4000, 4000)
        self.out = nn.Linear(4000, n_classes)
        # downsample to 1/2 and upscale
        # 250x250
        # self.encode2 = nn.Sequential(
        #     nn.AvgPool2d(2),
        #     nn.Conv2d(32, 32, 3, 1, 1),
        # )
        # downsample to 1/4 and upscale
        # 125x125
        # self.encode3 = nn.Sequential(
        #     nn.AvgPool2d(2),
        #     nn.Conv2d(32, 32, 3, 1, 1),
        # )
        # aux (N,C,H,W), C=point_count
        # self.aux = nn.Sequential(
        #     nn.Conv2d(32 * 3, n_classes, 3, 1, 1),
        # )

        # self.encode_shrink = nn.Sequential(
        #     nn.AvgPool2d(2),
        #     nn.Conv2d(32, 32, 3, 1, 1),
        #     nn.AvgPool2d(2),
        #     nn.Conv2d(32, 32, 3, 1, 1),
        # )
        # self.linear = nn.Linear(32 * 31 * 31, 64 * 64)
        # self.linear_out = nn.Linear(64 * 64, n_classes)

    def forward(self, x: torch.Tensor):
        # input_size = x.shape[2:4]
        # ncol = x.shape[3]
        # enc1 = self.conv0(x)
        # enc2 = self.encode2(enc1)
        # enc3 = self.encode3(enc2)

        # enc2 = F.interpolate(enc2, size=input_size)
        # enc3 = F.interpolate(enc3, size=input_size)
        # aux = self.aux(torch.cat([enc1, enc2, enc3], 1))

        # aux = torch.flatten(aux, 2, 3)  # (N,C,HxW)
        # aux = torch.max(aux, dim=-1).values  # (N,C), int
        # points = []
        # for i in aux:
        #     points.append([[torch.div(p, ncol), p % ncol, 1] for p in i])
        # returns (N,20,3), 20 flatten points, assumes it exists
        # return torch.tensor(points, dtype=torch.float, requires_grad=True)

        # enc = self.encode_shrink(enc3)
        # out = self.linear(enc.flatten(1, -1))
        # out = self.linear_out(out)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.fc1(x.flatten(1))
        x = self.fc2(x)
        out = self.out(x)
        return out
