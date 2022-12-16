import torch
import torch.nn as nn

from mmdet3d.models.builder import PROJECTORS


@PROJECTORS.register_module()
class MLPProjector(nn.Module):
    def __init__(self,
                 channels,
                 ):
        super(MLPProjector, self).__init__()
        self.mlp = []

        for i in range(len(channels) - 1):
            self.mlp.append(
                nn.Sequential(
                    nn.Linear(channels[i], channels[i + 1]),
                    nn.BatchNorm1d(channels[i + 1]),
                    nn.ReLU()
                )
            )

    def forward(self, x):
        for layer in self.mlp:
            x = layer(x)
        return x