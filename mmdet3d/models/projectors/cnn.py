import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import auto_fp16, force_fp32

from mmdet3d.models.builder import PROJECTORS

__all__ = ['CNNProjector']

@PROJECTORS.register_module()
class CNNProjector(nn.Module):
    def __init__(self,
                 channels,
                 kernel_size,
                 stride,
                 ):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.layers.append(
                nn.Sequential(
                    nn.Conv2d(channels[i], channels[i + 1], kernel_size=kernel_size, stride=stride, padding='valid'),
                    nn.BatchNorm2d(channels[i + 1]),
                    nn.ReLU()
                )
            )

    # def init_weights(self):
    #     pass

    @force_fp32(apply_to=('x',))
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = F.adaptive_avg_pool2d(x, 1).squeeze(-1)
        return x