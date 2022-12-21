import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import auto_fp16, force_fp32

from mmdet3d.models.builder import PROJECTORS


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
        self.last = nn.Linear(channels[-1], channels[-1])

    @force_fp32(apply_to=('x',))
    def forward(self, x, modal=None):
        for layer in self.layers:
            x = layer(x)
        x = F.adaptive_max_pool2d(x, 1).squeeze()
        x = self.last(x)
        return x


@PROJECTORS.register_module()
class SharedProjector(nn.Module):
    def __init__(self,
                 lidar_channel,
                 camera_channel,
                 channels,
                 kernel_size,
                 stride,
                 ):
        super().__init__()
        self.first_lidar = nn.Conv2d(lidar_channel, channels[0], kernel_size=1, stride=stride, padding='valid')
        self.first_camera = nn.Conv2d(camera_channel, channels[0], kernel_size=1, stride=stride, padding='valid')
        self.layers = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.layers.append(
                nn.Sequential(
                    nn.Conv2d(channels[i], channels[i + 1], kernel_size=kernel_size, stride=stride, padding='valid'),
                    # nn.BatchNorm2d(channels[i + 1]),        
                    nn.ReLU()        
                    )
            )
        self.last_lidar = nn.Linear(channels[-1], channels[-1])
        self.last_camera = nn.Linear(channels[-1], channels[-1])

    @force_fp32(apply_to=('x',))
    def forward(self, x, modal):
        if modal == 'lidar':
            x = self.first_lidar(x)
            x = F.relu(x)
            for layer in self.layers:
                x = layer(x)
            x = F.adaptive_max_pool2d(x, 1).squeeze()
            x = self.last_lidar(x)
            return x
        elif modal == 'camera':
            x = self.first_camera(x)
            x = F.relu(x)
            for layer in self.layers:
                x = layer(x)
            x = F.adaptive_max_pool2d(x, 1).squeeze()
            x = self.last_camera(x)
            return x