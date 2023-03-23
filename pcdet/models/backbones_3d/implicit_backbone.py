import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from ..backbones_2d.unet_2d import MultiScaleEncoder2D, MultiScaleEncoder2D3Scales


def Conv2dBlock(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
    m = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, dilation=dilation, bias=False),
        nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
    )
    return m


def normalize_coordinates_2d(points, grid_size, voxel_size, point_cloud_range):
    """ All points are in point_cloud_range, the center of the corner grid is regarded as -1/1
    Args:
        points: (..., 2), XY
        point_cloud_range: (6,), XY
    Returns:
        points_nml: (..., 2), XY, in range [-1, 1]. Points outside centers of corners are out of [-1, 1]
    """
    pc_min = point_cloud_range[:2]
    points_nml = ((points - pc_min) / voxel_size - 0.5) / (grid_size - 1)
    points_nml = points_nml * 2 - 1
    return points_nml


class ImplicitNet2d(nn.Module):
    def __init__(self, model_cfg, input_channels, output_channels, grid_size, voxel_size, point_cloud_range, stride_2d=4, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.grid_size = torch.FloatTensor(grid_size).cuda()
        self.voxel_size = torch.FloatTensor(voxel_size).cuda()
        assert(torch.all(self.grid_size % stride_2d) == 0)
        self.grid_size = self.grid_size / stride_2d
        self.voxel_size = self.voxel_size * stride_2d
        self.point_cloud_range = torch.FloatTensor(point_cloud_range).cuda()

        if stride_2d == 4 or stride_2d == 1:
            self.encoder = MultiScaleEncoder2D(input_channels)
        elif stride_2d == 8:
            self.encoder = MultiScaleEncoder2D3Scales(input_channels)
        else:
            raise NotImplementedError

        ms_dim = np.sum(self.encoder.out_dims)
        self.decoder = nn.Sequential(
            nn.Linear(ms_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_channels),
        )

    def forward(self, x, points_sample):
        """
        Args:
            points_sample: (B, S, 2)
            x: (B, C, H, W)
        Returns:
            logits_sample: (B, S, output_channels)
        """
        assert(len(points_sample.shape) == 3)
        assert(x.shape[0] == points_sample.shape[0])

        features, strides = self.encoder(x)
        features_sample = []
        for i, stride in enumerate(strides):
            assert(torch.all(self.grid_size % stride) == 0)
            grid_size_s = self.grid_size / stride
            voxel_size_s = self.voxel_size * stride

            points_nml = normalize_coordinates_2d(points_sample, grid_size_s, voxel_size_s, self.point_cloud_range)[:, None, :, :] # (B, 1, S, 2), XY

            feature_sample = F.grid_sample(features[i], points_nml, padding_mode='border', align_corners=True) # (B, C, 1, S)
            features_sample.append(feature_sample[:, :, 0, :]) # (B, C, S)
        features_sample = torch.cat(features_sample, dim=1) # (B, C1+...+C4, S)

        features_sample = rearrange(features_sample, 'b d n -> b n d') # (B, S, C1+...+C4)

        logits_sample = self.decoder(features_sample)

        return logits_sample




