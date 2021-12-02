import torch
import torch.nn as nn
from rfvision.models.builder import BACKBONES
from rflib.ops import PointFPModule, PointSAModule
from rflib.runner import auto_fp16
from rflib.runner import BaseModule


# @BACKBONES.register_module()
class PointNet2ForArticulation(BaseModule):
    def __init__(self, in_channels=3, init_cfg=None):
        super().__init__(init_cfg)
        self.sa1 = PointSAModule(num_point=512, radius=0.2, num_sample=64, mlp_channels=[in_channels, 64, 64, 128])
        self.sa2 = PointSAModule(num_point=128, radius=0.4, num_sample=64, mlp_channels=[128, 128, 128, 256],)
        self.sa3 = PointSAModule(num_point=None, radius=None, num_sample=None, mlp_channels=[256, 256, 512, 1024])
        self.fp3 = PointFPModule([1280, 256, 256])
        self.fp2 = PointFPModule([384, 256, 128])
        self.fp1 = PointFPModule([134, 128, 128, 128])

        self.fc = nn.Sequential(
            nn.Conv1d(128, 128, 1),
            nn.BatchNorm1d(128),
            nn.Dropout(p=0.5))

    @auto_fp16(apply_to=('points',))
    def forward(self, l0_xyz, l0_features):
        if l0_features is not None:
            points = torch.cat((l0_xyz, l0_features), 2)
        else:
            points = l0_xyz

        l0_xyz, l0_features = self._split_point_feats(points)

        # Set Abstraction layers
        l1_xyz, l1_features, _ = self.sa1(l0_xyz, l0_features)
        l2_xyz, l2_features, _ = self.sa2(l1_xyz, l1_features)
        l3_xyz, l3_features, _ = self.sa3(l2_xyz, l2_features)

        # Feature Propagation layers
        l2_features = self.fp3(l2_xyz, l3_xyz, l2_features, l3_features)
        l1_features = self.fp2(l1_xyz, l2_xyz, l1_features, l2_features)

        l0_features = self.fp1(l0_xyz, l1_xyz, torch.cat([l0_xyz, l0_features.transpose(1, 2)], 2).transpose(1,2), l1_features)

        out = self.fc(l0_features)

        return out, l3_features

    @staticmethod
    def _split_point_feats(points):
        xyz = points[..., 0:3].contiguous()
        if points.size(-1) > 3:
            features = points[..., 3:].transpose(1, 2).contiguous()
        else:
            features = None

        return xyz, features

if __name__ == '__main__':
    m = PointNet2ForArticulation(6).cuda()
    points = torch.rand(64, 1024, 3).cuda()
    features = torch.rand(64, 1024, 3).cuda()
    out, l3_features = m(points, features)
