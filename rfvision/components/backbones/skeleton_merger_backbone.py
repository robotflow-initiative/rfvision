from rfvision.components.backbones import BasePointNet
from rflib.ops import PointFPModule, PointSAModuleMSG
from rflib.runner import auto_fp16
from rfvision.models.builder import BACKBONES
import torch.nn as nn
import torch 
import torch.nn.functional as F

# @BACKBONES.register_module()
class PointNet2ForSkeletonMerger(BasePointNet):
    def __init__(self, num_classes):
        super().__init__()
        self.sa1 = PointSAModuleMSG(1024, [0.05, 0.1], [16, 32], [[6, 16, 16, 32], [6, 32, 32, 64]])
        self.sa2 = PointSAModuleMSG(256, [0.1, 0.2], [16, 32], [[32+64, 64, 64, 128], [32+64, 64, 96, 128]])
        self.sa3 = PointSAModuleMSG(64, [0.2, 0.4], [16, 32], [[128+128, 128, 196, 256], [128+128, 128, 196, 256]])
        self.sa4 = PointSAModuleMSG(16, [0.4, 0.8], [16, 32], [[256+256, 256, 256, 512], [256+256, 256, 384, 512]])

        self.fp4 = PointFPModule([512+512+256+256, 256, 256])
        self.fp3 = PointFPModule([128+128+256, 256, 256])
        self.fp2 = PointFPModule([32+64+256, 256, 128])
        self.fp1 = PointFPModule([128, 128, 128, 128])

        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    @auto_fp16(apply_to=('points',))
    def forward(self, points):
        l0_xyz, l0_features = self._split_point_feats(points)
        l1_xyz, l1_features, _ = self.sa1(l0_xyz, l0_features)
        l2_xyz, l2_features, _ = self.sa2(l1_xyz, l1_features)
        l3_xyz, l3_features, _ = self.sa3(l2_xyz, l2_features)
        l4_xyz, l4_features, _ = self.sa4(l3_xyz, l3_features)

        l3_features = self.fp4(l3_xyz, l4_xyz, l3_features, l4_features)
        l2_features = self.fp3(l2_xyz, l3_xyz, l2_features, l3_features)
        l1_features = self.fp2(l1_xyz, l2_xyz, l1_features, l2_features)
        l0_features = self.fp1(l0_xyz, l1_xyz, None       , l1_features)

        x = self.drop1(F.relu(self.bn1(self.conv1(l0_features))))
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)

        return x, l4_features

if __name__ == '__main__':
    from rfvision.utils import count_paras
    m = PointNet2ForSkeletonMerger(10).cuda()
    xyz = torch.rand(6, 2048, 9).cuda() # (batch_size, num_points, features(x,y,z + other_features))
    x, l4_features = m(xyz)
    
    print('total paras:', count_paras(m))