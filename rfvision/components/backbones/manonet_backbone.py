import torch.nn as nn
import torch
from rfvision.components.backbones import ResNet
from rfvision.models.builder import BACKBONES
from rfvision.components.utils.handtailor_utils import uvd2xyz


@BACKBONES.register_module()
class ManoNetBackbone(nn.Module):
    def __init__(self,
                 in_channels=512,
                 hidden_channels=(512, 512, 1024, 1024, 512, 256),
                 out_channels=12,
                 ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels

        self.channels = [in_channels] + list(hidden_channels)

        layers = []
        for i, j in zip(self.channels[:-1], self.channels[1:]):
            layers.append(nn.Linear(i, j))
            layers.append(nn.BatchNorm1d(j))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(self.channels[-1], self.out_channels))
        self.layers = nn.Sequential(*layers)

        self.decoder = ResNet(depth=18, in_channels=256)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.sigmoid = nn.Sigmoid()

        self.ref_bone_link = (0, 9)
        self.joint_root_idx = 9

    def forward(self, pred_dict_3d, K):
        # release parameters
        out_features_3d = pred_dict_3d['out_features_3d']
        pred_joints_uvd = pred_dict_3d['pred_joints_uvd']

        # convolution
        out_features_decoder = self.decoder(out_features_3d)[3]
        out_features_decoder = self.avgpool(out_features_decoder)
        out_features_decoder = torch.flatten(out_features_decoder, 1)
        shape_vector = self.layers(out_features_decoder)

        # post processing
        joint_bone = self.sigmoid(shape_vector[:, 0:1])
        root = self.sigmoid(shape_vector[:, 1:2])
        beta = shape_vector[:, 2:]

        joints_xyz = uvd2xyz(pred_joints_uvd,
                             root,
                             joint_bone,
                             K)
        joints_root = joints_xyz[:, self.joint_root_idx, :].unsqueeze(1)
        joints_xyz = joints_xyz - joints_root
        joint_bone = torch.norm(joints_xyz[:, self.ref_bone_link[1], :] - joints_xyz[:, self.ref_bone_link[0], :],
                                dim=1, keepdim=True)

        joint_bone = joint_bone.unsqueeze(1)  # (B,1,1)
        joints_xyz = joints_xyz / joint_bone

        pred_dict_mano = {'joints_xyz_mano': joints_xyz,
                          'beta': beta,
                          }
        return pred_dict_mano
