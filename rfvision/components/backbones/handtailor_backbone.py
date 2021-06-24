from robotflow.rflearner.bricks.backbones.resnet import Bottleneck
from robotflow.rflib.cnn import ConvModule
from robotflow.rflearner.builder import BACKBONES
import torch.nn as nn
import torch
import torch.nn.functional as F



def _Bottleneck(inplanes, planes, expansion=2):
    downsample = None
    if inplanes != planes * expansion:
        downsample = nn.Conv2d(inplanes, planes * expansion, kernel_size=1)
    return Bottleneck(inplanes, planes, expansion=expansion, downsample=downsample)


def _make_residual(inplanes, planes, num_block):
    layers = []
    for _ in range(0, num_block):
        layers.append(_Bottleneck(inplanes, planes, expansion=2))
    return nn.Sequential(*layers)


class _HourglassModule(nn.Module):
    def __init__(self, num_blocks=1, inplanes=64, depth=4):
        super().__init__()
        self.depth = depth
        self.hg = self._make_hourglass(num_blocks, inplanes, depth)

    def _make_hourglass(self, num_blocks, inplanes, depth):
        hg = []
        for i in range(depth):
            res = []
            for _ in range(3):
                res.append(_make_residual(inplanes, inplanes // 2, num_blocks))
            if i == 0:
                res.append(_make_residual(inplanes, inplanes // 2, num_blocks))
            hg.append(nn.ModuleList(res))
        return nn.ModuleList(hg)

    def _hourglass_foward(self, n, x):
        up1 = self.hg[n - 1][0](x)
        low1 = F.max_pool2d(x, 2, stride=2)
        low1 = self.hg[n - 1][1](low1)
        if n > 1:
            low2 = self._hourglass_foward(n - 1, low1)
        else:
            low2 = self.hg[n - 1][3](low1)
        low3 = self.hg[n - 1][2](low2)
        up2 = F.interpolate(low3, scale_factor=2)
        out = up1 + up2
        return out

    def forward(self, x):
        return self._hourglass_foward(self.depth, x)

@BACKBONES.register_module()
class HandTailor2DBackbone(nn.Module):
    def __init__(self,
                 num_joints=21,
                 num_stack=2):
        super().__init__()
        self.num_stack = num_stack
        self.num_joints = num_joints

        self.res_layers()
        self.fuse_layers()

    def fuse_layers(self):
        self.sigmoid = nn.Sigmoid()
        hg, res, fc, hm = [], [], [], []
        for _ in range(self.num_stack):
            hg.append(_HourglassModule(depth=4, inplanes=256))
            res.append(_make_residual(256, 128, 1))
            hm.append(nn.Conv2d(256, self.num_joints, kernel_size=1, bias=True))
            fc.append(ConvModule(256 + self.num_joints, 256, kernel_size=1, norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU')))
        self.hg = nn.ModuleList(hg)
        self.res = nn.ModuleList(res)
        self.fc = nn.ModuleList(fc)
        self.hm = nn.ModuleList(hm)

    def res_layers(self, ):
        self.conv1 = ConvModule(3, 64, kernel_size=7, stride=2, padding=3,
                                norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU'))
        self.res1 = _make_residual(inplanes=64, planes=64, num_block=1)
        self.res2 = _make_residual(inplanes=128, planes=128, num_block=1)
        self.res3 = _make_residual(inplanes=256, planes=128, num_block=1)
        self.maxpool = nn.MaxPool2d(2, stride=2)

    def forward_res_layers(self, x):
        out = self.conv1(x)       # shape (bz, 64, 128, 128)
        out = self.res1(out)      # shape (bz, 128, 128, 128)
        out = self.maxpool(out)   # shape (bz, 128, 64, 64)
        out = self.res2(out)      # shape (bz, 256, 64, 64)
        out = self.res3(out)      # shape (bz, 256, 64, 64)
        return out

    def forward_fuse_layers(self, out):
        self.sigmoid = nn.Sigmoid()
        heatmap_list, out_list = [], []
        for i in range(self.num_stack):
            out = self.hg[i](out)
            out = self.res[i](out)
            heatmap = self.sigmoid(self.hm[i](out))
            out = torch.cat((out, heatmap), 1)
            out = self.fc[i](out)
            heatmap_list.append(heatmap)
            out_list.append(out)
        return heatmap_list, out_list

    def forward(self, x):
        # x.shape : (bz, 3, 256, 256)
        out = self.forward_res_layers(x)
        heatmap_list, out_list = self.forward_fuse_layers(out)
        return heatmap_list, out_list


@BACKBONES.register_module()
class HandTailor3DBackbone(nn.Module):
    def __init__(self,
                 num_joints=21,
                 num_stack=2):
        super().__init__()
        self.num_stack = num_stack
        self.num_joints = num_joints
        self.fuse_layers()
    def fuse_layers(self):
        hg, res, fc, hm = [], [], [], []
        for _ in range(self.num_stack):
            hg.append(_HourglassModule(depth=4, inplanes=256))
            res.append(_make_residual(256, 128, 1))
            hm.append(nn.Conv2d(256, 2 * self.num_joints, kernel_size=1, bias=True))
            fc.append(ConvModule(256 + 2 * self.num_joints, 256, kernel_size=1, norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU')))
        self.hg = nn.ModuleList(hg)
        self.res = nn.ModuleList(res)
        self.fc = nn.ModuleList(fc)
        self.hm = nn.ModuleList(hm)

    def forward_fuse_layers(self, out):
        # out shape (bz, 256, 64, 64)
        self.sigmoid = nn.Sigmoid()
        heatmap_list, out_list = [], []
        for i in range(self.num_stack):
            out = self.hg[i](out)
            out = self.res[i](out)
            heatmap = self.sigmoid(self.hm[i](out))
            out = torch.cat((out, heatmap), 1)
            out = self.fc[i](out)
            heatmap_list.append(heatmap)
            out_list.append(out)
        return heatmap_list, out_list

    def forward(self, out):
        heatmap_list, out_list = self.forward_fuse_layers(out)
        return heatmap_list, out_list


if __name__ == '__main__':
    img = torch.rand(2, 3, 256, 256)
    m_2d = HandTailor2DBackbone()
    m_3d = HandTailor3DBackbone()

    # out_list_2d shape (bz, 256, 64, 64)
    # heatmap_list_3d shape (bz, 21, 64, 64)
    heatmap_list_2d, out_list_2d = m_2d(img)
    out = out_list_2d[-1]

    # out_list_3d shape (bz, 256, 64, 64)
    # heatmap_list_3d shape (bz, 42, 64, 64)
    heatmap_list_3d, out_list_3d = m_3d(out)



