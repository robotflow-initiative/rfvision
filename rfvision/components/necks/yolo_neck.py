# Copyright (c) 2019 Western Digital Corporation or its affiliates.

import torch
import torch.nn as nn
import torch.nn.functional as F
from rflib.cnn import ConvModule
from rflib.runner import BaseModule

from rfvision.models.builder import NECKS

class SpatialPyramidPooling(nn.Module):
    def __init__(self, pool_sizes=[5, 9, 13]):
        super(SpatialPyramidPooling, self).__init__()

        self.maxpools = nn.ModuleList([nn.MaxPool2d(pool_size, 1, pool_size // 2) for pool_size in pool_sizes])

    def forward(self, x):
        features = [maxpool(x) for maxpool in self.maxpools[::-1]]
        features = torch.cat(features + [x], dim=1)
        return features

class FuseStage(nn.Module):
    def __init__(self, 
                 in_channels,
                 is_reversal = False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='Mish')):
        super(FuseStage, self).__init__()
        cfg = dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

        if is_reversal:
            self.right_conv = None
            self.left_conv = ConvModule(in_channels, in_channels * 2, kernel_size=3, stride=2, padding = 1, **cfg)
        else:
            self.right_conv = nn.Sequential(ConvModule(in_channels, in_channels // 2, kernel_size = 1, stride = 1,**cfg),
                                            nn.Upsample(scale_factor=2, mode='nearest'))
            self.left_conv = ConvModule(in_channels, in_channels // 2, kernel_size = 1, stride = 1,**cfg)
    def forward(self, data):
        left, right = data
        left = self.left_conv(left)
        if self.right_conv:
            right = self.right_conv(right)
        return torch.cat((left, right), axis = 1)

class MakeNConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 n,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='Mish')):
        super().__init__()
        cfg = dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        double_out_channels = out_channels * 2
        if n == 3:
            self.m = nn.Sequential(
                ConvModule(in_channels, out_channels, kernel_size = 1, padding = 0,**cfg),
                ConvModule(out_channels,double_out_channels, kernel_size = 3, padding = 1,**cfg),
                ConvModule(double_out_channels, out_channels, kernel_size = 1, padding = 0,**cfg),
            )
        elif n == 5:
            self.m = nn.Sequential(
                ConvModule(in_channels, out_channels, kernel_size = 1, padding = 0,**cfg),
                ConvModule(out_channels, double_out_channels, kernel_size = 3, padding = 1,**cfg),
                ConvModule(double_out_channels, out_channels, kernel_size = 1, padding = 0,**cfg),
                ConvModule(out_channels, double_out_channels, kernel_size = 3, padding = 1,**cfg),
                ConvModule(double_out_channels, out_channels, kernel_size = 1, padding = 0,**cfg),
            )
        else:
            raise NotImplementedError

    def forward(self, x):
        return self.m(x)


class DetectionBlock(BaseModule):
    """Detection block in YOLO neck.

    Let out_channels = n, the DetectionBlock contains:
    Six ConvLayers, 1 Conv2D Layer and 1 YoloLayer.
    The first 6 ConvLayers are formed the following way:
        1x1xn, 3x3x2n, 1x1xn, 3x3x2n, 1x1xn, 3x3x2n.
    The Conv2D layer is 1x1x255.
    Some block will have branch after the fifth ConvLayer.
    The input channel is arbitrary (in_channels)

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Default: dict(type='BN', requires_grad=True)
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='LeakyReLU', negative_slope=0.1).
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
                 init_cfg=None):
        super(DetectionBlock, self).__init__(init_cfg)
        double_out_channels = out_channels * 2

        # shortcut
        cfg = dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv1 = ConvModule(in_channels, out_channels, 1, **cfg)
        self.conv2 = ConvModule(
            out_channels, double_out_channels, 3, padding=1, **cfg)
        self.conv3 = ConvModule(double_out_channels, out_channels, 1, **cfg)
        self.conv4 = ConvModule(
            out_channels, double_out_channels, 3, padding=1, **cfg)
        self.conv5 = ConvModule(double_out_channels, out_channels, 1, **cfg)

    def forward(self, x):
        tmp = self.conv1(x)
        tmp = self.conv2(tmp)
        tmp = self.conv3(tmp)
        tmp = self.conv4(tmp)
        out = self.conv5(tmp)
        return out


@NECKS.register_module()
class YOLOV3Neck(BaseModule):
    """The neck of YOLOV3.

    It can be treated as a simplified version of FPN. It
    will take the result from Darknet backbone and do some upsampling and
    concatenation. It will finally output the detection result.

    Note:
        The input feats should be from top to bottom.
            i.e., from high-lvl to low-lvl
        But YOLOV3Neck will process them in reversed order.
            i.e., from bottom (high-lvl) to top (low-lvl)

    Args:
        num_scales (int): The number of scales / stages.
        in_channels (List[int]): The number of input channels per scale.
        out_channels (List[int]): The number of output channels  per scale.
        conv_cfg (dict, optional): Config dict for convolution layer.
            Default: None.
        norm_cfg (dict, optional): Dictionary to construct and config norm
            layer. Default: dict(type='BN', requires_grad=True)
        act_cfg (dict, optional): Config dict for activation layer.
            Default: dict(type='LeakyReLU', negative_slope=0.1).
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 num_scales,
                 in_channels,
                 out_channels,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
                 init_cfg=None):
        super(YOLOV3Neck, self).__init__(init_cfg)
        assert (num_scales == len(in_channels) == len(out_channels))
        self.num_scales = num_scales
        self.in_channels = in_channels
        self.out_channels = out_channels

        # shortcut
        cfg = dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

        # To support arbitrary scales, the code looks awful, but it works.
        # Better solution is welcomed.
        self.detect1 = DetectionBlock(in_channels[0], out_channels[0], **cfg)
        for i in range(1, self.num_scales):
            in_c, out_c = self.in_channels[i], self.out_channels[i]
            inter_c = out_channels[i - 1]
            self.add_module(f'conv{i}', ConvModule(inter_c, out_c, 1, **cfg))
            # in_c + out_c : High-lvl feats will be cat with low-lvl feats
            self.add_module(f'detect{i+1}',
                            DetectionBlock(in_c + out_c, out_c, **cfg))

    def forward(self, feats):
        assert len(feats) == self.num_scales

        # processed from bottom (high-lvl) to top (low-lvl)
        outs = []
        out = self.detect1(feats[-1])
        outs.append(out)

        for i, x in enumerate(reversed(feats[:-1])):
            conv = getattr(self, f'conv{i+1}')
            tmp = conv(out)

            # Cat with low-lvl feats
            tmp = F.interpolate(tmp, scale_factor=2)
            tmp = torch.cat((tmp, x), 1)

            detect = getattr(self, f'detect{i+2}')
            out = detect(tmp)
            outs.append(out)

        return tuple(outs)


@NECKS.register_module()
class YOLOV4Neck(BaseModule):
    def __init__(self,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='Mish'),
                 init_cfg=None):
        super(YOLOV4Neck, self).__init__(init_cfg)
        cfg = dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        
        self.layers = nn.ModuleList([
            # SPP
            nn.Sequential(MakeNConv(1024, 512, 3,**cfg,),
                          SpatialPyramidPooling(),
                          MakeNConv(2048, 512, 3,**cfg)),
            # PANet
            nn.Sequential(FuseStage(512, **cfg),
                          MakeNConv(512, 256, 5,**cfg)),
             
            nn.Sequential(FuseStage(256, **cfg),
                          MakeNConv(256, 128, 5,**cfg)),
             
            nn.Sequential(FuseStage(128, **cfg, is_reversal=True),
                          MakeNConv(512,256, 5, **cfg)),
             
            nn.Sequential(FuseStage(256, **cfg, is_reversal=True),
                          MakeNConv(1024,512, 5,**cfg))
            ])
        
    def forward(self, x):
        out3, out4, out5 = x
        out5 = self.layers[0](out5)
        out4 = self.layers[1]([out4, out5])

        out3 = self.layers[2]([out3, out4])
        out4 = self.layers[3]([out3, out4])
        out5 = self.layers[4]([out4, out5])

        return (out5, out4, out3)
                
                
                