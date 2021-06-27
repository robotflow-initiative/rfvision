'''
This impementation based on https://github.com/hhaAndroid/mmdetection-mini
'''

from collections import OrderedDict
import torch.nn as nn
import torch
import logging
from torch.nn.modules.batchnorm import _BatchNorm
from rfvision.models.builder import BACKBONES
from rflib.cnn import ConvModule, kaiming_init, constant_init
from rflib.runner import load_checkpoint


class ResConv2dBatchLeaky(nn.Module):
    def __init__(self,
                 in_channels,
                 inter_channels,
                 kernel_size,
                 stride=1,
                 return_extra=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='LeakyReLU', negative_slope=0.1)):
        super(ResConv2dBatchLeaky, self).__init__()
        cfg = dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

        self.return_extra = return_extra
        self.in_channels = in_channels
        self.inter_channels = inter_channels
        self.kernel_size = kernel_size
        self.stride = stride
        if isinstance(kernel_size, (list, tuple)):
            self.padding = [int(ii / 2) for ii in kernel_size]
        else:
            self.padding = int(kernel_size / 2)

        self.layers0 = ConvModule(self.in_channels // 2, self.inter_channels, self.kernel_size, self.stride,
                                  self.padding, **cfg)
        self.layers1 = ConvModule(self.inter_channels, self.inter_channels, self.kernel_size, self.stride,
                                  self.padding, **cfg)
        self.layers2 = ConvModule(self.in_channels, self.in_channels, 1, 1, **cfg)

    def forward(self, x):
        y0 = x
        channel = x.shape[1]
        x0 = x[:, channel // 2:, ...]
        x1 = self.layers0(x0)
        x2 = self.layers1(x1)
        x3 = torch.cat((x2, x1), dim=1)
        x4 = self.layers2(x3)
        x = torch.cat((y0, x4), dim=1)
        if self.return_extra:
            return x, x4
        else:
            return x


@BACKBONES.register_module()
class YOLOV4TinyBackbone(nn.Module):
    def __init__(self,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
                 init_cfg=None,
                 ):
        super(YOLOV4TinyBackbone, self).__init__()
        cfg = dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        backbone = OrderedDict([
            ('0_convbatch', ConvModule(3, 32, 3, 2, 1, **cfg)),
            ('1_convbatch', ConvModule(32, 64, 3, 2, 1, **cfg)),
            ('2_convbatch', ConvModule(64, 64, 3, 1, 1, **cfg)),
            ('3_resconvbatch', ResConv2dBatchLeaky(64, 32, 3, 1, **cfg)),
            ('4_max', nn.MaxPool2d(2, 2)),
            ('5_convbatch', ConvModule(128, 128, 3, 1, 1, **cfg)),
            ('6_resconvbatch', ResConv2dBatchLeaky(128, 64, 3, 1, **cfg)),
            ('7_max', nn.MaxPool2d(2, 2)),
            ('8_convbatch', ConvModule(256, 256, 3, 1, 1, **cfg)),
            ('9_resconvbatch', ResConv2dBatchLeaky(256, 128, 3, 1, return_extra=True, **cfg)),
        ])

        self.layers = nn.Sequential(backbone)

    def init_weights(self, init_cfg=None):
        if isinstance(init_cfg, str):
            logger = logging.getLogger()
            load_checkpoint(self, init_cfg, strict=False, logger=logger)
        elif init_cfg is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)

    def forward(self, x):
        stem, extra_x = self.layers(x)
        return (stem, extra_x)
