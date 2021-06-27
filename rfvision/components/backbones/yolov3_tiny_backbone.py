'''
This impementation based on https://github.com/hhaAndroid/mmdetection-mini
'''


from collections import OrderedDict
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm

from rflib.runner import load_checkpoint
from rflib.cnn import ConvModule, kaiming_init, constant_init
from rfvision.models.builder import BACKBONES
import logging


@BACKBONES.register_module()
class YOLOV3TinyBackbone(nn.Module):
    def __init__(self,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='LeakyReLU', negative_slope=0.1)
                 ):
        super(YOLOV3TinyBackbone, self).__init__()
        cfg = dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

        # Network
        layer_list = [
            OrderedDict([
                ('0_convbatch', ConvModule(3, 16, 3, 1, 1, **cfg)),
                ('1_max', nn.MaxPool2d(2, 2)),
                ('2_convbatch', ConvModule(16, 32, 3, 1, 1, **cfg)),
                ('3_max', nn.MaxPool2d(2, 2)),
                ('4_convbatch', ConvModule(32, 64, 3, 1, 1, **cfg)),
            ]),
            OrderedDict([
                ('5_max', nn.MaxPool2d(2, 2)),
                ('6_convbatch', ConvModule(64, 128, 3, 1, 1, **cfg)),
            ]),
            OrderedDict([
                ('7_max', nn.MaxPool2d(2, 2)),
                ('8_convbatch', ConvModule(128, 256, 3, 1, 1, **cfg)),
            ]),

            OrderedDict([
                ('9_max', nn.MaxPool2d(2, 2)),
                ('10_convbatch', ConvModule(256, 512, 3, 1, 1, **cfg)),
                ('10_zero_pad', nn.ZeroPad2d((0, 1, 0, 1))),
                ('11_max', nn.MaxPool2d(2, 1)),
                ('12_convbatch', ConvModule(512, 1024, 3, 1, 1, **cfg)),
                ('13_convbatch', ConvModule(1024, 256, 1, 1, **cfg)),
            ]),
        ]

        self.layers = nn.ModuleList([nn.Sequential(layer_dict) for layer_dict in layer_list])


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
        stem = self.layers[0](x)
        stage4 = self.layers[1](stem)
        stage5 = self.layers[2](stage4)
        stage6 = self.layers[3](stage5)
        features = [stage6, stage5]
        return features

