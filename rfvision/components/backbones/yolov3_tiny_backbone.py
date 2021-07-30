'''
This impementation based on https://github.com/hhaAndroid/mmdetection-mini
'''


from collections import OrderedDict
import torch.nn as nn
from rflib.runner import BaseModule
from rflib.cnn import ConvModule
from rfvision.models.builder import BACKBONES


@BACKBONES.register_module()
class YOLOV3TinyBackbone(BaseModule):
    def __init__(self,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
                 init_cfg=None,
                 ):
        super(YOLOV3TinyBackbone, self).__init__(init_cfg)
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

    def forward(self, x):
        stem = self.layers[0](x)
        stage4 = self.layers[1](stem)
        stage5 = self.layers[2](stage4)
        stage6 = self.layers[3](stage5)
        features = [stage6, stage5]
        return features

