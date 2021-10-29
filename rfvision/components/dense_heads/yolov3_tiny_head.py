'''
This impementation based on https://github.com/hhaAndroid/mmdetection-mini
'''
from collections import OrderedDict
import torch.nn as nn
import torch
from torch.nn.modules.batchnorm import _BatchNorm
from .yolo_head import YOLOV3Head
from rfvision.models.builder import HEADS
from rflib.cnn import ConvModule, kaiming_init, constant_init

@HEADS.register_module()
class YOLOV3TinyHead(YOLOV3Head):
    def _init_layers(self):
        layer_list = [
            # stage 6
            OrderedDict([
                ('14_convbatch', ConvModule(self.in_channels[0], self.out_channels[0], 3, 1, 1, **self.cfg)),
                ('15_conv', nn.Conv2d(self.out_channels[0], self.num_anchors * self.num_attrib, 1, 1, 0)),
            ]),
            # stage 5
            # stage5 / upsample
            OrderedDict([
                ('18_convbatch', ConvModule(self.in_channels[1], self.out_channels[1], 1, 1, **self.cfg)),
                ('19_upsample', nn.Upsample(scale_factor=2)),
            ]),
            # stage5 / head
            OrderedDict([
                ('21_convbatch', ConvModule(self.in_channels[1] + self.out_channels[1], self.out_channels[1] * 2, 3, 1, 1, **self.cfg)),
                ('22_conv', nn.Conv2d(self.out_channels[1]*2, self.num_anchors * self.num_attrib, 1, 1, 0)),
            ]),
        ]
        self.layers = nn.ModuleList([nn.Sequential(layer_dict) for layer_dict in layer_list])

    def forward(self, feats):
        stage6 = self.layers[0](feats[0])
        stage5_upsample = self.layers[1](feats[0])
        stage5 = self.layers[2](torch.cat((stage5_upsample, feats[1]), 1))
        return (stage6, stage5),

    def init_weights(self, pretrained=None):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                constant_init(m, 1)
