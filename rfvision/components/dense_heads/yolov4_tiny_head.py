'''
This impementation based on https://github.com/hhaAndroid/mmdetection-mini
'''
from collections import OrderedDict
import torch.nn as nn
import torch
from .yolo_head import YOLOV3Head
from rfvision.models.builder import HEADS
from rflib.cnn import ConvModule, kaiming_init, constant_init
from torch.nn.modules.batchnorm import _BatchNorm



@HEADS.register_module()
class YOLOV4TinyHead(YOLOV3Head):
    def _init_layers(self):
        head = [
            OrderedDict([
                ('10_max', nn.MaxPool2d(2, 2)),
                ('11_conv', ConvModule(self.in_channels[0], self.in_channels[0], 3, 1, 1, **self.cfg)),
                ('12_conv', ConvModule(self.in_channels[0], self.out_channels[0], 1, 1 , **self.cfg)),
            ]),

            OrderedDict([
                ('13_conv', ConvModule(self.in_channels[1], self.in_channels[0], 3, 1, 1, **self.cfg)),
                ('14_conv', nn.Conv2d(self.in_channels[0], self.num_anchors * self.num_attrib, 1)),
            ]),

            OrderedDict([
                ('15_convbatch', ConvModule(self.in_channels[1], self.out_channels[1], 1, 1, **self.cfg)),
                ('16_upsample', nn.Upsample(scale_factor=2)),
            ]),

            OrderedDict([
                ('17_convbatch', ConvModule(self.out_channels[0]+self.out_channels[1], 256, 3, 1, 1, **self.cfg)),
                ('18_conv', nn.Conv2d(256, self.num_anchors * self.num_attrib, 1)),
            ]),
        ]
        self.layers = nn.ModuleList([nn.Sequential(layer_dict) for layer_dict in head])

    def forward(self, feats):
        stem, extra_x = feats
        stage0 = self.layers[0](stem)
        head0 = self.layers[1](stage0)
        stage1 = self.layers[2](stage0)
        stage2 = torch.cat((stage1, extra_x), dim=1)
        head1 = self.layers[3](stage2)  
        return (head0, head1),

    def init_weights(self, pretrained=None):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                constant_init(m, 1)
