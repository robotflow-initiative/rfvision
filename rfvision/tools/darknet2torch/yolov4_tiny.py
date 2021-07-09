'''
This impementation based on https://github.com/hhaAndroid/mmdetection-mini
'''


from rfvision.tools.darknet2torch.darknet_weight_converter import WeightLoader
from rflib.cnn import ConvModule
from rfvision.components.backbones.yolov4_tiny_backbone import ResConv2dBatchLeaky
import torch
import torch.nn as nn
from collections import OrderedDict


class YOLOV4Tiny(nn.Module):

    def __init__(self, pretrained=None):
        super().__init__()
        self.custom_layers = (ResConv2dBatchLeaky,)
        cfg = dict(conv_cfg=None,
                   norm_cfg=dict(type='BN', requires_grad=True),
                   act_cfg=dict(type='LeakyReLU', negative_slope=0.1))
        # Network
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


        head = [
            OrderedDict([
                ('10_max', nn.MaxPool2d(2, 2)),
                ('11_conv', ConvModule(512, 512, 3, 1, 1, **cfg)),
                ('12_conv', ConvModule(512, 256, 1, 1 , **cfg)),
            ]),

            OrderedDict([
                ('13_conv', ConvModule(256, 512, 3, 1, 1, **cfg)),
                ('14_conv', nn.Conv2d(512, 3 * (5 + 80), 1)),
            ]),

            OrderedDict([
                ('15_convbatch', ConvModule(256, 128, 1, 1, **cfg)),
                ('16_upsample', nn.Upsample(scale_factor=2)),
            ]),

            OrderedDict([
                ('17_convbatch', ConvModule(384, 256, 3, 1, 1, **cfg)),
                ('18_conv', nn.Conv2d(256, 3 * (5 + 80), 1)),
            ]),
        ]

        self.backbone = nn.Sequential(backbone)
        self.head = nn.ModuleList([nn.Sequential(layer_dict) for layer_dict in head])
        self.init_weights(pretrained)

    def __modules_recurse(self, mod=None):
        """ This function will recursively loop over all module children.
        Args:
            mod (torch.nn.Module, optional): Module to loop over; Default **self**
        """
        if mod is None:
            mod = self
        for module in mod.children():
            if isinstance(module, (nn.ModuleList, nn.Sequential, self.custom_layers)):
                yield from self.__modules_recurse(module)
            else:
                yield module

    def init_weights(self, pretrained=None):
        if pretrained is not None:
            weights = WeightLoader(pretrained)
            for module in self.__modules_recurse():
                try:
                    weights.load_layer(module)
                    print(f'Layer loaded: {module}')
                    if weights.start >= weights.size:
                        print(f'Finished loading weights [{weights.start}/{weights.size} weights]')
                        break
                except NotImplementedError:
                    print(f'Layer skipped: {module.__class__.__name__}')

    def forward(self, x):
        stem, extra_x = self.backbone(x)
        stage0 = self.head[0](stem)
        head0 = self.head[1](stage0)

        stage1 = self.head[2](stage0)
        stage2 = torch.cat((stage1, extra_x), dim=1)
        head1 = self.head[3](stage2)
        head = [head1, head0]
        return head


if __name__ == '__main__':
    weights_path = '/home/hanyang/weights/yolov4-tiny.weights'
    m = YOLOV4Tiny(pretrained=weights_path)
    new_state_dict = OrderedDict()
    for k, v in m.state_dict().items():
        if k.startswith('backbone'):
            name = k.replace('backbone', 'backbone.layers')
        else:
            name = k.replace('head', 'bbox_head.layers')
        new_state_dict[name] = v
    data = {"state_dict": new_state_dict}
    torch.save(data, '/home/hanyang/weights/yolov4_tiny.pth')