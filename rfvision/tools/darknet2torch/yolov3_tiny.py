'''
This impementation based on https://github.com/hhaAndroid/mmdetection-mini
'''



from rfvision.tools.darknet2torch.darknet_weight_converter import WeightLoader
from rflib.cnn import ConvModule
import torch.nn as nn
from collections import OrderedDict


class YOLOV3Tiny(nn.Module):

    def __init__(self, pretrained=None):
        super().__init__()
        cfg = dict(conv_cfg=None,
                   norm_cfg=dict(type='BN', requires_grad=True),
                   act_cfg=dict(type='LeakyReLU', negative_slope=0.1))
        # Network
        backbone = [
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

        head = [
            OrderedDict([
                ('14_convbatch', ConvModule(256, 512, 3, 1, 1, **cfg)),
                ('15_conv', nn.Conv2d(512, 3 * (5 + 80), 1)),
            ]),
            # stage 5
            # stage5 / upsample
            OrderedDict([
                ('18_convbatch', ConvModule(256, 128, 1, 1, **cfg)),
                ('19_upsample', nn.Upsample(scale_factor=2)),
            ]),
            # stage5 / head
            OrderedDict([
                ('21_convbatch', ConvModule(256 + 128, 256, 3, 1, 1, **cfg)),
                ('22_conv', nn.Conv2d(256, 3 * (5 + 80), 1)),
            ]),
        ]

        self.backbone = nn.ModuleList([nn.Sequential(layer_dict) for layer_dict in backbone])
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
            if isinstance(module, (nn.ModuleList, nn.Sequential)):
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


if __name__ =='__main__':
    import torch

    # darknet 权重路径 https://github.com/AlexeyAB/darknet
    tiny_yolov3_weights_path = '/home/hanyang/weights/yolov3-tiny.weights'

    tiny_yolov3 = YOLOV3Tiny(pretrained=tiny_yolov3_weights_path)

    new_state_dict = OrderedDict()
    for k, v in tiny_yolov3.state_dict().items():
        if k.startswith('backbone'):
            name = k.replace('backbone', 'backbone.layers')
        elif k.startswith('head'):
            name = k.replace('head', 'bbox_head.layers')
        new_state_dict[name] = v
    data = {"state_dict": new_state_dict}
    torch.save(data, '/home/hanyang/weights/yolov3_tiny.pth')