'''
This impementation based on https://github.com/hhaAndroid/mmdetection-mini
'''

from rfvision.tools.darknet2torch.darknet_weight_converter import WeightLoader
from rflib.cnn import ConvModule
from rfvision.components.backbones.cspdarknet import CSPBlock, ResBlock
from rfvision.components.necks.yolo_neck import MakeNConv, FuseStage, SpatialPyramidPooling
import torch.nn as nn
import torch
from collections import OrderedDict

class Head(nn.Module):
    def __init__(self, nchannels, nanchors, nclasses):
        super().__init__()
        cfg = dict(conv_cfg=None,
                   norm_cfg=dict(type='BN', requires_grad=True),
                   act_cfg=dict(type='LeakyReLU', negative_slope=0.1))
        mid_nchannels = 2 * nchannels
        layer_list = [
            ConvModule(nchannels, mid_nchannels, 3, 1, 1, **cfg),
            nn.Conv2d(mid_nchannels, nanchors * (5 + nclasses), 1, 1, 0),
        ]
        self.feature = nn.Sequential(*layer_list)

    def forward(self, data):
        x = self.feature(data)
        return x



class YOLOV4(nn.Module):
    def __init__(self, layers=(1, 2, 8, 8, 4), pretrained=False):
        super().__init__()
        cfg = dict(conv_cfg=None,
                   norm_cfg=dict(type='BN', requires_grad=True),
                   act_cfg=dict(type='LeakyReLU', negative_slope=0.1))
        self.custom_layers = (CSPBlock, ResBlock, Head, MakeNConv, FuseStage)
        num_anchors = (3, 3, 3)
        in_channels_list = (512, 256, 128)
        self.inplanes = 32
        self.conv0 = ConvModule(3, self.inplanes, kernel_size=3, stride=1, **cfg)
        self.feature_channels = [64, 128, 256, 512, 1024]

        self.backbone = nn.ModuleList([
            CSPBlock(self.inplanes, self.feature_channels[0], layers[0], first=True),
            CSPBlock(self.feature_channels[0], self.feature_channels[1], layers[1], first=False),
            CSPBlock(self.feature_channels[1], self.feature_channels[2], layers[2], first=False),
            CSPBlock(self.feature_channels[2], self.feature_channels[3], layers[3], first=False),
            CSPBlock(self.feature_channels[3], self.feature_channels[4], layers[4], first=False)
        ])

        self.neck1 = nn.ModuleList([
            # neck1
            nn.Sequential(MakeNConv(1024, 512, 3, **cfg, ),
                          SpatialPyramidPooling(),
                          MakeNConv(2048, 512, 3, **cfg)),

            nn.Sequential(FuseStage(512, **cfg),
                          MakeNConv(512, 256, 5, **cfg)),

            nn.Sequential(FuseStage(256, **cfg),
                          MakeNConv(256, 128, 5, **cfg)),
        ])
        self.head3 = nn.ModuleList([nn.Sequential(Head(in_channels_list[2], num_anchors[2], 80)),])
        self.neck2 = nn.ModuleList([nn.Sequential(FuseStage(128, **cfg, is_reversal=True), MakeNConv(512, 256, 5, **cfg)),])
        self.head2 = nn.ModuleList([nn.Sequential(Head(in_channels_list[1], num_anchors[1], 80)),])
        self.neck3 = nn.ModuleList([nn.Sequential(FuseStage(256, **cfg, is_reversal=True), MakeNConv(1024, 512, 5, **cfg)),])
        self.head1 = nn.ModuleList([nn.Sequential(Head(in_channels_list[0], num_anchors[0], 80)),])
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
        return x

if __name__ == '__main__':
    weight_path = '/home/hanyang/weights/yolov4_multisize_mish_leaky.weights'
    m = YOLOV4(pretrained=weight_path)
    new_state_dict = OrderedDict()

    for k, v in m.state_dict().items():
        if k.startswith('conv0'):
            name = k.replace('conv0', 'backbone.conv0')
        elif k.startswith('backbone'):
            name = k.replace('backbone', 'backbone.stages')
        elif k.startswith('neck1'):
            name = k.replace('neck1', 'neck.layers')
        elif k.startswith('neck2'):
            name = k.replace('neck2.0', 'neck.layers.3')
        elif k.startswith('neck3'):
            name = k.replace('neck3.0', 'neck.layers.4')
        elif k.startswith('head1.0.0.feature.0'):
            name = k.replace('head1.0.0.feature.0', 'bbox_head.convs_bridge.0')
        elif k.startswith('head2.0.0.feature.0'):
            name = k.replace('head2.0.0.feature.0', 'bbox_head.convs_bridge.1')
        elif k.startswith('head3.0.0.feature.0'):
            name = k.replace('head3.0.0.feature.0', 'bbox_head.convs_bridge.2')
        elif k.startswith('head1.0.0.feature.1'):
            name = k.replace('head1.0.0.feature.1', 'bbox_head.convs_pred.0')
        elif k.startswith('head2.0.0.feature.1'):
            name = k.replace('head2.0.0.feature.1', 'bbox_head.convs_pred.1')
        elif k.startswith('head3.0.0.feature.1'):
            name = k.replace('head3.0.0.feature.1', 'bbox_head.convs_pred.2')
        else:
            name = k
        new_state_dict[name] = v
    data = {"state_dict": new_state_dict}
    torch.save(data, '/home/hanyang/weights/yolov4_multisize_mish_leaky.pth')