'''
This impementation based on https://github.com/hhaAndroid/mmdetection-mini
'''

import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm
from robotflow.rflib.cnn import ConvModule, kaiming_init, constant_init
from robotflow.rflib.runner import load_checkpoint
import logging
from robotflow.rflearner.builder import BACKBONES

class ResBlock(nn.Module):
    def __init__(self, 
                 channels, 
                 hidden_channels=None,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='Mish')):
        cfg = dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        super().__init__()

        if hidden_channels is None:
            hidden_channels = channels
        self.block = nn.Sequential(
            ConvModule(channels, hidden_channels, 1, **cfg),
            ConvModule(hidden_channels, channels, 3, padding =1 ,**cfg)
        )
        
    def forward(self, x):
        return x + self.block(x)

# CSPdarknet的结构块
class CSPBlock(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 num_blocks, 
                 first,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='Mish')):
        super().__init__()
        cfg = dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

        self.downsample_conv = ConvModule(in_channels, out_channels, 3, stride=2, padding=1, **cfg)
        if first:

            self.split_conv0 = ConvModule(out_channels, out_channels, 1, **cfg)
            self.split_conv1 = ConvModule(out_channels, out_channels, 1, **cfg)
            self.blocks_conv = nn.Sequential(
                ResBlock(channels=out_channels, hidden_channels=out_channels // 2, **cfg),
                ConvModule(out_channels, out_channels, 1, **cfg)
            )
            self.concat_conv = ConvModule(out_channels *2 , out_channels, 1, **cfg)
        else:
            self.split_conv0 = ConvModule(out_channels, out_channels // 2, 1, **cfg)
            self.split_conv1 = ConvModule(out_channels, out_channels // 2, 1, **cfg)
            

            self.blocks_conv = nn.Sequential(
                *[ResBlock(out_channels // 2, **cfg) for _ in range(num_blocks)],
                ConvModule(out_channels // 2, out_channels // 2, 1, **cfg)
            )
            self.concat_conv = ConvModule(out_channels, out_channels, 1, **cfg)

    def forward(self, x):
        x = self.downsample_conv(x)
        x0 = self.split_conv0(x)
        x1 = self.split_conv1(x)
        x1 = self.blocks_conv(x1)
        x = torch.cat([x1, x0], dim=1)
        x = self.concat_conv(x)
        return x


@BACKBONES.register_module()
class CSPDarknet(nn.Module):    
    def __init__(self, 
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='Mish')):
        cfg = dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        super().__init__()
        
        self.layers,  = (1, 2, 8, 8, 4),
        self.channels = (32, 64, 128, 256, 512, 1024)
        
        self.conv0 = ConvModule(3, 32, 3, padding=1, **cfg)
        self.stages = nn.ModuleList([
            CSPBlock(self.channels[0], self.channels[1], self.layers[0], first=True, **cfg),
            CSPBlock(self.channels[1], self.channels[2], self.layers[1], first=False,**cfg),
            CSPBlock(self.channels[2], self.channels[3], self.layers[2], first=False,**cfg),
            CSPBlock(self.channels[3], self.channels[4], self.layers[3], first=False,**cfg),
            CSPBlock(self.channels[4], self.channels[5], self.layers[4], first=False,**cfg)
        ])

    def forward(self, x):
        x = self.conv0(x)
        x = self.stages[0](x)
        x = self.stages[1](x)
        out3 = self.stages[2](x)
        out4 = self.stages[3](out3)
        out5 = self.stages[4](out4)
        return [out3, out4, out5]
    
    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)


if __name__ == '__main__':
    import numpy as np

    img = np.zeros((1, 3, 416, 416), dtype='float32')
    tensor = torch.from_numpy(img)
    backbone = CSPDarknet()
    # backbone_dict = backbone.state_dict()
    # total_elements = 0
    # for key,value in backbone_dict.items():
    #     total_elements += value.numel()
    #     print(key,':',value.numel())

    outs = backbone.forward(tensor)
    for out in outs:
        print(out.size())