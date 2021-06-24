# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 15:24:54 2021

@author: tycoer
"""
from robotflow.rflib.cnn import ConvModule
from robotflow.rflib.cnn.bricks import Mish 
import torch.nn as nn
import torch
import math
import functools

def _make_divisible(x, width_multiple, divisor):
    return math.ceil(x * width_multiple / divisor) * divisor


def _make_round(x, depth_multiple=1.0):
    return max(round(x * depth_multiple), 1) if x > 1 else x


def make_divisible(divisor, width_multiple=1.0):
    return functools.partial(_make_divisible, divisor=divisor, width_multiple=width_multiple)


def make_round(depth_multiple=1.0):
    return functools.partial(_make_round, depth_multiple=depth_multiple)


    
class CSPBlock1(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_resblocks=1,
                 shortcut=True,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='Mish')
                 ):
        super().__init__()
        hidden_channels = int(out_channels // 2)
        cfg = dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

        self.conv1 = ConvModule(in_channels, hidden_channels, 1, 1, **cfg)
        self.conv2 = nn.Conv2d(in_channels, hidden_channels, 1, 1, bias=False)
        self.conv3 = nn.Conv2d(hidden_channels, hidden_channels, 1, 1, bias=False)
        self.conv4 = ConvModule(2 * hidden_channels, out_channels, 1, 1, **cfg)
        self.bn = nn.BatchNorm2d(2 * hidden_channels)
        self.act = Mish()
        self.res = nn.Sequential(*[ResBlock(hidden_channels, hidden_channels, e = 1, shortcut = shortcut) for _ in range(num_resblocks)])
    def forward(self, x):
        y1 = self.conv3(self.res(self.conv1(x)))
        y2 = self.conv2(x)
        return self.conv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))
    
class SPPCSP(nn.Module):
    # CSP SPP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, 
                 in_channels,
                 out_channels,
                 pool_sizes=[5, 9, 13], 
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='Mish')):
        super(SPPCSP, self).__init__()
        cfg = dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

        hidden_channels = out_channels  # hidden channels
        
        self.cv1 = ConvModule(in_channels, hidden_channels, 1, 1, **cfg)
        self.cv2 = nn.Conv2d(in_channels, hidden_channels, 1, 1, bias=False)
        self.cv3 = ConvModule(hidden_channels, hidden_channels, 3, 1, padding=1, **cfg)
        self.cv4 = ConvModule(hidden_channels, hidden_channels, 1, 1, **cfg)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in pool_sizes])
        self.cv5 = ConvModule(4 * hidden_channels, hidden_channels, 1, 1, **cfg)
        self.cv6 = ConvModule(hidden_channels, hidden_channels, 3, 1, padding=1,**cfg)
        self.bn = nn.BatchNorm2d(2 * hidden_channels) 
        self.act = Mish()
        self.cv7 = ConvModule(2 * hidden_channels, out_channels, 1, 1, **cfg)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y1 = self.cv6(self.cv5(torch.cat([x1] + [m(x1) for m in self.m], 1)))
        y2 = self.cv2(x)
        return self.cv7(self.act(self.bn(torch.cat((y1, y2), dim=1))))

class ResBlock(nn.Module):
    def __init__(self, 
                 in_channels,
                 out_channels,
                 e = 1,
                 shortcut = True,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='Mish')):
        cfg = dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        super().__init__()

        hidden_channels = int(out_channels * e)
        self.block = nn.Sequential(
            ConvModule(in_channels, hidden_channels, 1, 1, **cfg),
            ConvModule(hidden_channels, out_channels, 3, 1, padding =1 ,**cfg)
        )
        self.add = shortcut and in_channels == out_channels
    def forward(self, x):
        if self.add == True:
            return x + self.block(x)
        else:
            return self.block(x)
    
class CSPBlock2(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_resblocks=1,
                 shortcut=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='Mish')
                 ):
        super().__init__()
        hidden_channels = int(out_channels)
        cfg = dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

        self.cv1 = ConvModule(in_channels, hidden_channels, 1, 1, **cfg)
        self.cv2 = nn.Conv2d(hidden_channels, hidden_channels, 1, 1, bias=False)
        self.cv3 = ConvModule(2 * hidden_channels, out_channels, 1, 1, **cfg)
        self.bn = nn.BatchNorm2d(2 * hidden_channels)
        self.act = Mish()
        self.m = nn.Sequential(*[ResBlock(hidden_channels, hidden_channels, e = 1, shortcut = shortcut) for _ in range(num_resblocks)])
    def forward(self, x):
        x1 = self.cv1(x)
        y1 = self.m(x1)
        y2 = self.cv2(x1)
        return self.cv3(self.act(self.bn(torch.cat((y1, y2), dim=1))))