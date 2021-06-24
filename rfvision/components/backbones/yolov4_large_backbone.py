from rflib.cnn import ConvModule, kaiming_init, constant_init
from rfvision.components.utils.yolo_large_utils import make_divisible, make_round, CSPBlock1
from rfvision.models.builder import BACKBONES

import torch.nn as nn
from collections import OrderedDict
from torch.nn.modules.batchnorm import _BatchNorm


@BACKBONES.register_module()
class YOLOV4LargeBackbone(nn.Module):
    def __init__(self,
                 stage_name='P5',
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='Mish')):
        super().__init__()
        self.cfg = dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        assert stage_name in ['P5','P6','P7']
        
        if stage_name == 'P5':
            width_multiple = 1
            depth_multiple = 1
            
            self.make_div8_fun = make_divisible(divisor = 8, width_multiple = width_multiple)
            self.make_round_fun = make_round(depth_multiple = depth_multiple)
            
            layer_list = self._init_layers_P5()
        elif stage_name == 'P6':
            width_multiple = 1
            depth_multiple = 1
            
            self.make_div8_fun = make_divisible(divisor = 8, width_multiple = width_multiple)
            self.make_round_fun = make_round(depth_multiple = depth_multiple)
            
            layer_list = self._init_layers_P6()
        elif stage_name == 'P7':
            width_multiple = 1
            depth_multiple = 1.25
            
            self.make_div8_fun = make_divisible(divisor = 8, width_multiple = width_multiple)
            self.make_round_fun = make_round(depth_multiple = depth_multiple)
            
            layer_list = self._init_layers_P7()
        self.layers = nn.ModuleList([nn.Sequential(layer_dict) for layer_dict in layer_list])

    def _init_layers_P5(self,):
        layer_list = [
            OrderedDict([
                ('p0_conv0', ConvModule(3,                         self.make_div8_fun(32),   3 , 1, padding=1, **self.cfg)),
                ]),
            OrderedDict([
                ('p1_conv0', ConvModule(self.make_div8_fun(32),   self.make_div8_fun(64),   3 , 2, padding=1, **self.cfg)),
                ('p1_CSP0',   CSPBlock1(self.make_div8_fun(64),   self.make_div8_fun(64),   1 ,               **self.cfg)),
                ]),
            
            OrderedDict([
                ('p2_conv0', ConvModule(self.make_div8_fun(64),   self.make_div8_fun(128),  3 , 2, padding=1, **self.cfg)),
                ('p2_CSP0',   CSPBlock1(self.make_div8_fun(128),  self.make_div8_fun(128),  self.make_round_fun(3) , **self.cfg)),
                ]),
            OrderedDict([
                ('p3_conv0', ConvModule(self.make_div8_fun(128),  self.make_div8_fun(256),  3 , 2, padding=1, **self.cfg)),
                ('p3_CSP0',   CSPBlock1(self.make_div8_fun(256),  self.make_div8_fun(256),  self.make_round_fun(15), **self.cfg)),
                ]),
            OrderedDict([
                ('p4_conv0', ConvModule(self.make_div8_fun(256),  self.make_div8_fun(512),  3 , 2, padding=1, **self.cfg)),
                ('p4_CSP0',   CSPBlock1(self.make_div8_fun(512),  self.make_div8_fun(512),  self.make_round_fun(15), **self.cfg)),
                ]),
            
            OrderedDict([
                ('p5_conv0', ConvModule(self.make_div8_fun(512),  self.make_div8_fun(1024), 3 , 2, padding=1, **self.cfg)),
                ('p5_CSP0',   CSPBlock1(self.make_div8_fun(1024), self.make_div8_fun(1024), self.make_round_fun(7), **self.cfg)),
                ]),
            ]
                
        return layer_list
    
    def _init_layers_P6(self, ):
        layer_list = self._init_layers_P5()
        layer_list.append(
            OrderedDict([
                ('p6_conv0', ConvModule(self.make_div8_fun(1024), self.make_div8_fun(1024), 3 , 2, padding=1, **self.cfg)),
                ('p6_CSP0',   CSPBlock1(self.make_div8_fun(1024), self.make_div8_fun(1024), self.make_round_fun(7) , **self.cfg)),
                ]),
                )
        return layer_list
    def _init_layers_P7(self, ):
        layer_list = self._init_layers_P6()
        layer_list.append(
            OrderedDict([
                ('p7_conv0', ConvModule(self.make_div8_fun(1024), self.make_div8_fun(1024), 3 , 2, padding=1, **self.cfg)),
                ('p7_CSP0',   CSPBlock1(self.make_div8_fun(1024), self.make_div8_fun(1024), self.make_round_fun(7) , **self.cfg)),
                ]),
                )
        return layer_list


    def forward(self, x):
        feats = ()
        feat = x
        for i, layer in enumerate(self.layers):
            feat = layer(feat)
            if i >= 3: # append output in stage 3~ 
                feats += (feat,)
        return feats
    
    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            pass
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')
            
 