from rflib.cnn import ConvModule, kaiming_init, constant_init
from rfvision.components.utils.yolo_large_utils import SPPCSP, CSPBlock2, make_divisible, make_round
from rfvision.models.builder import NECKS

from torch.nn.modules.batchnorm import _BatchNorm
import torch.nn as nn
import torch
@NECKS.register_module()
class YOLOV4LargeNeck(nn.Module):
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
                        
            self.layers = self._init_layers_P5()
        elif stage_name == 'P6':
            width_multiple = 1
            depth_multiple = 1
            
            self.make_div8_fun = make_divisible(divisor = 8, width_multiple = width_multiple)
            self.make_round_fun = make_round(depth_multiple = depth_multiple)
            
            self.layers = self._init_layers_P6()
        elif stage_name == 'P7':
            width_multiple = 1.25
            depth_multiple = 1
            
            self.make_div8_fun = make_divisible(divisor = 8, width_multiple = width_multiple)
            self.make_round_fun = make_round(depth_multiple = depth_multiple)
            
            self.layers = self._init_layers_P7()
        self.stage_name = stage_name
        
    def forward(self, feats):
        if self.stage_name == 'P5':
            return self.forward_P5(feats)
        elif self.stage_name == 'P6':
            return self.forward_P6(feats)        
        elif self.stage_name == 'P7':
            return self.forward_P7(feats)
        
        
    def _init_layers_P5(self):
        # p3 out_channels 256
        # p4 out_channels 512
        # p5 out_channels 1024
        layers = nn.ModuleList([
            SPPCSP(    self.make_div8_fun(1024), self.make_div8_fun(512), **self.cfg),               #0
            ConvModule(self.make_div8_fun(512),  self.make_div8_fun(256),  1, 1, **self.cfg),       #1
            nn.Upsample(scale_factor=2, mode='nearest'),      #2
            ConvModule(self.make_div8_fun(512),  self.make_div8_fun(256),  1, 1, **self.cfg),       #3
            
            CSPBlock2( self.make_div8_fun(512),  self.make_div8_fun(256), num_resblocks=self.make_round_fun(3), **self.cfg), #4
            ConvModule(self.make_div8_fun(256),  self.make_div8_fun(128),  1, 1, **self.cfg),          #5
            nn.Upsample(scale_factor=2, mode='nearest'),      #6
            ConvModule(self.make_div8_fun(256),  self.make_div8_fun(128),  1, 1, **self.cfg),          #7
            CSPBlock2( self.make_div8_fun(256),  self.make_div8_fun(128), num_resblocks=self.make_round_fun(3), **self.cfg), #8
            ConvModule(self.make_div8_fun(128),  self.make_div8_fun(256),  3, 1, padding = 1, **self.cfg),       #9
            ConvModule(self.make_div8_fun(128),  self.make_div8_fun(256),  3, 2, padding = 1, **self.cfg),       #10
            CSPBlock2( self.make_div8_fun(512),  self.make_div8_fun(256), num_resblocks=self.make_round_fun(3), **self.cfg), #11
            ConvModule(self.make_div8_fun(256),  self.make_div8_fun(512),  3, 1, padding = 1, **self.cfg),       #12
            ConvModule(self.make_div8_fun(256),  self.make_div8_fun(512),  3, 2, padding = 1, **self.cfg),       #13
            CSPBlock2( self.make_div8_fun(1024), self.make_div8_fun(512), num_resblocks=self.make_round_fun(3), **self.cfg), #14
            ConvModule(self.make_div8_fun(512),  self.make_div8_fun(1024), 3, 1, padding = 1, **self.cfg),       #15
            ])
        return layers
    
    
    def _init_layers_P6(self):
        # p3 out_channels 256
        # p4 out_channels 512
        # p5 out_channels 1024
        # p6 out_channels 1024
        layers = nn.ModuleList([
            SPPCSP(    self.make_div8_fun(1024), self.make_div8_fun(512), **self.cfg),               #0
            ConvModule(self.make_div8_fun(512),  self.make_div8_fun(512),  1, 1, **self.cfg),        #1
            nn.Upsample(scale_factor=2, mode='nearest'), #2
            ConvModule(self.make_div8_fun(1024), self.make_div8_fun(512),  1, 1, **self.cfg),   #3
            CSPBlock2( self.make_div8_fun(1024), self.make_div8_fun(512), num_resblocks=self.make_round_fun(3), **self.cfg),#4
            ConvModule(self.make_div8_fun(512),  self.make_div8_fun(256),  1, 1, **self.cfg),       #5
            nn.Upsample(scale_factor=2, mode='nearest'),      #6
            ConvModule(self.make_div8_fun(512),  self.make_div8_fun(256),  1, 1, **self.cfg),       #7
            CSPBlock2( self.make_div8_fun(512),  self.make_div8_fun(256), num_resblocks=self.make_round_fun(3), **self.cfg), #8
            ConvModule(self.make_div8_fun(256),  self.make_div8_fun(128),  1, 1, **self.cfg),          #9
            nn.Upsample(scale_factor=2, mode='nearest'),      #10
            ConvModule(self.make_div8_fun(256),  self.make_div8_fun(128),  1, 1, **self.cfg),          #11
            CSPBlock2( self.make_div8_fun(256),  self.make_div8_fun(128), num_resblocks=self.make_round_fun(3), **self.cfg), #12
            ConvModule(self.make_div8_fun(128),  self.make_div8_fun(256),  3, 1, padding = 1, **self.cfg),#13
            ConvModule(self.make_div8_fun(128),  self.make_div8_fun(256),  3, 2, padding = 1, **self.cfg),#14
            CSPBlock2( self.make_div8_fun(512),  self.make_div8_fun(256), num_resblocks=self.make_round_fun(3), **self.cfg), #15
            ConvModule(self.make_div8_fun(256),  self.make_div8_fun(512),  3, 1, padding = 1, **self.cfg),#16
            ConvModule(self.make_div8_fun(256),  self.make_div8_fun(512),  3, 2, padding = 1, **self.cfg),#17
            CSPBlock2( self.make_div8_fun(1024), self.make_div8_fun(512), num_resblocks=self.make_round_fun(3), **self.cfg), #18
            ConvModule(self.make_div8_fun(512),  self.make_div8_fun(1024), 3, 1, padding = 1, **self.cfg),#19
            ConvModule(self.make_div8_fun(512),  self.make_div8_fun(512),  3, 2, padding = 1, **self.cfg),#20
            CSPBlock2( self.make_div8_fun(1024), self.make_div8_fun(512), num_resblocks=self.make_round_fun(3), **self.cfg), #21
            ConvModule(self.make_div8_fun(512),  self.make_div8_fun(1024), 3, 1, padding = 1, **self.cfg),#22
            ])
        return layers
        
    def _init_layers_P7(self):
        # p3 out_channels 256
        # p4 out_channels 512
        # p5 out_channels 1024
        # p6 out_channels 1024
        # p7 out_channels 1024

        layers = nn.ModuleList([
            SPPCSP(    self.make_div8_fun(1024), self.make_div8_fun(512), **self.cfg),               #0
            ConvModule(self.make_div8_fun(512),  self.make_div8_fun(512),  1, 1, **self.cfg),        #1
            nn.Upsample(scale_factor=2, mode='nearest'), #2
            
            ConvModule(self.make_div8_fun(1024), self.make_div8_fun(512),  1, 1, **self.cfg),   #3
            CSPBlock2( self.make_div8_fun(1024), self.make_div8_fun(512), num_resblocks=self.make_round_fun(3), **self.cfg),#4
            ConvModule(self.make_div8_fun(512),  self.make_div8_fun(512),  1, 1, **self.cfg),       #5
            nn.Upsample(scale_factor=2, mode='nearest'),      #6
            
            ConvModule(self.make_div8_fun(1024), self.make_div8_fun(512),  1, 1, **self.cfg),   #7
            CSPBlock2( self.make_div8_fun(1024), self.make_div8_fun(512), num_resblocks=self.make_round_fun(3), **self.cfg),#8
            ConvModule(self.make_div8_fun(512),  self.make_div8_fun(256),  1, 1, **self.cfg),       #9
            nn.Upsample(scale_factor=2, mode='nearest'),      #10
            
            ConvModule(self.make_div8_fun(512),  self.make_div8_fun(256),  1, 1, **self.cfg),       #11
            CSPBlock2( self.make_div8_fun(512),  self.make_div8_fun(256), num_resblocks=self.make_round_fun(3), **self.cfg), #12
            ConvModule(self.make_div8_fun(256),  self.make_div8_fun(128),  1, 1, **self.cfg),          #13
            nn.Upsample(scale_factor=2, mode='nearest'),      #14
            ConvModule(self.make_div8_fun(256),  self.make_div8_fun(128),  1, 1, **self.cfg),          #15
            CSPBlock2( self.make_div8_fun(256),  self.make_div8_fun(128), num_resblocks=self.make_round_fun(3), **self.cfg), #16
            ConvModule(self.make_div8_fun(128),  self.make_div8_fun(256),  3, 1, padding = 1, **self.cfg),#17
            ConvModule(self.make_div8_fun(128),  self.make_div8_fun(256),  3, 2, padding = 1, **self.cfg),#18
            CSPBlock2( self.make_div8_fun(512),  self.make_div8_fun(256), num_resblocks=self.make_round_fun(3), **self.cfg), #19
            ConvModule(self.make_div8_fun(256),  self.make_div8_fun(512),  3, 1, padding = 1, **self.cfg),#20
            ConvModule(self.make_div8_fun(256),  self.make_div8_fun(512),  3, 2, padding = 1, **self.cfg),#21
            CSPBlock2( self.make_div8_fun(1024), self.make_div8_fun(512), num_resblocks=self.make_round_fun(3), **self.cfg), #22
            ConvModule(self.make_div8_fun(512),  self.make_div8_fun(1024), 3, 1, padding = 1, **self.cfg),#23
            ConvModule(self.make_div8_fun(512),  self.make_div8_fun(512),  3, 2, padding = 1, **self.cfg),#24
            CSPBlock2( self.make_div8_fun(1024), self.make_div8_fun(512), num_resblocks=self.make_round_fun(3), **self.cfg), #25
            ConvModule(self.make_div8_fun(512),  self.make_div8_fun(1024), 3, 1, padding = 1, **self.cfg),#26
            ConvModule(self.make_div8_fun(512),  self.make_div8_fun(512),  3, 2, padding = 1, **self.cfg),#27
            CSPBlock2( self.make_div8_fun(1024), self.make_div8_fun(512), num_resblocks=self.make_round_fun(3), **self.cfg), #28
            ConvModule(self.make_div8_fun(512),  self.make_div8_fun(1024), 3, 1, padding = 1, **self.cfg),#29
            ])
        return layers
    
        
    def forward_P5(self, feats):
        p3, p4, p5 = feats
        
        x0 = self.layers[0](p5) # 512
        x1 = self.layers[1](x0) # 256
        x2 = self.layers[2](x1) # 256
        
        x3 = self.layers[3](p4) # 256
        x_2cat3 = torch.cat((x2, x3), dim=1) # 512

        x4 = self.layers[4](x_2cat3) # 256
        x5 = self.layers[5](x4) # 128
        x6 = self.layers[6](x5) # 128
    
        x7 = self.layers[7](p3) # 128
        x_6cat7 = torch.cat((x6, x7), dim=1) # 256
        
        x8 = self.layers[8](x_6cat7) # 128
        x9 = self.layers[9](x8) #p3 #256
        x10 = self.layers[10](x8) # 256
        
        x_4cat10 = torch.cat((x4 , x10), dim=1) # 512
        x11 = self.layers[11](x_4cat10)   # 256
        x12 = self.layers[12](x11) #p4 # 512
        x13 = self.layers[13](x11) # 512
        x_0cat12 = torch.cat((x0, x13),dim=1) # 1024
        
        
        x14 = self.layers[14](x_0cat12) # 512 
        x15 = self.layers[15](x14) #p5 # 1024
        return x15, x12, x9 # 1024, 512, 256 
    
    def forward_P6(self, feats):
        p3, p4, p5, p6 = feats 
        x0 = self.layers[0](p6) # 512
        x1 = self.layers[1](x0) # 512
        x2 = self.layers[2](x1) # 512
        
        x3 = self.layers[3](p5) # 512
        x_2cat3 = torch.cat((x2, x3), dim=1) # 1024

        x4 = self.layers[4](x_2cat3) # 512
        x5 = self.layers[5](x4) # 256
        x6 = self.layers[6](x5) # 256
    
        x7 = self.layers[7](p4) # 256
        x_6cat7 = torch.cat((x6, x7), dim=1) # 512
        
        x8 = self.layers[8](x_6cat7) # 256
        x9 = self.layers[9](x8)  #128
        x10 = self.layers[10](x9) # 128
        x11 = self.layers[11](p3) # 128
        x_10cat11 = torch.cat((x10 , x11), dim=1) # 256

        x12 = self.layers[12](x_10cat11)   # 128
        x13 = self.layers[13](x12) #p3 # 256
        x14 = self.layers[14](x12) # 256
        x_8cat14 = torch.cat((x8, x14),dim=1) # 512
        
        x15 = self.layers[15](x_8cat14) # 256 
        x16 = self.layers[16](x15) #p4 # 512
        x17 = self.layers[17](x15) #512
        
        x_4cat17 = torch.cat((x4 , x17), dim=1) # 1024
        x18 = self.layers[18](x_4cat17) #p5 512
        x19 = self.layers[19](x18) #1024
        x20 = self.layers[20](x18) # 512
        
        x_0cat20 = torch.cat((x0, x20), dim = 1) # 1024
        x21 = self.layers[21](x_0cat20) #512
        x22 = self.layers[22](x21) #p6 #1024

        return x22, x19, x16, x13 # 1024, 1024, 512, 256
        
    def forward_P7(self, feats):
        p3, p4, p5, p6, p7 = feats 
        
        x0 = self.layers[0](p7) # 512
        x1 = self.layers[1](x0) # 512
        x2 = self.layers[2](x1) # 512
        
        x3 = self.layers[3](p6) # 512
        x_2cat3 = torch.cat((x2, x3), dim=1) # 1024

        x4 = self.layers[4](x_2cat3) # 512
        x5 = self.layers[5](x4) # 512
        x6 = self.layers[6](x5) # 512
    
        x7 = self.layers[7](p5) # 512
        x_6cat7 = torch.cat((x6, x7), dim=1) # 1024
        
        x8 = self.layers[8](x_6cat7) # 512
        x9 = self.layers[9](x8)  #256
        x10 = self.layers[10](x9) # 256
        x11 = self.layers[11](p4) # 256
        x_10cat11 = torch.cat((x10 , x11), dim=1) # 512

        x12 = self.layers[12](x_10cat11)   # 256
        x13 = self.layers[13](x12) #p3 # 128
        x14 = self.layers[14](x13) # 128
        
        x15 = self.layers[15](p3) # 128 
        x_14cat15 = torch.cat((x14, x15),dim=1) # 256

        x16 = self.layers[16](x_14cat15) #p4 # 128
        x17 = self.layers[17](x16) #256
        x18 = self.layers[18](x16) #p5 256
        x_12cat18 = torch.cat((x12 , x18), dim=1) # 512
        
        x19 = self.layers[19](x_12cat18) #p5 256
        x20 = self.layers[20](x19) # 512
        x21 = self.layers[21](x19) # 512
        x_8cat21 = torch.cat((x8, x21), dim = 1) # 1024
        x22 = self.layers[22](x_8cat21) #512
        x23 = self.layers[23](x22) #p6 #1024
        x24 = self.layers[24](x22) #512
        x_4cat24 = torch.cat((x4, x24), dim = 1) # 1024

        x25 = self.layers[25](x_4cat24)#512
        x26 = self.layers[26](x25)#1024
        x27 = self.layers[27](x25)#512
        x_0cat27 = torch.cat((x4, x27), dim = 1) # 1024
        x28 = self.layers[26](x_0cat27)#512
        x29 = self.layers[27](x28)#1024
        return x29, x26, x23, x20, x17 # 1024, 1024, 1024, 512, 256
        
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
        
