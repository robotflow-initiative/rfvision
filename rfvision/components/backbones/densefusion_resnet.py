import torch
import torch.nn as nn
from torch.nn import functional as F

from rfvision.models.builder import BACKBONES
from .resnet import ResNet


class PSPModule(nn.Module):
    def __init__(self, features, out_features=1024, sizes=(1, 2, 3, 6)):
        super(PSPModule, self).__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1)
        self.relu = nn.ReLU()

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.interpolate(input=stage(feats), size=(h, w), mode='bilinear') for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)


class PSPUpsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PSPUpsample, self).__init__()
        self.conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode = 'bilinear'),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.PReLU()
        )

    def forward(self, x):
        return self.conv(x)

class PSPNet(nn.Module):
    def __init__(self, n_classes=21, sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, depth=18,
                  pretrained=False):
        super(PSPNet, self).__init__()
        self.feats = ResNet(depth,                            
                            out_indices=(2,3),
                            strides=(1,2,1,1), 
                            dilations = (1,1,2,4))
        self.psp = PSPModule(psp_size, 1024, sizes)
        self.drop_1 = nn.Dropout2d(p=0.3)

        self.up_1 = PSPUpsample(1024, 256)
        self.up_2 = PSPUpsample(256, 64)
        self.up_3 = PSPUpsample(64, 64)

        self.drop_2 = nn.Dropout2d(p=0.15)
        self.final = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1),
            nn.LogSoftmax(dim=1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(deep_features_size, 256),
            nn.ReLU(),
            nn.Linear(256, n_classes)
        )

    def forward(self, x):
        class_f, f = self.feats(x)
        p = self.psp(f)
        p = self.drop_1(p)

        p = self.up_1(p)
        p = self.drop_2(p)

        p = self.up_2(p)
        p = self.drop_2(p)

        p = self.up_3(p)

        return self.final(p)

psp_models = {
    'resnet18': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, depth=18),
    'resnet34': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, depth=34),
    'resnet50': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, depth=50),
    'resnet101': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, depth=101),
    'resnet152': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, depth=152)
}



@BACKBONES.register_module()
class DenseFusionResNet(nn.Module):

    def __init__(self, usegpu=True):
        super(DenseFusionResNet, self).__init__()

        self.model = psp_models['resnet18'.lower()]()

    def forward(self, x):
        x = self.model(x)
        return x
