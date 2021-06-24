# Copyright (c) 2019 Western Digital Corporation or its affiliates.

from ..builder import DETECTORS
from .single_stage import SingleStageDetector
from rflib.cnn import kaiming_init, constant_init
from rflib.runner import load_checkpoint
import logging
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm

@DETECTORS.register_module()
class YOLOV3(SingleStageDetector):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(YOLOV3, self).__init__(backbone, neck, bbox_head, train_cfg,
                                     test_cfg, pretrained, init_cfg)


@DETECTORS.register_module()
class YOLOV4Tiny(SingleStageDetector):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(YOLOV4Tiny, self).__init__(backbone, neck, bbox_head, train_cfg,
                                         test_cfg, pretrained)

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

@DETECTORS.register_module()
class YOLOV3Tiny(SingleStageDetector):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(YOLOV3Tiny, self).__init__(backbone, neck, bbox_head, train_cfg,
                                         test_cfg, pretrained)

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


@DETECTORS.register_module()
class YOLOV4(SingleStageDetector):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(YOLOV4, self).__init__(backbone, neck, bbox_head, train_cfg,
                                         test_cfg, pretrained)

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
