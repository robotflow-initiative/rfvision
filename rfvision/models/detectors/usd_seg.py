import torch
import numpy as np
from rfvision.models.builder import DETECTORS
from .yolo import YOLOV3
from .fcos import FCOS
from rfvision.core import bbox_mask2result


@DETECTORS.register_module()
class USDSegYOLOV3(YOLOV3):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 num_bases=-1,
                 bases_path=None,
                 method='None',
                 use_mask_loss=False):
        super(USDSegYOLOV3, self).__init__(backbone, neck, bbox_head,
                                           train_cfg, test_cfg, init_cfg)

        self.register_buffer('bases', torch.tensor(np.load(bases_path)).float())
        assert num_bases == len(self.bases)
        self.num_bases = num_bases

        if method not in ['cosine', 'cosine_r']:
            raise NotImplementedError('%s not supported.' % method)
        self.method = method
        self.use_mask_loss = use_mask_loss

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_coefs,
                      gt_bboxes_ignore=None,
                      gt_resized_masks=None,
                      ):

        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        loss_inputs = outs + (gt_bboxes, gt_labels, gt_coefs, img_metas)

        losses = self.bbox_head.loss(
            *loss_inputs,
            gt_bboxes_ignore=gt_bboxes_ignore,
            bases=self.bases if self.use_mask_loss else None,
            gt_resized_masks=gt_resized_masks
        )
        return losses

    def simple_test(self, img, img_meta, rescale=False):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)

        bbox_inputs = outs + (img_meta, self.test_cfg, rescale)
        bbox_list = self.bbox_head.get_bboxes(*bbox_inputs)


        if self.method == 'cosine' or self.method == 'cosine_r':
            results = [
                bbox_mask2result(det_bboxes, det_coefs, det_labels, self.bbox_head.num_classes, img_meta[0],
                                 self.bases, self.method)
                for det_bboxes, det_labels, det_coefs in bbox_list]

        return results


@DETECTORS.register_module()
class USDSegFCOS(FCOS):
    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 method='None',
                 bases_path=None,
                 num_bases=-1
                 ):
        super(USDSegFCOS, self).__init__(backbone, neck, bbox_head, train_cfg,
                                   test_cfg, init_cfg)
        if method not in ['cosine']:
            raise RuntimeError('%s method is not supported!' % method)
        self.method = method

        self.register_buffer('bases', torch.tensor(np.load(bases_path)).float())
        assert num_bases == len(self.bases)
        self.num_bases = num_bases

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_coefs,
                      gt_bboxes_ignore=None):
        x = self.extract_feat(img)
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_coefs,
                                              gt_bboxes_ignore)
        return losses

    def simple_test(self, img, img_metas, rescale=False):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        bbox_list = self.bbox_head.get_bboxes(
            *outs, img_metas, rescale=rescale)
        # skip post-processing when exporting to ONNX
        if torch.onnx.is_in_onnx_export():
            return bbox_list

        results = [
            bbox_mask2result(det_bboxes, det_coefs, det_labels, self.bbox_head.num_classes, img_metas[0],
                                self.bases, self.method)
            for det_bboxes, det_labels, det_coefs in bbox_list]

        return results

