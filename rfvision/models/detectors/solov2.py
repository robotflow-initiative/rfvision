from .single_stage_instance_seg import SingleStageInstanceSegmentor
from ..builder import DETECTORS, build_head
import torch

@DETECTORS.register_module()
class SOLOv2(SingleStageInstanceSegmentor):

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 mask_head=None,
                 mask_feat_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,):

        super(SOLOv2, self).__init__(backbone=backbone,
                                     neck=neck,
                                     bbox_head=bbox_head,
                                     mask_head=mask_head,
                                     train_cfg=train_cfg,
                                     test_cfg=test_cfg,
                                     init_cfg=init_cfg)

        if mask_feat_head is not None:
            self.mask_feat_head = build_head(mask_feat_head)

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None):
        gt_masks = [
            gt_mask.to_tensor(dtype=torch.bool, device=img.device)
            for gt_mask in gt_masks
        ]

        ############## tycoer #############
        gt_labels = [gt_label + 1 for gt_label in gt_labels]
        ###################################
        x = self.extract_feat(img)
        outs = self.mask_head(x)
        # outs = self.bbox_head(x)

        if self.mask_feat_head:
            mask_feat_pred = self.mask_feat_head(
                x[self.mask_feat_head.start_level:self.mask_feat_head.end_level + 1])
            loss_inputs = outs + (mask_feat_pred, gt_bboxes, gt_labels, gt_masks, img_metas, self.train_cfg)
        else:
            loss_inputs = outs + (gt_bboxes, gt_labels, gt_masks, img_metas, self.train_cfg)
        losses = self.mask_head.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        # losses = self.bbox_head.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

    def simple_test(self, img, img_metas, rescale=False):
        x = self.extract_feat(img)
        outs = self.mask_head(x, eval=True)
        # outs = self.bbox_head(x, eval=True)

        if self.mask_feat_head:
        # if self.with_mask_feat_head:
            mask_feat_pred = self.mask_feat_head(
                x[self.mask_feat_head.
                  start_level:self.mask_feat_head.end_level + 1])
            seg_inputs = outs + (mask_feat_pred, img_metas, self.test_cfg, rescale)
        else:
            seg_inputs = outs + (img_metas, self.test_cfg, rescale)
        seg_result = self.mask_head.get_seg(*seg_inputs)

        format_results_list = []
        for results in seg_result:
            format_results_list.append(self.format_results(results))

        return format_results_list


