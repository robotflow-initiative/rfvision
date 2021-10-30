# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import torch.nn as nn
from rfvision.components.keypoint_head import Heatmap3DHead, Heatmap1DHead, MultilabelClassificationHead
from rfvision.core.evaluation_pose.top_down_eval import (
    keypoints_from_heatmaps3d, multilabel_classification_accuracy, pose_pck_accuracy)
from rfvision.core.post_processing_pose import flip_back
from rfvision.models.builder import build_loss, HEADS
from rfvision.components.necks import GlobalAveragePooling
from rflib.runner import BaseModule


@HEADS.register_module()
class Topdown3DHeatmapSimpleHead(BaseModule):
    def __init__(self,
                 keypoint_head_cfg,
                 root_head_cfg,
                 hand_type_head_cfg=None,
                 loss_keypoint=None,
                 loss_root_depth=None,
                 loss_hand_type=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None
                 ):
        super().__init__(init_cfg=init_cfg)
        self.hand_head = Heatmap3DHead(**keypoint_head_cfg)
        self.root_head = Heatmap1DHead(**root_head_cfg)

        if hand_type_head_cfg is not None:
            self.with_hand_type_head = True
            self.hand_type_head = MultilabelClassificationHead(
                **hand_type_head_cfg)
        else:
            self.with_hand_type_head = False

        self.neck = GlobalAveragePooling()

        self.keypoint_loss = build_loss(loss_keypoint)
        self.root_depth_loss = build_loss(loss_root_depth)
        self.hand_type_loss = build_loss(loss_hand_type)

        self.train_cfg = {} if train_cfg is None else train_cfg
        self.test_cfg = {} if test_cfg is None else test_cfg
        self.target_type = self.test_cfg.get('target_type', 'GaussianHeatmap')

    def forward(self, x):
        """Forward function."""
        x = self.extract_feats(x)
        outputs = []
        outputs.append(self.hand_head(x))
        x = self.neck(x)
        outputs.append(self.root_head(x))
        outputs.append(self.hand_type_head(x))
        return outputs

    def get_loss(self, output, target, target_weight):
        """Calculate loss for hand keypoint heatmaps, relative root depth and
        hand type.

        Args:
            output (list[Tensor]): a list of outputs from multiple heads.
            target (list[Tensor]): a list of targets for multiple heads.
            target_weight (list[Tensor]): a list of targets weight for
            multiple heads.
        """
        losses = dict()

        # hand keypoint loss
        assert not isinstance(self.keypoint_loss, nn.Sequential)
        out, tar, tar_weight = output[0], target[0], target_weight[0]
        assert tar.dim() == 5 and tar_weight.dim() == 3
        losses['hand_loss'] = self.keypoint_loss(out, tar, tar_weight)

        # relative root depth loss
        assert not isinstance(self.root_depth_loss, nn.Sequential)
        out, tar, tar_weight = output[1], target[1], target_weight[1]
        assert tar.dim() == 2 and tar_weight.dim() == 2
        losses['rel_root_loss'] = self.root_depth_loss(out, tar, tar_weight)

        # hand type loss
        if self.with_hand_type_head:
            assert not isinstance(self.hand_type_loss, nn.Sequential)
            out, tar, tar_weight = output[2], target[2], target_weight[2]
            assert tar.dim() == 2 and tar_weight.dim() in [1, 2]
            losses['hand_type_loss'] = self.hand_type_loss(out, tar, tar_weight)

        return losses

    def get_accuracy(self, output, target, target_weight):
        if self.with_hand_type_head:
            accuracy = dict()
            avg_acc = multilabel_classification_accuracy(
                output[2].detach().cpu().numpy(),
                target[2].detach().cpu().numpy(),
                target_weight[2].detach().cpu().numpy(),
            )
            accuracy['acc_classification'] = float(avg_acc)
        else:
            pass


    def inference_model(self, x, flip_pairs=None):
        """Inference function.

        Returns:
            output (list[np.ndarray]): list of output hand keypoint
            heatmaps, relative root depth and hand type.

        Args:
            x (torch.Tensor[NxKxHxW]): Input features.
            flip_pairs (None | list[tuple()):
                Pairs of keypoints which are mirrored.
        """
        x = self.extract_feats(x)
        output = self.forward(x)

        if flip_pairs is not None:
            # flip 3D heatmap
            heatmap_3d = output[0]
            N, K, D, H, W = heatmap_3d.shape
            # reshape 3D heatmap to 2D heatmap
            heatmap_3d = heatmap_3d.reshape(N, K * D, H, W)
            # 2D heatmap flip
            heatmap_3d_flipped_back = flip_back(
                heatmap_3d.detach().cpu().numpy(),
                flip_pairs,
                target_type=self.target_type)
            # reshape back to 3D heatmap
            heatmap_3d_flipped_back = heatmap_3d_flipped_back.reshape(
                N, K, D, H, W)
            # feature is not aligned, shift flipped heatmap for higher accuracy
            if self.test_cfg.get('shift_heatmap', False):
                heatmap_3d_flipped_back[...,
                                        1:] = heatmap_3d_flipped_back[..., :-1]
            output[0] = heatmap_3d_flipped_back

            # flip relative hand root depth
            output[1] = -output[1].detach().cpu().numpy()

            # flip hand type
            hand_type = output[2].detach().cpu().numpy()
            hand_type_flipped_back = hand_type.copy()
            hand_type_flipped_back[:, 0] = hand_type[:, 1]
            hand_type_flipped_back[:, 1] = hand_type[:, 0]
            output[2] = hand_type_flipped_back

        else:
            output = [out.detach().cpu().numpy() for out in output]

        return output

    def decode(self, img_metas, output, **kwargs):

        """Decode hand keypoint, relative root depth and hand type.

        Args:
            img_metas (list(dict)): Information about data augmentation
                By default this includes:
                - "image_file: path to the image file
                - "center": center of the bbox
                - "scale": scale of the bbox
                - "rotation": rotation of the bbox
                - "bbox_score": score of bbox
                - "heatmap3d_depth_bound": depth bound of hand keypoint
                 3D heatmap
                - "root_depth_bound": depth bound of relative root depth
                 1D heatmap

            output (list[np.ndarray]): model predicted 3D heatmaps, relative
            root depth and hand type.
        """

        batch_size = len(img_metas)
        result = {}
        # heatmap3d_depth_bound = np.ones(batch_size, dtype=np.float32)
        # root_depth_bound = np.ones(batch_size, dtype=np.float32)
        center = np.zeros((batch_size, 2), dtype=np.float32)
        scale = np.zeros((batch_size, 2), dtype=np.float32)
        image_paths = []
        score = np.ones(batch_size, dtype=np.float32)
        if 'bbox_id' in img_metas[0]:
            bbox_ids = []
        else:
            bbox_ids = None

        for i in range(batch_size):
            heatmap3d_depth_bound = img_metas[i]['heatmap3d_depth_bound']
            root_depth_bound = img_metas[i]['root_depth_bound']
            center[i, :] = img_metas[i]['center']
            scale[i, :] = img_metas[i]['scale']
            image_paths.append(img_metas[i]['image_file'])

            if 'bbox_score' in img_metas[i]:
                score[i] = np.array(img_metas[i]['bbox_score']).reshape(-1)
            if bbox_ids is not None:
                bbox_ids.append(img_metas[i]['bbox_id'])

        all_boxes = np.zeros((batch_size, 6), dtype=np.float32)
        all_boxes[:, 0:2] = center[:, 0:2]
        all_boxes[:, 2:4] = scale[:, 0:2]
        # scale is defined as: bbox_size / 200.0, so we
        # need multiply 200.0 to get bbox size
        all_boxes[:, 4] = np.prod(scale * 200.0, axis=1)
        all_boxes[:, 5] = score
        result['boxes'] = all_boxes
        result['image_paths'] = image_paths
        result['bbox_ids'] = bbox_ids

        # decode 3D heatmaps of hand keypoints
        heatmap3d = output[0]
        preds, maxvals = keypoints_from_heatmaps3d(heatmap3d, center, scale)
        keypoints_3d = np.zeros((batch_size, preds.shape[1], 4),
                                dtype=np.float32)
        keypoints_3d[:, :, 0:3] = preds[:, :, 0:3]
        keypoints_3d[:, :, 3:4] = maxvals
        # transform keypoint depth to camera space
        # tycoer
        # denormalize x * (max - min) + min
        keypoints_3d[:, :, 2] = keypoints_3d[:, :, 2] / self.hand_head.depth_size * (heatmap3d_depth_bound[1] - heatmap3d_depth_bound[0]) + heatmap3d_depth_bound[0]

        result['preds'] = keypoints_3d

        # decode relative hand root depth
        # transform relative root depth to camera space
        # tycoer
        # denormalize x * (max - min) + min
        result['rel_root_depth'] = output[1] / self.root_head.heatmap_size * (root_depth_bound[1] - root_depth_bound[0]) + root_depth_bound[0]

        result['hand_type'] = output[2] > 0.5
        return result

    def extract_feats(self, x):
        if isinstance(x, (tuple, list)):
            x = x[-1]
        return x