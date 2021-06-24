# Copyright (c) 2019 Western Digital Corporation or its affiliates.

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from rflib.cnn import ConvModule, xavier_init, normal_init
from rflib.runner import force_fp32
from rfvision.core import multiclass_nms_with_coef, distance2bbox, multi_apply
from rfvision.models.builder import HEADS, build_loss
from .base_dense_head import BaseDenseHead
from .fcos_head import FCOSHead, INF

_EPSILON = 1e-6


@HEADS.register_module()
class USDSegYOLOV3Head(BaseDenseHead):
    """YOLOV3Head Paper link: https://arxiv.org/abs/1804.02767.

    Args:
        num_classes (int): The number of object classes (w/o background)
        num_scales (int): The number of scales / stages.
        num_anchors_per_scale (int): The number of anchors per scale.
            The official implementation uses 3.
        in_channels (List[int]): Number of input channels per scale.
        out_channels (List[int]): The number of output channels per scale
            before the final 1x1 layer.
        strides (List[int]): The stride of each scale.
            Should be in descending order.
        anchor_base_sizes (List[List[int]]): The sizes of anchors.
            The official implementation uses
                [[(116, 90), (156, 198), (373, 326)],
                [( 30, 61), ( 62,  45), ( 59, 119)],
                [( 10, 13), ( 16,  30), ( 33,  23)]]
        ignore_thresh (float): Set negative samples if gt-anchor iou
            is smaller than ignore_thresh. Default: 0.5
        one_hot_smoother (float): Set a non-zero value to enable label-smooth
            Default: 0.
        xy_use_logit (bool): Use log scale regression for bbox center
            Default: False
        balance_conf (bool): Whether to balance the confidence when calculating
            loss. Default: False
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Default: dict(type='BN', requires_grad=True)
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='LeakyReLU', negative_slope=0.1).
        train_cfg (dict): Training config of YOLOV3 head. Default: None.
        test_cfg (dict): Testing config of YOLOV3 head. Default: None.
    """

    def __init__(self,
                 num_classes,
                 num_scales,
                 num_anchors_per_scale,
                 in_channels,
                 out_channels,
                 strides,
                 anchor_base_sizes,
                 ignore_thresh=0.5,
                 one_hot_smoother=0.,
                 xy_use_logit=False,
                 balance_conf=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
                 train_cfg=None,
                 test_cfg=None,
                 num_bases=32,
                 method='None',
                 loss_mask=None,
                 coef_weight=1,
        ):
        super(USDSegYOLOV3Head, self).__init__()
        # Check params
        assert (num_scales == len(in_channels) == len(out_channels) ==
                len(strides) == len(anchor_base_sizes))
        for anchor_base_size in anchor_base_sizes:
            assert (len(anchor_base_size) == num_anchors_per_scale)
            for anchor_size in anchor_base_size:
                assert (len(anchor_size) == 2)

        self.num_classes = num_classes
        self.num_scales = num_scales
        self.num_anchors_per_scale = num_anchors_per_scale
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.strides = strides
        self.anchor_base_sizes = anchor_base_sizes

        self.ignore_thresh = ignore_thresh
        self.one_hot_smoother = one_hot_smoother
        self.xy_use_logit = xy_use_logit
        self.balance_conf = balance_conf

        # self.num_attrib = num_classes + bboxes (4) + objectness (1)
        self.num_attrib = self.num_classes + 5
        self.last_layer_dim = self.num_anchors_per_scale * self.num_attrib

        cfg = dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.convs_bridge = nn.ModuleList()
        self.convs_final = nn.ModuleList()
        for i_scale in range(self.num_scales):
            in_c = self.in_channels[i_scale]
            out_c = self.out_channels[i_scale]
            conv_bridge = ConvModule(in_c, out_c, 3, padding=1, **cfg)
            conv_final = nn.Conv2d(out_c, self.last_layer_dim, 1, bias=True)

            self.convs_bridge.append(conv_bridge)
            self.convs_final.append(conv_final)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # USD-Seg
        self.num_bases = num_bases
        # self.use_dcn=use_dcn  # TODO: Add DCN support
        if method not in ['cosine', 'cosine_r']:
            raise NotImplementedError('%s not supported.' % method)
        self.method = method

        self.last_layer_dim_usd = self.num_anchors_per_scale * self.num_bases
        self.convs_bridge_usd = nn.ModuleList()
        self.convs_final_usd = nn.ModuleList()
        for i_scale in range(self.num_scales):
            in_c = self.in_channels[i_scale]
            out_c = self.out_channels[i_scale]
            conv_bridge = ConvModule(in_c, out_c, 3, padding=1, **cfg)
            conv_final = nn.Conv2d(out_c, self.last_layer_dim_usd, 1, bias=True)

            self.convs_bridge_usd.append(conv_bridge)
            self.convs_final_usd.append(conv_final)
        self.coef_weight = coef_weight
        # Mask Loss
        if loss_mask is None:
            self.loss_mask = None
        else:
            self.loss_mask = build_loss(loss_mask)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, feats):
        assert len(feats) == self.num_scales
        results = []
        for i in range(self.num_scales):
            x = feats[i]
            x_det = self.convs_bridge[i](x)
            out_det = self.convs_final[i](x_det)

            x_usd = self.convs_bridge_usd[i](x)
            out_usd = self.convs_final_usd[i](x_usd)

            out = torch.cat((out_det, out_usd), 1)

            results.append(out)

        return tuple(results),

    @force_fp32(apply_to=('results_raw', ))
    def get_bboxes(self, results_raw, img_metas, cfg=None, rescale=False):
        """Transform network output for a batch into bbox predictions.

        Args:
            results_raw (list[Tensor]): Raw predictions for a batch of images.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            cfg (rflib.Config): Test / postprocessing configuration,
                if None, test_cfg would be used
            rescale (bool): If True, return boxes in original image space

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1. The second item is a
                (n,) tensor where each item is the predicted class label of the
                corresponding box.
        """
        result_list = []
        for img_id in range(len(img_metas)):
            result_raw_list = [
                results_raw[i][img_id].detach() for i in range(self.num_scales)
            ]
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self.get_bboxes_single(result_raw_list, scale_factor,
                                               cfg, rescale)
            result_list.append(proposals)
        return result_list

    @staticmethod
    def _get_anchors_grid_xy(num_grid_h, num_grid_w, stride, device='cpu'):
        """Get grid offset according to the stride.

        Args:
            num_grid_h (int): The height of the grid.
            num_grid_w (int): The width of the grid.
            stride (int): The stride.
            device (torch.device): The desired device of the generated grid.

        Returns:
            tuple[torch.Tensor]: x and y grid offset according to the stride
                in shape (1, num_grid_h, num_grid_w)
        """
        grid_x = torch.arange(
            num_grid_w, dtype=torch.float,
            device=device).repeat(num_grid_h, 1)
        grid_y = torch.arange(
            num_grid_h, dtype=torch.float,
            device=device).repeat(num_grid_w, 1)

        grid_x = grid_x.unsqueeze(0) * stride
        grid_y = grid_y.t().unsqueeze(0) * stride

        return grid_x, grid_y

    def get_bboxes_single(self, results_raw, scale_factor, cfg, rescale=False):
        """Transform outputs for a single batch item into bbox predictions.

        Args:
            results_raw (list[Tensor]): Raw predictions for different scales
                of each single image in the batch.
            scale_factor (ndarray): Scale factor of the image arrange as
                (w_scale, h_scale, w_scale, h_scale).
            cfg (rflib.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
        Returns:
            Tensor: Labeled boxes in shape (n, 5), where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1.
        """
        cfg = self.test_cfg if cfg is None else cfg
        assert len(results_raw) == self.num_scales
        multi_lvl_bboxes = []
        multi_lvl_coefs = []
        multi_lvl_cls_scores = []
        multi_lvl_conf_scores = []
        for i_scale in range(self.num_scales):
            # get some key info for current scale
            result_raw = results_raw[i_scale]
            num_grid_h = result_raw.size(1)
            num_grid_w = result_raw.size(2)
            stride = self.strides[i_scale]

            # for each sacle, reshape to (n_anchors, h, w, n_classes+5+n_bases)
            prediction_raw = result_raw.view(self.num_anchors_per_scale,
                                             self.num_attrib + self.num_bases,
                                             num_grid_h, num_grid_w).permute(
                                                 0, 2, 3, 1).contiguous()

            # grid x y offset, with stride step included
            grid_x, grid_y = self._get_anchors_grid_xy(num_grid_h, num_grid_w,
                                                       stride,
                                                       result_raw.device)

            # Get outputs x, y
            x_center_pred = torch.sigmoid(
                prediction_raw[..., 0]) * stride + grid_x  # Center x
            y_center_pred = torch.sigmoid(
                prediction_raw[..., 1]) * stride + grid_y  # Center y

            anchors = torch.tensor(
                self.anchor_base_sizes[i_scale],
                device=result_raw.device,
                dtype=torch.float32)

            anchor_w = anchors[:, 0:1].view((-1, 1, 1))
            anchor_h = anchors[:, 1:2].view((-1, 1, 1))

            w_pred = torch.exp(prediction_raw[..., 2]) * anchor_w  # Width
            h_pred = torch.exp(prediction_raw[..., 3]) * anchor_h  # Height

            # bbox_pred: convert to x1y1x2y2
            bbox_pred = torch.stack(
                (x_center_pred - w_pred / 2, y_center_pred - h_pred / 2,
                 x_center_pred + w_pred / 2, y_center_pred + h_pred / 2),
                dim=3).view((-1, 4))
            # conf and cls
            conf_pred = torch.sigmoid(prediction_raw[..., 4]).view(-1)
            cls_pred = torch.sigmoid(prediction_raw[..., 5:5+self.num_classes]).view(
                -1, self.num_classes)  # Cls pred one-hot.

            coef_pred = prediction_raw[..., 5+self.num_classes:].view(
                -1, self.num_bases)

            # Filtering out all predictions with conf < conf_thr
            conf_thr = cfg.get('conf_thr', -1)
            conf_inds = conf_pred.ge(conf_thr).nonzero().flatten()
            bbox_pred = bbox_pred[conf_inds, :]
            cls_pred = cls_pred[conf_inds, :]
            conf_pred = conf_pred[conf_inds]
            coef_pred = coef_pred[conf_inds, :]

            # Get top-k prediction
            nms_pre = cfg.get('nms_pre', -1)
            if 0 < nms_pre < conf_pred.size(0):
                _, topk_inds = conf_pred.topk(nms_pre)
                bbox_pred = bbox_pred[topk_inds, :]
                cls_pred = cls_pred[topk_inds, :]
                conf_pred = conf_pred[topk_inds]
                coef_pred = coef_pred[topk_inds, :]

            # Save the result of current scale
            multi_lvl_bboxes.append(bbox_pred)
            multi_lvl_coefs.append(coef_pred)
            multi_lvl_cls_scores.append(cls_pred)
            multi_lvl_conf_scores.append(conf_pred)

        # Merge the results of different scales together
        multi_lvl_bboxes = torch.cat(multi_lvl_bboxes)
        multi_lvl_coefs = torch.cat(multi_lvl_coefs)
        multi_lvl_cls_scores = torch.cat(multi_lvl_cls_scores)
        multi_lvl_conf_scores = torch.cat(multi_lvl_conf_scores)

        if multi_lvl_conf_scores.size(0) == 0:
            return torch.zeros((0, 5)), torch.zeros((0, self.num_bases)), torch.zeros((0, ))

        if rescale:
            multi_lvl_bboxes /= multi_lvl_bboxes.new_tensor(scale_factor)

        # In mmdet 2.x, the class_id for background is num_classes.
        # i.e., the last column.
        padding = multi_lvl_cls_scores.new_zeros(multi_lvl_cls_scores.shape[0],
                                                 1)
        multi_lvl_cls_scores = torch.cat([multi_lvl_cls_scores, padding],
                                         dim=1)

        det_bboxes, det_labels, det_coefs = multiclass_nms_with_coef(
            multi_lvl_bboxes,
            multi_lvl_cls_scores,
            multi_lvl_coefs,
            cfg.score_thr,
            cfg.nms,
            cfg.max_per_img,
            score_factors=multi_lvl_conf_scores,
            num_bases=self.num_bases)
        return det_bboxes, det_labels, det_coefs

    @force_fp32(apply_to=('preds_raw', ))
    def loss(self,
             preds_raw,
             gt_bboxes,
             gt_labels,
             gt_coefs,
             img_metas,
             gt_bboxes_ignore=None,
             bases=None,
             gt_resized_masks=None):
        """Compute loss of the head.

        Args:
            preds_raw (list[Tensor]): Raw predictions for a batch of images.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """

        losses = {'loss_xy': 0, 'loss_wh': 0, 'loss_conf': 0, 'loss_cls': 0, 'loss_coef': 0}

        for img_id in range(len(img_metas)):
            pred_raw_list = []
            anchor_grids = []
            # For each scale of each image:
            for i_scale in range(self.num_scales):
                # 1. get some key information
                pred_raw = preds_raw[i_scale][img_id]
                num_grid_h = pred_raw.size(1)
                num_grid_w = pred_raw.size(2)
                # 2. reshape to (n_anchors, h, w, n_classes+5+n_bases)
                pred_raw = pred_raw.view(self.num_anchors_per_scale,
                                         self.num_attrib+self.num_bases, num_grid_h,
                                         num_grid_w).permute(0, 2, 3,
                                                             1).contiguous()
                # 3. get the grid of the anchors
                anchor_grid = self.get_anchors(
                    num_grid_h, num_grid_w, i_scale, device=pred_raw.device)

                pred_raw_list.append(pred_raw)
                anchor_grids.append(anchor_grid)

            # Then, generate target for each image
            gt_t_across_scale, negative_mask_across_scale = \
                self._preprocess_target_single_img(gt_bboxes[img_id],
                                                   gt_labels[img_id],
                                                   gt_coefs[img_id],
                                                   anchor_grids,
                                                   self.ignore_thresh,
                                                   self.one_hot_smoother,
                                                   self.xy_use_logit)
            # Calculate loss
            losses_per_img = self.loss_single(
                pred_raw_list,
                gt_t_across_scale,
                negative_mask_across_scale,
                xy_use_logit=self.xy_use_logit,
                balance_conf=self.balance_conf)

            for loss_term in losses:
                term_no_loss = loss_term[5:]
                losses[loss_term] += losses_per_img[term_no_loss]

        return losses

    def loss_single(self,
                    preds_raw,
                    gts_t,
                    neg_masks,
                    xy_use_logit=False,
                    balance_conf=False):
        """Compute loss of a single image from a batch.

        Args:
            preds_raw (list[Tensor]): Raw predictions for an image from the
                batch.
            gts_t (list[Tensor]): The Ground-Truth targets across scales.
            neg_masks (list[Tensor]): The negative masks across scales.
            xy_use_logit (bool): Use log scale regression for bbox center
                Default: False
            balance_conf (bool): Whether to balance the confidence when
                calculating loss. Default: False

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """

        losses = {'xy': 0, 'wh': 0, 'conf': 0, 'cls': 0, 'coef': 0}

        for i_scale in range(self.num_scales):
            pred_raw = preds_raw[i_scale]
            gt_t = gts_t[i_scale]
            neg_mask = neg_masks[i_scale].float()
            pos_mask = gt_t[..., 4]
            pos_and_neg_mask = neg_mask + pos_mask
            pos_mask = pos_mask.unsqueeze(dim=-1)
            if torch.max(pos_and_neg_mask) > 1.:
                warnings.warn('There is overlap between pos and neg sample.')
                pos_and_neg_mask = pos_and_neg_mask.clamp(min=0., max=1.)

            pred_t_xy = pred_raw[..., :2]
            pred_t_wh = pred_raw[..., 2:4]
            pred_conf = pred_raw[..., 4]
            pred_label = pred_raw[..., 5:5+self.num_classes]
            pred_coef = pred_raw[..., 5+self.num_classes:]

            gt_t_xy = gt_t[..., :2]
            gt_t_wh = gt_t[..., 2:4]
            gt_conf = gt_t[..., 4]
            gt_label = gt_t[..., 5:5+self.num_classes]
            gt_coef = gt_t[..., 5+self.num_classes:]

            if balance_conf:
                num_pos_gt = max(int(torch.sum(gt_conf)), 1)
                grid_size = list(gt_conf.size())
                num_total_grids = 1
                for s in grid_size:
                    num_total_grids *= s
                pos_weight = num_total_grids / num_pos_gt
                conf_loss_weight = 1 / pos_weight
            else:
                pos_weight = 1
                conf_loss_weight = 1

            pos_weight = gt_label.new_tensor(pos_weight)

            losses_cls = F.binary_cross_entropy_with_logits(
                pred_label, gt_label, reduction='none')

            losses_cls *= pos_mask

            losses_conf = F.binary_cross_entropy_with_logits(
                pred_conf, gt_conf, reduction='none',
                pos_weight=pos_weight) * pos_and_neg_mask * conf_loss_weight

            if xy_use_logit:
                losses_xy = F.mse_loss(
                    pred_t_xy, gt_t_xy, reduction='none') * pos_mask * 2
            else:
                losses_xy = F.binary_cross_entropy_with_logits(
                    pred_t_xy, gt_t_xy, reduction='none') * pos_mask * 2

            losses_wh = F.mse_loss(
                pred_t_wh, gt_t_wh, reduction='none') * pos_mask * 2

            losses_coef = F.cosine_embedding_loss(
                pred_coef.view(-1, self.num_bases),
                gt_coef.view(-1, self.num_bases),
                torch.tensor(1, device=pred_coef.device),
                reduction='none') * pos_mask.view(-1) * self.coef_weight

            losses['cls'] += torch.sum(losses_cls)
            losses['conf'] += torch.sum(losses_conf)
            losses['xy'] += torch.sum(losses_xy)
            losses['wh'] += torch.sum(losses_wh)
            losses['coef'] += torch.sum(losses_coef)

        return losses

    def _preprocess_target_single_img(self,
                                      gt_bboxes,
                                      gt_labels,
                                      gt_coefs,
                                      anchor_grids,
                                      ignore_thresh,
                                      one_hot_smoother=0,
                                      xy_use_logit=False):
        """Generate matching bounding box prior and converted GT."""
        negative_mask_across_scale = []
        gt_t_across_scale = []

        for anchor_grid in anchor_grids:  # len(anchor_grids) == num_scales
            negative_mask_size = list(anchor_grid.size())[:-1]
            negative_mask = anchor_grid.new_ones(
                negative_mask_size, dtype=torch.uint8)
            negative_mask_across_scale.append(negative_mask)
            gt_t_size = negative_mask_size + [self.num_attrib + self.num_bases]
            gt_t = anchor_grid.new_zeros(gt_t_size)
            gt_t_across_scale.append(gt_t)

        for gt_bbox, gt_label, gt_coef in zip(gt_bboxes, gt_labels, gt_coefs):
            # For each gt, convert to cxywh
            gt_cx = (gt_bbox[0] + gt_bbox[2]) / 2
            gt_cy = (gt_bbox[1] + gt_bbox[3]) / 2
            gt_w = gt_bbox[2] - gt_bbox[0]
            gt_h = gt_bbox[3] - gt_bbox[1]
            gt_bbox_cxywh = torch.stack((gt_cx, gt_cy, gt_w, gt_h))

            iou_to_match_across_scale = []
            grid_coord_across_scale = []

            for i_scale in range(self.num_scales):
                # For each sacle of each gt:
                stride = self.strides[i_scale]
                anchor_grid = anchor_grids[i_scale]
                # 1. Calculate the iou between each anchor and the gt
                iou_gt_anchor = iou_multiple_to_one(
                    anchor_grid, gt_bbox_cxywh, center=True)
                # 2. Neg. sample if <= ignore thresh
                negative_mask = (iou_gt_anchor <= ignore_thresh)
                negative_mask_across_scale[i_scale] *= negative_mask
                # 3. Save the coord of the anchor grid that the gt is in
                # in this sacle
                w_grid = int(gt_cx // stride)
                h_grid = int(gt_cy // stride)
                grid_coord_across_scale.append((h_grid, w_grid))

                # AND operation, only negative when all are negative
                iou_to_match = list(iou_gt_anchor[:, h_grid, w_grid])
                iou_to_match_across_scale.extend(iou_to_match)

            # get idx of max iou across all scales & anchors for current gt
            max_match_iou_idx = max(
                range(len(iou_to_match_across_scale)),
                key=lambda x: iou_to_match_across_scale[x])

            # decide which scale / anchor is matched (convert back)
            match_scale = max_match_iou_idx // self.num_anchors_per_scale
            match_anchor_in_scale = max_match_iou_idx - \
                match_scale * self.num_anchors_per_scale
            # extract the matched piror bbox to generate the target
            match_grid_h, match_grid_w = grid_coord_across_scale[match_scale]
            match_anchor_w, match_anchor_h = self.anchor_base_sizes[
                match_scale][match_anchor_in_scale]

            # Generate gt bbox for regression
            gt_tw = torch.log((gt_w / match_anchor_w).clamp(min=_EPSILON))  # w
            gt_th = torch.log((gt_h / match_anchor_h).clamp(min=_EPSILON))  # h
            # bbox center
            gt_tcx = (gt_cx / self.strides[match_scale] - match_grid_w).clamp(
                _EPSILON, 1 - _EPSILON)
            gt_tcy = (gt_cy / self.strides[match_scale] - match_grid_h).clamp(
                _EPSILON, 1 - _EPSILON)
            if xy_use_logit:  # log for bbox center
                gt_tcx = torch.log(gt_tcx /
                                   (1. - gt_tcx))  # inverse of sigmoid
                gt_tcy = torch.log(gt_tcy /
                                   (1. - gt_tcy))  # inverse of sigmoid
            # cat cx, cy, w, h together
            gt_t_bbox = torch.stack((gt_tcx, gt_tcy, gt_tw, gt_th))

            # In mmdet 2.x, label “K” means background, and labels
            # [0, K-1] correspond to the K = num_categories object categories.
            gt_label_one_hot = F.one_hot(
                gt_label, num_classes=self.num_classes).float()

            if one_hot_smoother != 0:  # label smooth
                gt_label_one_hot = gt_label_one_hot * (
                    1 - one_hot_smoother) + one_hot_smoother / self.num_classes

            gt_t_across_scale[match_scale][match_anchor_in_scale, match_grid_h,
                                           match_grid_w, :4] = gt_t_bbox
            gt_t_across_scale[match_scale][match_anchor_in_scale, match_grid_h,
                                           match_grid_w, 4] = 1.
            gt_t_across_scale[match_scale][match_anchor_in_scale, match_grid_h,
                                           match_grid_w, 5:5+self.num_classes] = gt_label_one_hot
            gt_t_across_scale[match_scale][match_anchor_in_scale, match_grid_h,
                                           match_grid_w, 5+self.num_classes:] = gt_coef

            # although iou fall under a certain thres,
            # since it has max iou, still positive
            negative_mask_across_scale[match_scale][match_anchor_in_scale,
                                                    match_grid_h,
                                                    match_grid_w] = 0

        return gt_t_across_scale, negative_mask_across_scale

    def get_anchors(self, num_grid_h, num_grid_w, scale, device='cpu'):
        """Get the grid offset according to the anchors and the scale.

        Args:
            num_grid_h (int): The height of the grid.
            num_grid_w (int): The width of the grid.
            scale (int): The index of scale.
            device (torch.device): The desired device of the generated grid.

        Returns:
            torch.Tensor: The anchors in cxcywh format in shape
                (num_anchors, num_grid_h, num_grid_w, 4)
        """
        assert scale in range(self.num_scales)
        anchors = torch.tensor(
            self.anchor_base_sizes[scale], device=device, dtype=torch.float32)
        num_anchors = anchors.size(0)
        stride = self.strides[scale]

        grid_x, grid_y = self._get_anchors_grid_xy(num_grid_h, num_grid_w,
                                                   stride, device)

        grid_x += stride / 2  # convert to center of the grid,
        grid_y += stride / 2  # that is, making the raw prediction 0, not -inf
        grid_x = grid_x.expand((num_anchors, -1, -1))
        grid_y = grid_y.expand((num_anchors, -1, -1))

        anchor_w = anchors[:, 0:1].view((-1, 1, 1))
        anchor_h = anchors[:, 1:2].view((-1, 1, 1))
        anchor_w = anchor_w.expand((-1, num_grid_h, num_grid_w))
        anchor_h = anchor_h.expand((-1, num_grid_h, num_grid_w))

        anchor_cxywh = torch.stack((grid_x, grid_y, anchor_w, anchor_h), dim=3)

        return anchor_cxywh


def iou_multiple_to_one(bboxes1, bbox2, center=False, zero_center=False):
    """Calculate the IOUs between bboxes1 (multiple) and bbox2 (one).

    Args:
        bboxes1: (Tensor) A n-D tensor representing first group of bboxes.
            The dimension is (..., 4).
            The last dimension represent the bbox, with coordinate (x, y, w, h)
            or (cx, cy, w, h).
        bbox2: (Tensor) A 1D tensor representing the second bbox.
            The dimension is (4,).
        center: (bool). Whether the bboxes are in format (cx, cy, w, h).
        zero_center: (bool). Whether to align two bboxes so their center
            is aligned.

    Returns:
        iou_: (Tensor) A (n-1)-D tensor representing the IOUs.
            It has one less dim than bboxes1
    """

    epsilon = 1e-6

    x1 = bboxes1[..., 0]
    y1 = bboxes1[..., 1]
    w1 = bboxes1[..., 2]
    h1 = bboxes1[..., 3]

    x2 = bbox2[0]
    y2 = bbox2[1]
    w2 = bbox2[2]
    h2 = bbox2[3]

    area1 = w1 * h1
    area2 = w2 * h2

    if zero_center:
        w_intersect = torch.min(w1, w2).clamp(min=0)
        h_intersect = torch.min(h1, h2).clamp(min=0)
    else:
        if center:
            x1 = x1 - w1 / 2
            y1 = y1 - h1 / 2
            x2 = x2 - w2 / 2
            y2 = y2 - h2 / 2
        right1 = (x1 + w1)
        right2 = (x2 + w2)
        top1 = (y1 + h1)
        top2 = (y2 + h2)
        left1 = x1
        left2 = x2
        bottom1 = y1
        bottom2 = y2
        w_intersect = (torch.min(right1, right2) -
                       torch.max(left1, left2)).clamp(min=0)
        h_intersect = (torch.min(top1, top2) -
                       torch.max(bottom1, bottom2)).clamp(min=0)
    area_intersect = h_intersect * w_intersect

    iou_ = area_intersect / (area1 + area2 - area_intersect + epsilon)

    return iou_


@HEADS.register_module()
class USDSegFCOSHead(FCOSHead):
    def __init__(self,
                 num_classes,
                 in_channels,
                 regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512),
                                 (512, INF)),
                 center_sampling=False,
                 center_sample_radius=1.5,
                 norm_on_bbox=False,
                 centerness_on_reg=False,
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox=dict(type='IoULoss', loss_weight=1.0),
                 loss_centerness=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 # <-- USD-Seg begin
                 loss_coef=dict(
                     type='SmoothL1Loss',
                     beta=1.0,
                     reduction='mean',
                     loss_weight=1.0),
                 num_bases=-1,
                 method='None',
                # USD-Seg End -->
                 **kwargs):
        self.num_bases = num_bases
        super(USDSegFCOSHead, self).__init__(
            num_classes,
            in_channels,
            regress_ranges,
            center_sampling,
            center_sample_radius,
            norm_on_bbox,
            centerness_on_reg,
            loss_cls,
            loss_bbox,
            loss_centerness,
            norm_cfg,
            **kwargs
        )

        # self.use_dcn=use_dcn  # TODO: Add DCN support
        if method not in ['cosine']:
            raise NotImplementedError('%s not supported.' % method)
        self.method = method
        self.loss_coef = build_loss(loss_coef)

    def _init_layers(self):
        """Initialize layers of the head."""
        super()._init_layers()
        self._init_coef_convs()

    def _init_coef_convs(self):
        """Initialize coef regression conv layers of the head."""
        self.coef_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            if self.dcn_on_last_conv and i == self.stacked_convs - 1:
                conv_cfg = dict(type='DCNv2')
            else:
                conv_cfg = self.conv_cfg
            self.coef_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.conv_bias))
    
    def _init_predictor(self):
        """Initialize predictor layers of the head."""
        self.conv_cls = nn.Conv2d(
            self.feat_channels, self.cls_out_channels, 3, padding=1)
        self.conv_reg = nn.Conv2d(self.feat_channels, 4, 3, padding=1)
        self.conv_coef = nn.Conv2d(self.feat_channels, self.num_bases, 3, padding=1)

    def init_weights(self):
        """Initialize weights of the head."""
        super().init_weights()
        for m in self.coef_convs:
            if isinstance(m.conv, nn.Conv2d):
                normal_init(m.conv, std=0.01)
        normal_init(self.conv_coef, std=0.01)

    def forward_single(self, x, scale, stride):
        cls_score, bbox_pred, centerness = super().forward_single(x, scale, stride)

        coef_feat = x
        for coef_layer in self.coef_convs:
            coef_feat = coef_layer(coef_feat)
        coef_pred = self.conv_coef(coef_feat)

        return cls_score, bbox_pred, centerness, coef_pred

    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_coefs,
                      gt_bboxes_ignore=None,
                      ):
        outs = self(x)
        loss_inputs = outs + (gt_bboxes, gt_labels, gt_coefs, img_metas)
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

    def get_targets(self, points, gt_bboxes_list, gt_coefs_list, gt_labels_list):
        """Compute regression, classification and centerss targets for points
        in multiple images.

        Args:
            points (list[Tensor]): Points of each fpn level, each has shape
                (num_points, 2).
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image,
                each has shape (num_gt, 4).
            gt_labels_list (list[Tensor]): Ground truth labels of each box,
                each has shape (num_gt,).

        Returns:
            tuple:
                concat_lvl_labels (list[Tensor]): Labels of each level. \
                concat_lvl_bbox_targets (list[Tensor]): BBox targets of each \
                    level.
        """
        assert len(points) == len(self.regress_ranges)
        num_levels = len(points)
        # expand regress ranges to align with points
        expanded_regress_ranges = [
            points[i].new_tensor(self.regress_ranges[i])[None].expand_as(
                points[i]) for i in range(num_levels)
        ]
        # concat all levels points and regress ranges
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(points, dim=0)

        # the number of points per img, per lvl
        num_points = [center.size(0) for center in points]

        # get labels and bbox_targets of each image
        labels_list, bbox_targets_list, coef_targets_list = multi_apply(
            self._get_target_single,
            gt_bboxes_list,
            gt_coefs_list,
            gt_labels_list,
            points=concat_points,
            regress_ranges=concat_regress_ranges,
            num_points_per_lvl=num_points)

        # split to per img, per level
        labels_list = [labels.split(num_points, 0) for labels in labels_list]
        bbox_targets_list = [
            bbox_targets.split(num_points, 0)
            for bbox_targets in bbox_targets_list
        ]

        coef_targets_list = [
            coef_targets.split(num_points, 0)
            for coef_targets in coef_targets_list
        ]

        # concat per level image
        concat_lvl_labels = []
        concat_lvl_bbox_targets = []
        concat_lvl_coef_targets = []
        for i in range(num_levels):
            concat_lvl_labels.append(
                torch.cat([labels[i] for labels in labels_list]))

            bbox_targets = torch.cat(
                [bbox_targets[i] for bbox_targets in bbox_targets_list])
            if self.norm_on_bbox:
                bbox_targets = bbox_targets / self.strides[i]
            concat_lvl_bbox_targets.append(bbox_targets)

            concat_lvl_coef_targets.append(
                torch.cat(
                    [coef_targets[i] for coef_targets in coef_targets_list]))
        return concat_lvl_labels, concat_lvl_bbox_targets, concat_lvl_coef_targets

    def _get_target_single(self, gt_bboxes, gt_coefs, gt_labels, points, regress_ranges,
                           num_points_per_lvl):
        """Compute regression and classification targets for a single image."""
        num_points = points.size(0)
        num_gts = gt_labels.size(0)
        if num_gts == 0:
            return gt_labels.new_full((num_points,), self.num_classes), \
                   gt_bboxes.new_zeros((num_points, 4))

        areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (
            gt_bboxes[:, 3] - gt_bboxes[:, 1])
        # TODO: figure out why these two are different
        # areas = areas[None].expand(num_points, num_gts)
        areas = areas[None].repeat(num_points, 1)
        regress_ranges = regress_ranges[:, None, :].expand(
            num_points, num_gts, 2)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)
        xs, ys = points[:, 0], points[:, 1]
        xs = xs[:, None].expand(num_points, num_gts)
        ys = ys[:, None].expand(num_points, num_gts)

        left = xs - gt_bboxes[..., 0]
        right = gt_bboxes[..., 2] - xs
        top = ys - gt_bboxes[..., 1]
        bottom = gt_bboxes[..., 3] - ys
        bbox_targets = torch.stack((left, top, right, bottom), -1)

        if self.center_sampling:
            # condition1: inside a `center bbox`
            radius = self.center_sample_radius
            center_xs = (gt_bboxes[..., 0] + gt_bboxes[..., 2]) / 2
            center_ys = (gt_bboxes[..., 1] + gt_bboxes[..., 3]) / 2
            center_gts = torch.zeros_like(gt_bboxes)
            stride = center_xs.new_zeros(center_xs.shape)

            # project the points on current lvl back to the `original` sizes
            lvl_begin = 0
            for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl):
                lvl_end = lvl_begin + num_points_lvl
                stride[lvl_begin:lvl_end] = self.strides[lvl_idx] * radius
                lvl_begin = lvl_end

            x_mins = center_xs - stride
            y_mins = center_ys - stride
            x_maxs = center_xs + stride
            y_maxs = center_ys + stride
            center_gts[..., 0] = torch.where(x_mins > gt_bboxes[..., 0],
                                             x_mins, gt_bboxes[..., 0])
            center_gts[..., 1] = torch.where(y_mins > gt_bboxes[..., 1],
                                             y_mins, gt_bboxes[..., 1])
            center_gts[..., 2] = torch.where(x_maxs > gt_bboxes[..., 2],
                                             gt_bboxes[..., 2], x_maxs)
            center_gts[..., 3] = torch.where(y_maxs > gt_bboxes[..., 3],
                                             gt_bboxes[..., 3], y_maxs)

            cb_dist_left = xs - center_gts[..., 0]
            cb_dist_right = center_gts[..., 2] - xs
            cb_dist_top = ys - center_gts[..., 1]
            cb_dist_bottom = center_gts[..., 3] - ys
            center_bbox = torch.stack(
                (cb_dist_left, cb_dist_top, cb_dist_right, cb_dist_bottom), -1)
            inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0
        else:
            # condition1: inside a gt bbox
            inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0

        # condition2: limit the regression range for each location
        max_regress_distance = bbox_targets.max(-1)[0]
        inside_regress_range = (
            (max_regress_distance >= regress_ranges[..., 0])
            & (max_regress_distance <= regress_ranges[..., 1]))

        # if there are still more than one objects for a location,
        # we choose the one with minimal area
        areas[inside_gt_bbox_mask == 0] = INF
        areas[inside_regress_range == 0] = INF
        min_area, min_area_inds = areas.min(dim=1)

        labels = gt_labels[min_area_inds]
        labels[min_area == INF] = self.num_classes  # set as BG
        bbox_targets = bbox_targets[range(num_points), min_area_inds]

        # USD-Seg
        pos_inds = torch.where(labels != self.num_classes)
        coef_targets = torch.zeros((num_points, self.num_bases), device=gt_coefs.device)
        
        # for p in pos_inds:
        #     pos_coef_id = min_area_inds[p]
        #     pos_coef = gt_coefs[pos_coef_id]
        #     coef_targets[p] = pos_coef
        # Vectorization
        coef_targets[pos_inds] = gt_coefs[min_area_inds[pos_inds]].float()

        return labels, bbox_targets, coef_targets

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses', 'coef_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             centernesses,
             coef_preds,
             gt_bboxes,
             gt_labels,
             gt_coefs,
             img_metas,
             gt_bboxes_ignore=None):
        assert len(cls_scores) == len(bbox_preds) == len(centernesses) == len(coef_preds)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                           bbox_preds[0].device)
        labels, bbox_targets, coef_targets = self.get_targets(all_level_points, gt_bboxes,
                                                gt_coefs, gt_labels)

        num_imgs = cls_scores[0].size(0)
        # flatten cls_scores, bbox_preds and centerness
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_coef_preds = [
            coef_pred.permute(0, 2, 3, 1).reshape(-1, self.num_bases)
            for coef_pred in coef_preds
        ]
        flatten_centerness = [
            centerness.permute(0, 2, 3, 1).reshape(-1)
            for centerness in centernesses
        ]
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_coef_preds = torch.cat(flatten_coef_preds)  # [num_pixel, num_bases]
        flatten_centerness = torch.cat(flatten_centerness)
        flatten_labels = torch.cat(labels)
        flatten_bbox_targets = torch.cat(bbox_targets)
        flatten_coef_targets = torch.cat(coef_targets)      # [num_pixel, num_bases]

        # repeat points to align with bbox_preds
        flatten_points = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_level_points])

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((flatten_labels >= 0)
                    & (flatten_labels < bg_class_ind)).nonzero().reshape(-1)
        num_pos = len(pos_inds)
        loss_cls = self.loss_cls(
            flatten_cls_scores, flatten_labels,
            avg_factor=num_pos + num_imgs)  # avoid num_pos is 0

        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_coef_preds = flatten_coef_preds[pos_inds]
        pos_centerness = flatten_centerness[pos_inds]

        if num_pos > 0:
            pos_bbox_targets = flatten_bbox_targets[pos_inds]
            pos_coef_targets = flatten_coef_targets[pos_inds]
            pos_centerness_targets = self.centerness_target(pos_bbox_targets)
            pos_points = flatten_points[pos_inds]
            pos_decoded_bbox_preds = distance2bbox(pos_points, pos_bbox_preds)
            pos_decoded_target_preds = distance2bbox(pos_points,
                                                     pos_bbox_targets)
            # centerness weighted iou loss
            loss_bbox = self.loss_bbox(
                pos_decoded_bbox_preds,
                pos_decoded_target_preds,
                weight=pos_centerness_targets,
                avg_factor=pos_centerness_targets.sum())
            loss_centerness = self.loss_centerness(pos_centerness,
                                                   pos_centerness_targets)

            if self.method == 'var':
                raise RuntimeError('Method var is deprecated.')
            elif self.method == 'cosine' or self.method == 'cosine_r':
                loss_coef = self.loss_coef(pos_coef_preds,
                                           pos_coef_targets,
                                           weight=pos_centerness_targets,
                                           avg_factor=pos_centerness_targets.sum())
            else:
                raise RuntimeError('Method %s is not supported.' % self.method)

        else:
            loss_bbox = pos_bbox_preds.sum()
            loss_coef = pos_coef_preds.sum()
            loss_centerness = pos_centerness.sum()

        return dict(
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_centerness=loss_centerness,
            loss_coef=loss_coef)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses', 'coef_preds'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   centernesses,
                   coef_preds,
                   img_metas,
                   cfg=None,
                   rescale=False,
                   with_nms=True):
        """Transform network output for a batch into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                with shape (N, num_points * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_points * 4, H, W).
            centernesses (list[Tensor]): Centerness for each scale level with
                shape (N, num_points * 1, H, W).
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            cfg (robotflow.rflib.Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used. Default: None.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1. The second item is a
                (n,) tensor where each item is the predicted class label of the
                corresponding box.
        """
        assert len(cls_scores) == len(bbox_preds) == len(coef_preds)
        num_levels = len(cls_scores)

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        mlvl_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                      bbox_preds[0].device)
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            centerness_pred_list = [
                centernesses[i][img_id].detach() for i in range(num_levels)
            ]
            coef_pred_list = [
                coef_preds[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            det_bboxes = self._get_bboxes_single(
                cls_score_list, bbox_pred_list, coef_pred_list, centerness_pred_list,
                mlvl_points, img_shape, scale_factor, cfg, rescale, with_nms)
            result_list.append(det_bboxes)
        return result_list

    def _get_bboxes_single(self,
                           cls_scores,
                           bbox_preds,
                           coef_preds,
                           centernesses,
                           mlvl_points,
                           img_shape,
                           scale_factor,
                           cfg,
                           rescale=False,
                           with_nms=True):
        """Transform outputs for a single batch item into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for a single scale level
                with shape (num_points * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for a single scale
                level with shape (num_points * 4, H, W).
            centernesses (list[Tensor]): Centerness for a single scale level
                with shape (num_points * 4, H, W).
            mlvl_points (list[Tensor]): Box reference for a single scale level
                with shape (num_total_points, 4).
            img_shape (tuple[int]): Shape of the input image,
                (height, width, 3).
            scale_factor (ndarray): Scale factor of the image arrange as
                (w_scale, h_scale, w_scale, h_scale).
            cfg (robotflow.rflib.Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            tuple(Tensor):
                det_bboxes (Tensor): BBox predictions in shape (n, 5), where
                    the first 4 columns are bounding box positions
                    (tl_x, tl_y, br_x, br_y) and the 5-th column is a score
                    between 0 and 1.
                det_labels (Tensor): A (n,) tensor where each item is the
                    predicted class label of the corresponding box.
        """
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_points) == len(coef_preds)
        mlvl_bboxes = []
        mlvl_coefs = []
        mlvl_scores = []
        mlvl_centerness = []
        for cls_score, bbox_pred, coef_pred, centerness, points in zip(
                cls_scores, bbox_preds, coef_preds, centernesses, mlvl_points):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:] == coef_pred.size()[-2:]
            scores = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels).sigmoid()
            centerness = centerness.permute(1, 2, 0).reshape(-1).sigmoid()

            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            coef_pred = coef_pred.permute(1, 2, 0).reshape(-1, self.num_bases)
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, _ = (scores * centerness[:, None]).max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                coef_pred = coef_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                centerness = centerness[topk_inds]
            bboxes = distance2bbox(points, bbox_pred, max_shape=img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_centerness.append(centerness)
            mlvl_coefs.append(coef_pred)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        mlvl_coefs = torch.cat(mlvl_coefs)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        # remind that we set FG labels to [0, num_class-1] since robotflow v2.0
        # BG cat_id: num_class
        mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)
        mlvl_centerness = torch.cat(mlvl_centerness)

        if with_nms:
            det_bboxes, det_labels, det_coefs = multiclass_nms_with_coef(
                mlvl_bboxes,
                mlvl_scores,
                mlvl_coefs,
                cfg.score_thr,
                cfg.nms,
                cfg.max_per_img,
                score_factors=mlvl_centerness,
                num_bases=self.num_bases)
            return det_bboxes, det_labels, det_coefs
        else:
            raise RuntimeWarning('Warning: get_bbox_single with use_nms==False should be carefully used!')
            return mlvl_bboxes, mlvl_scores, mlvl_centerness, mlvl_coefs
