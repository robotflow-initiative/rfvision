import torch.nn.functional as F
from rfvision.models.builder import LOSSES
import torch

@LOSSES.register_module()
class HandTailorLoss:
    def __init__(self):
        pass

    def loss2d(self, feature, heatmap, heatmap_weight=None,):
        batch_size = feature.size(0)
        num_joints = feature.size(1)
        heatmaps_pred = feature.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = heatmap.reshape((batch_size, num_joints, -1)).split(1, 1)

        loss = 0.
        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze(1)
            heatmap_gt = heatmaps_gt[idx].squeeze(1)
            if heatmap_weight is not None:
                loss += F.mse_loss(heatmap_pred.mul(heatmap_weight[:, idx]),
                                   heatmap_gt.mul(heatmap_weight[:, idx]))
            else:
                loss += F.mse_loss(heatmap_pred, heatmap_gt)
        #loss = loss / num_joints
        return loss

    def loss3d(self, pred_joints_uvd, gt_joints_uvd):
        num_joints = pred_joints_uvd.size(1)
        loss = 0.
        for idx in range(num_joints):
            loss += F.mse_loss(pred_joints_uvd.float() * 1000,
                                 gt_joints_uvd.float() * 1000)
        return loss

    def loss_so3(self, pred_so3, gt_so3):
        loss = F.mse_loss(pred_so3, gt_so3)
        return loss

    def loss_quat(self, pred_quat, gt_quat):
        loss = F.mse_loss(pred_quat, gt_quat)
        return loss
