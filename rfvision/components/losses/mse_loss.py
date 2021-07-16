import torch.nn as nn
import torch.nn.functional as F

from rfvision.models.builder import LOSSES
from .utils import weighted_loss


@LOSSES.register_module()
class HeatmapMSELoss(nn.Module):
    def __init__(self, loss_weight=1.):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, hm_pred, hm_gt, hm_weight=None):
        b, c = hm_pred.shape[:2]
        hm_pred = hm_pred.reshape(b, c, -1).split(1, 1)
        hm_gt = hm_gt.reshape(b, c, -1).split(1, 1)

        loss = 0.
        if hm_weight is not None:
            hm_pred, hm_gt = hm_pred * hm_weight, hm_gt * hm_weight
            for i in range(c):
                loss += F.mse_loss(hm_pred[i], hm_gt[i])
        else:
            for i in range(c):
                loss += F.mse_loss(hm_pred[i], hm_gt[i])
        return loss / c * self.loss_weight


@weighted_loss
def mse_loss(pred, target):
    """Warpper of mse loss."""
    return F.mse_loss(pred, target, reduction='none')


@LOSSES.register_module()
class MSELoss(nn.Module):
    """MSELoss.

    Args:
        reduction (str, optional): The method that reduces the loss to a
            scalar. Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of the loss. Defaults to 1.0
    """

    def __init__(self, reduction='mean', loss_weight=1.0):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None):
        """Forward function of loss.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): Weight of the loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.

        Returns:
            torch.Tensor: The calculated loss
        """
        loss = self.loss_weight * mse_loss(
            pred,
            target,
            weight,
            reduction=self.reduction,
            avg_factor=avg_factor)
        return loss
