from .accuracy import Accuracy, accuracy
from .cross_entropy_loss import (CrossEntropyLoss, binary_cross_entropy,
                                 cross_entropy, mask_cross_entropy)
from .focal_loss import FocalLoss, sigmoid_focal_loss
from .ghm_loss import GHMC, GHMR
from .iou_loss import (BoundedIoULoss, CIoULoss, DIoULoss, GIoULoss, IoULoss,
                       bounded_iou_loss, iou_loss)
from .mse_loss import MSELoss, mse_loss, JointsMSELoss
from .smooth_l1_loss import L1Loss, SmoothL1Loss, l1_loss, smooth_l1_loss
from .utils import reduce_loss, weight_reduce_loss, weighted_loss

from .arti_loss import ArtiNOCSLoss, ArtiMIoULoss, ArtiVECTLoss
from .cosine_simlarity_loss import CosineSimilarityLoss, cosine_similarity_loss
from .axis_aligned_iou_loss import AxisAlignedIoULoss, axis_aligned_iou_loss
from .chamfer_distance import ChamferDistance, chamfer_distance
from rfvision.components.losses_pose.regression_loss import L1LossPose
from .dice_loss import DiceLoss

# TODO: remove FocalLossSOLO
from .sigmoid_focal_loss_solo import FocalLossSOLO

__all__ = [
    'accuracy', 'Accuracy', 'cross_entropy', 'binary_cross_entropy',
    'mask_cross_entropy', 'CrossEntropyLoss', 'sigmoid_focal_loss',
    'FocalLoss', 'smooth_l1_loss', 'SmoothL1Loss', 'mse_loss', 'MSELoss', 'iou_loss', 'bounded_iou_loss',
    'IoULoss', 'BoundedIoULoss', 'GIoULoss', 'DIoULoss', 'CIoULoss', 'GHMC',
    'GHMR', 'reduce_loss', 'weight_reduce_loss', 'weighted_loss', 'L1Loss',
    'l1_loss', 'JointsMSELoss',
    'ArtiVECTLoss', 'ArtiNOCSLoss', 'ArtiMIoULoss',
    'CosineSimilarityLoss', 'cosine_similarity_loss', 'ChamferDistance',
    'chamfer_distance', 'axis_aligned_iou_loss', 'AxisAlignedIoULoss',
    'L1LossPose', 'DiceLoss', 'FocalLossSOLO'
]
