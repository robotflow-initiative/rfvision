from rfvision.models.builder import DETECTORS, build_backbone, build_loss, build_detector
from rfvision.models import BaseDetector
from rfvision.components.utils.handtailor_utils import hm_to_uvd
from rflib.cnn import kaiming_init, constant_init
from rflib.runner import load_checkpoint
import logging
import torch

@DETECTORS.register_module()
class HandTailor(BaseDetector):
    '''
    The handtailor has a multi-stage training process:
    stage 0 (A independent stage) : train iknet independently.
    stage 1 - forward_train_2d : train (backbone2d + head2d)
    stage 2 - forward_train_3d : train (backbone2d + head2d trained in forward_train_2d) + (backbone3d + head3d)

    stage 3 - forward_train_mano : train (backbone2d + head2d trained in forward_train_3d) +
                                         (backbone3d + head3d trained in forward_train_3d) +
                                         iknet (trained in stage0) + manonet
    Therefore, the function 'forward' is the combination of stage 0 ~ stage 3.
    '''



    def __init__(self,
                 manonet,
                 iknet,
                 loss,
                 backbone2d,
                 backbone3d,
                 epoch_2d=100,
                 epoch_3d=100,
                 epoch_mano=100,
                 save_model_seperately = True,
                 **kwargs,
                 ):
        super().__init__()
        self.loss = build_loss(loss)
        self.backbone2d = build_backbone(backbone2d)
        self.backbone3d = build_backbone(backbone3d)
        self.manonet = build_backbone(manonet)
        self.iknet = build_detector(iknet)

        self.epoch_2d = epoch_2d
        self.epoch_3d = epoch_3d
        self.epoch_mano = epoch_mano
        self.save_model_seperately = save_model_seperately

    def forward_train_2d(self, **kwargs):
        # release
        img = kwargs['img']
        heatmap = kwargs['heatmap']
        # heatmap_weight = kwargs['heatmap_weight']

        heatmap_list, feature_list = self.backbone2d(img)
        out_headmap_2d = heatmap_list[-1]
        out_feature_2d = feature_list[-1]

        pred_dict_2d = {'out_features_2d': out_feature_2d}
        loss2d = self.loss.loss2d(out_headmap_2d, heatmap, None)
        losses = {'loss2d': loss2d}
        return losses, pred_dict_2d

    def forward_train_3d(self, **kwargs):
        gt_joints_uvd = kwargs['joints_uvd']

        losses, pred_dict_2d = self.forward_train_2d(**kwargs)

        heatmap_list, feature_list = self.backbone3d(pred_dict_2d['out_features_2d'])
        out_heatmap_3d = heatmap_list[-1]
        out_feature_3d = feature_list[-1]

        pred_joints_uvd = hm_to_uvd(out_heatmap_3d)
        pred_dict_3d = {'out_features_3d': out_feature_3d,
                        'out_heatmap_3d': out_heatmap_3d[:, :21, ...],
                        'pred_joints_uvd': pred_joints_uvd,
                        }

        loss3d = self.loss.loss3d(pred_joints_uvd, gt_joints_uvd)
        losses['loss3d'] = loss3d
        return losses, pred_dict_3d

    def forward_train_mano(self, **kwargs):
        K = kwargs['K']
        # train
        losses, pred_dict_3d = self.forward_train_3d(**kwargs)
        pred_dict_mano = self.manonet(pred_dict_3d, K)
        pred_so3, pred_quat = self.iknet.forward_test(pred_dict_mano['joints_xyz_mano'])

        # compute loss
        loss_so3 = self.loss.loss_so3(pred_so3, torch.zeros_like(pred_so3))
        loss_quat = self.loss.loss_quat(pred_quat, torch.zeros_like(pred_quat))

        # combine loss
        losses['loss_so3'] = loss_so3
        losses['loss_quat'] = loss_quat
        return losses

    def forward_test_2d(self, **kwargs):
        heatmap_list, feature_list = self.backbone2d(kwargs['img'])
        out_feature_2d = feature_list[-1]
        pred_dict_2d = {'out_features_2d': out_feature_2d}
        return pred_dict_2d

    def forward_test_3d(self, **kwargs):
        pred_dict_2d = self.forward_test_2d(**kwargs)
        heatmap_list, feature_list = self.backbone3d(pred_dict_2d['out_features_2d'])
        out_heatmap_3d = heatmap_list[-1]
        out_feature_3d = feature_list[-1]

        pred_joints_uvd = hm_to_uvd(out_heatmap_3d)
        pred_dict_3d = {'out_features_3d': out_feature_3d,
                        'out_heatmap_3d': out_heatmap_3d[:, :21, ...],
                        'pred_joints_uvd': pred_joints_uvd,
                        }
        return pred_dict_3d

    def forward_test_mano(self, **kwargs):
        pred_dict_3d = self.forward_test_3d(**kwargs)
        pred_dict_mano = self.manonet(pred_dict_3d, kwargs['K'])
        pred_dict_ik = self.iknet.forward_test(pred_dict_mano['joints_xyz_mano'])

        pred_dict_handtailor = {
            'heatmap': pred_dict_3d['out_heatmap_3d'],
            'joints_uvd': pred_dict_3d['pred_joints_uvd'],
            'joints_xyz': pred_dict_mano['joints_xyz_mano'],
            'beta': pred_dict_mano['beta'],
            'quat': pred_dict_ik['quat'],
            'so3': pred_dict_ik['so3'],
        }
        return pred_dict_handtailor

    def forward_test(self, **kwargs):
        pred_dict_handtailor = self.forward_test_mano(**kwargs)
        return pred_dict_handtailor


    def forward(self, return_loss=True, current_epoch=-1, **kwargs):
        if return_loss == True:
            if current_epoch < self.epoch_2d:
                losses, _ = self.forward_train_2d(**kwargs)
                return losses

            elif self.epoch_2d <= current_epoch < self.epoch_2d + self.epoch_3d:
                losses, _ = self.forward_train_3d(**kwargs)
                return losses

            elif self.epoch_2d + self.epoch_3d <= current_epoch < self.epoch_2d + self.epoch_3d + self.epoch_mano:
                losses = self.forward_train_mano(**kwargs)
                return losses
        else:
            pred_dict = self.forward_test(**kwargs)
            return pred_dict,

    def train_step(self, data, optimizer, current_epoch=-1):
        # This train step only can be used for handtailor with 'HandTailorRunner'
        # A new parameter 'current_epoch' is added. Therefore the different training stages
        # can be controlled by epoch.
        losses = self(**data, current_epoch=current_epoch)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))

        return outputs

    def val_step(self, data, optimizer, current_epoch=-1):
        return self.train_step(data, optimizer, current_epoch)

    def simple_test(self, img, img_metas, **kwargs):
        pass

    def aug_test(self, imgs, img_metas, **kwargs):
        pass

    def extract_feat(self, imgs):
        pass

