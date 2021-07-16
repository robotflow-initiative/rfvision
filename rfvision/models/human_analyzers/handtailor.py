from rfvision.models.builder import DETECTORS, build_backbone, build_loss, build_detector
from rfvision.models import BaseDetector
from rfvision.components.utils import heatmap_to_uv, batch_uv2xyz
import torch

# DEPTH_MIN = 3
# DEPTH_RANGE = -1.5


def heatmap_to_uvd(hm3d):
    b, c, h, w = hm3d.size()
    hm2d = hm3d[:, :21, ...]
    depth = hm3d[:, 21:, ...]
    uv = heatmap_to_uv(hm2d, mode='average') / torch.FloatTensor([[w, h]])
    hm2d = hm2d.view(b, 1, c // 2, -1)
    depth = depth.view(b, 1, c // 2, -1)
    hm2d = hm2d / torch.sum(hm2d, -1, keepdim=True)
    d = torch.sum(depth * hm2d, -1).permute(0, 2, 1)
    joints_uvd = torch.cat((uv, d), dim=-1)
    return joints_uvd


@DETECTORS.register_module()
class HandTailor(BaseDetector):
    '''
    The handtailor has a multi-stage training process:
    stage 0 (A independent stage) : train iknet independently.
    stage 1 - forward_train_2d : train (backbone_2d)
    stage 2 - forward_train_3d : train (backbone_2d trained in forward_train_2d) + (backbone_3d)

    stage 3 - forward_train_mano : train (backbone_2d trained in forward_train_3d) +
                                         (backbone_3d trained in forward_train_3d) +
                                         iknet (trained in stage0) + manonet
    Therefore, the function 'forward' is the combination of stage 0 ~ stage 3.
    '''



    def __init__(self,
                 manonet,
                 iknet,
                 backbone_2d,
                 backbone_3d,
                 loss_2d,
                 loss_3d,
                 loss_so3,
                 loss_joints_xyz,
                 loss_beta,
                 normalize_z=True,
                 epoch_2d=100,
                 epoch_3d=100,
                 epoch_mano=100,
                 init_cfg=None,
                 **kwargs):

        super().__init__(init_cfg)
        self.backbone_2d = build_backbone(backbone_2d)
        self.backbone_3d = build_backbone(backbone_3d)
        self.manonet = build_backbone(manonet)
        self.iknet = build_detector(iknet)

        self.loss_2d = build_loss(loss_2d)
        self.loss_3d = build_loss(loss_3d)
        self.loss_so3 = build_loss(loss_so3)
        self.loss_joints_xyz = build_loss(loss_joints_xyz)
        self.loss_beta = build_loss(loss_beta)

        self.normalize_z = normalize_z
        self.epoch_2d = epoch_2d
        self.epoch_3d = epoch_3d
        self.epoch_mano = epoch_mano


    def forward_train_2d(self, return_loss=True, **kwargs ):
        # release
        img = kwargs['img']
        heatmap = kwargs['heatmap']
        # heatmap_weight = kwargs['heatmap_weight']

        heatmap_list, feature_list = self.backbone_2d(img)
        out_headmap_2d = heatmap_list[-1]
        out_feature_2d = feature_list[-1]

        pred_dict_2d = {'out_features_2d': out_feature_2d}
        if return_loss == True:
            loss2d = self.loss_2d(out_headmap_2d, heatmap, None)
            losses = {'loss2d': loss2d}
            return losses, pred_dict_2d
        else:
            return pred_dict_2d

    def forward_train_3d(self, return_loss=True, **kwargs):
        losses, pred_dict_2d = self.forward_train_2d(**kwargs)
        heatmap_list, feature_list = self.backbone_3d(pred_dict_2d['out_features_2d'])

        out_heatmap_3d = heatmap_list[-1]
        out_feature_3d = feature_list[-1]

        pred_dict_3d = {'out_features_3d': out_feature_3d,
                        'out_heatmap_3d': out_heatmap_3d[:, :21, ...],
                        }

        if return_loss == True:
            pred_joints_uvd = heatmap_to_uvd(out_heatmap_3d)
            joints_xyz, joints_uv = kwargs['joints_xyz'], kwargs['joints_uv']
            if self.normalize_z == True:
                root_joint = joints_xyz[9]
                joint_bone = torch.norm(root_joint - joints_xyz[0])
                joints_z_normalized = (joints_xyz[:, :, 2:] - root_joint[2]) / joint_bone
                # joints_z_normalized = (joints_z_normalized - DEPTH_MIN) / DEPTH_RANGE  # shape (21, 1)
                gt_joints_uvd = torch.cat((joints_uv, joints_z_normalized), dim=-1)
            else:
                gt_joints_uvd = torch.cat((joints_uv, joints_xyz[:, :, 2:]), dim=-1)
            loss3d = self.loss_3d(pred_joints_uvd, gt_joints_uvd)
            losses['loss3d'] = loss3d
            return losses, pred_dict_3d
        else:
            return pred_dict_3d


    def forward_train_mano(self, return_loss=True, **kwargs):
        K = kwargs['K']
        img_shape = kwargs['img_shape']
        # gt_joints_xyz = kwargs['joints_xyz']
        # train
        losses, pred_dict_3d = self.forward_train_3d(**kwargs)
        joints_uvd = heatmap_to_uvd(pred_dict_3d['out_heatmap_3d'])
        joints_uv = joints_uvd[:, :, :2] * img_shape  # denormalize uv

        beta, root_joint_z, joint_bone = self.manonet(pred_dict_3d['out_features_3d'])

        if self.normalize_z == True:
            # denormalize
            joints_z_denormalized = joints_uvd[:, :, 2:] * joint_bone + root_joint_z
            # joints_z_denormalized = joints_z_normalized * DEPTH_RANGE + DEPTH_MIN
            joints_xyz = batch_uv2xyz(uv=joints_uv,
                                      K=K,
                                      depth=joints_z_denormalized)
            # normalize
            root_joint = joints_xyz[9]
            joint_bone = torch.norm(root_joint - joints_xyz[0])
            joints_z_normalized = (joints_xyz[:, :, 2:] - root_joint[2]) / joint_bone
            joints_xyz[:, :, 2] = joints_z_normalized
            so3, quat = self.iknet.forward_test(joints_xyz)

        else:
            joints_xyz = batch_uv2xyz(uv=joints_uv,
                                      K=K,
                                      depth=joints_uvd[:, :, 2:])
            root_joint = joints_xyz[9]
            joint_bone = torch.norm(root_joint - joints_xyz[0])


        pred_mano_dict ={
            'heatmap': pred_dict_3d['out_heatmap_3d'],
            'root_joint': root_joint,
            'joint_bone': joint_bone,
            'beta': beta,
            'quat': quat,
            'theta': so3,
        }

        if return_loss == True:
            # compute loss
            loss_so3 = self.loss_so3(so3, torch.zeros_like(so3))
            # loss_joints_xyz = self.loss_joints_xyz(quat, torch.zeros_like(quat))
            loss_beta = self.loss_beta(beta, torch.zeros_like(beta))
            # combine loss
            losses['loss_so3'] = loss_so3
            # losses['loss_joints_xyz'] = loss_joints_xyz
            losses['loss_beta'] = loss_beta
            return losses
        else:
            return pred_mano_dict

    def forward_test_mano(self, **kwargs):
        pred_dict_mano = self.forward_train_mano(kwargs, return_loss=False)
        return pred_dict_mano

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
            pred_dict_mano = self.forward_test_mano(kwargs)
            return pred_dict_mano


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

