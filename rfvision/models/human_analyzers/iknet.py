from rfvision.models.builder import DETECTORS, build_backbone, build_head, build_loss, build_detector
from rfvision.models import BaseDetector


@DETECTORS.register_module()
class IKNet(BaseDetector):
    def __init__(self,
                 backbone,
                 init_cfg=None,
                 **kwargs,):
        super().__init__()
        self.backbone = build_backbone(backbone)
        self.init_weights(init_cfg=init_cfg)

    def init_weights(self, init_cfg=None):
        self.backbone.init_weights(init_cfg)

    def forward_train(self, joints_xyz, quat):
        gt_joints_xyz = joints_xyz
        gt_quat = quat
        pred_so3, pred_quat = self.backbone(gt_joints_xyz)
        losses = self.backbone.loss_ik(pred_quat, gt_quat)
        return losses

    def forward_test(self, joints_xyz):
        gt_joints_xyz = joints_xyz
        pred_so3, pred_quat = self.backbone(gt_joints_xyz)
        pred_dict = {'so3': pred_so3,
                     'quat': pred_quat}
        return pred_dict

    def forward(self, return_loss=True, **kwargs):
        joints_xyz = kwargs['joints_xyz_ik']
        quat = kwargs['quat']
        if return_loss == True:
            losses = self.forward_train(joints_xyz, quat)
            return losses
        else:
            pred_dict = self.forward_test(joints_xyz)
            return pred_dict,

    def train_step(self, data, optimizer):
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)
        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['quat']))
        return outputs

    def val_step(self, data, optimizer):
        return self.train_step(data, optimizer)

    def aug_test(self,):
        pass

    def extract_feat(self,):
        pass

    def simple_test(self,):
        pass
