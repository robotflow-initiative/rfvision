from rfvision.models.builder import build_backbone, build_head, POSE_ESTIMATORS
import torch
from ..detectors import BaseDetector


@POSE_ESTIMATORS.register_module()
class ANCSH(BaseDetector):
    '''
    This module is a recurrence of 'Category-Level Articulated Object Pose Estimation'
    http://cn.arxiv.org/abs/1912.11913
    '''
    
    def __init__(self,
                 backbone,
                 train_cfg,
                 test_cfg,
                 nocs_head=None,
                 init_cfg=None):
        super(ANCSH, self).__init__(init_cfg)

        self.backbone = build_backbone(backbone)

        if nocs_head is not None:
            self.nocs_head = build_head(nocs_head)

    def forward(self, return_loss=True, **input):
        if return_loss:
            return self.forward_train(**input)
        else:
            return self.forward_test(**input)

    def forward_train(self, **input):
        P = input['parts_pts']
        if 'parts_pts_feature' in input.keys():
            P_feature = input['parts_pts_feature']
        else:
            P_feature = None
        if P.dim() == 4:
            assert P.shape[1] == 2
            P1 = P[:, 0, :, :]
            P2 = P[:, 1, :, :]

            feat1, feat1_encode = self.backbone(P1)
            feat2, feat2_encode = self.backbone(P2)

            feat = torch.cat((feat1, feat2), dim=2)
            feat_encode = torch.stack([feat1_encode, feat2_encode], dim=2)
        else:
            feat, feat_encode = self.backbone(torch.cat((P, P_feature), 2))
        pred_dict = self.nocs_head(feat)
        loss_result = self.nocs_head.loss(pred_dict, mode='train', **input)

        return loss_result

    def forward_test(self, **input):
        if 'parts_pts' not in input:
            return None
        else:
            P = input['parts_pts']
            # P = input['nocs_g'].double()
            if 'parts_pts_feature' in input.keys():
                P_feature = input['parts_pts_feature']
            else:
                P_feature = None
            if P.dim() == 4:
                assert P.shape[1] == 2
                P1 = P[:, 0, :, :]
                P2 = P[:, 1, :, :]

                feat1, feat1_encode = self.backbone(P1)
                feat2, feat2_encode = self.backbone(P2)

                feat = torch.cat((feat1, feat2), dim=2)
                feat_encode = torch.cat((feat1_encode, feat2_encode), dim=2)
            else:
                feat, feat_encode = self.backbone(torch.cat((P, P_feature), 2))
            # feat = self.backbone(P, P_feature)
            pred_dict = self.nocs_head(feat, feat_encode)
            return pred_dict
    
    def simple_test(self):
        pass
    
    def aug_test(self):
        pass
    
    def extract_feat(self):
        pass