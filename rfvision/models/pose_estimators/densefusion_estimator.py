from robotflow.rflearner.builder import build_backbone, build_head, build_neck, DETECTORS
import torch
from ..detectors import BaseDetector
@DETECTORS.register_module()
class DenseFusionEstimator(BaseDetector):
    def __init__(self,
                 backbone,
                 neck,
                 pose_head,
                 train_cfg,
                 test_cfg):
        super(DenseFusionEstimator, self).__init__()

        self.backbone = build_backbone(backbone)
        self.neck = build_neck(neck)
        self.pose_head = build_head(pose_head)

    def forward(self, return_loss=True, **input):
        if return_loss:
            return self.forward_train(**input)
        else:
            return self.forward_test(**input)

    def forward_train(self, **input):
        img = input['img']
        x = input['cloud']
        choose = input['choose']
        model_points = input['model_points']
        obj = input['index']
        target = input['target']

        out_img = self.backbone(img)

        bs, di, _, _ = out_img.size()

        emb = out_img.view(bs, di, -1)
        choose = choose.repeat(1, di, 1)
        emb = torch.gather(emb, 2, choose).contiguous()

        x = x.transpose(2, 1).contiguous()
        ap_x = self.neck(x, emb)

        out_rx, out_tx, out_cx = self.pose_head(ap_x, obj, bs)

        pred_dict = dict(out_rx=out_rx,
                         out_tx=out_tx,
                         out_cx=out_cx,
                         emb=emb.detach())
        loss_result, new_points, new_target = self.pose_head.loss(pred_dict, target, model_points, obj, x)

        return pred_dict, loss_result, new_points, new_target

    def forward_test(self, **input):
        img = input['img_masked']
        x = input['cloud']
        choose = input['choose']
        obj = input['index']

        out_img = self.backbone(img)

        bs, di, _, _ = out_img.size()

        emb = out_img.view(bs, di, -1)
        choose = choose.repeat(1, di, 1)
        emb = torch.gather(emb, 2, choose).contiguous()

        x = x.transpose(2, 1).contiguous()
        ap_x = self.neck(x, emb)

        out_rx, out_tx, out_cx = self.pose_head(ap_x, obj)

        pred_dict = dict(out_rx=out_rx,
                         out_tx=out_tx,
                         out_cx=out_cx,
                         emb=emb.detach())

        return pred_dict
    
    def simple_test(self):
        pass
    
    def aug_test(self):
        pass
    
    def extract_feat(self):
        pass