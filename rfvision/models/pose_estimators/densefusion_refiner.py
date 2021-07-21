from rfvision.models.builder import build_neck,build_head, POSE_ESTIMATORS
import torch
from ..detectors import BaseDetector

@POSE_ESTIMATORS.register_module()
class DenseFusionRefiner(BaseDetector):
    def __init__(self,
                 neck,
                 pose_head,
                 train_cfg,
                 test_cfg):
        super(DenseFusionRefiner, self).__init__()

        self.neck = build_neck(neck)
        self.pose_head = build_head(pose_head)

    def forward(self, x, new_target, pred_dict, return_loss=True, **input):
        if return_loss:
            return self.forward_train(x, new_target, pred_dict, **input)
        else:
            return self.forward_test(x, new_target, pred_dict, **input)

    def forward_train(self, new_points, new_target, pred_dict, **input):
        bs = new_points.size()[0]
        emb = pred_dict['emb']
        obj = input['index']
        model_points = input['model_points']

        x = new_points.transpose(2, 1).contiguous()
        ap_x = self.neck(x, emb)

        out_rx, out_tx = self.pose_head(ap_x, emb, obj, bs)

        pred_dict = dict(out_rx=out_rx,
                         out_tx=out_tx,
                         emb=emb)

        loss_result, new_points, new_target = self.pose_head.loss(pred_dict, new_target, model_points, obj, new_points)

        return pred_dict, loss_result, new_points, new_target

    def forward_test(self, x, pred_dict, **input):
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