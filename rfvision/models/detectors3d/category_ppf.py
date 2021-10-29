from rfvision.models import BaseDetector, build_head, build_loss, DETECTORS


class CategoryPPF(BaseDetector):
    def __init__(self,
                 point_encoder,
                 ppf_encoder,
                 init_cfg,
                 ):
        super().__init__(init_cfg=init_cfg)
        self.point_encoder = point_encoder
        self.ppf_encoder = ppf_encoder

    def forward_train(self, imgs, img_metas, **kwargs):
        pass

