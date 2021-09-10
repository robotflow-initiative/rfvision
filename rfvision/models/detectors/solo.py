from rfvision.models.builder import DETECTORS, build_backbone, build_neck, build_head
from rfvision.models.detectors import BaseDetector

@DETECTORS.register_module()
class SOLOV2(BaseDetector):

    def __init__(self,
                 backbone,
                 neck,
                 mask_feat_head,
                 bbox_head,
                 init_cfg=None,
                 test_cfg=None,
                 train_cfg=None,
                 ):
        super(SOLOV2, self).__init__(init_cfg=init_cfg)
        self.backbone = build_backbone(backbone)
        self.mask_feat_head = build_head(mask_feat_head)
        self.neck = build_neck(neck)
        self.bbox_head = build_head(bbox_head)
        self.test_cfg = test_cfg

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        x = self.neck(x)
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/get_flops.py`
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None
                      ):

        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        mask_feat_pred = self.mask_feat_head(
            x[self.mask_feat_head.start_level:self.mask_feat_head.end_level + 1])
        loss_inputs = outs + (mask_feat_pred, gt_bboxes, gt_labels, gt_masks, img_metas)

        losses = self.bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

    '''
    img_metas context
    'filename': 'data/casia-SPT_val/val/JPEGImages/00238.jpg', 
    'ori_shape': (402, 600, 3), 'img_shape': (448, 669, 3), 
    'pad_shape': (448, 672, 3), 'scale_factor': 1.1144278606965174, 'flip': False, 
    'img_norm_cfg': {'mean': array([123.675, 116.28 , 103.53 ], dtype=float32), 
    'std': array([58.395, 57.12 , 57.375], dtype=float32), 'to_rgb': True}}

    '''

    def forward_test(self, imgs, img_metas, **kwargs):
        """
        Args:
            imgs (List[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_meta (List[List[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch
        """
        for var, name in [(imgs, 'imgs'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))

        num_augs = len(imgs)
        if num_augs != len(img_metas):
            raise ValueError(
                'num of augmentations ({}) != num of image meta ({})'.format(
                    len(imgs), len(img_metas)))
        # TODO: remove the restriction of imgs_per_gpu == 1 when prepared
        imgs_per_gpu = imgs[0].size(0)
        assert imgs_per_gpu == 1

        if num_augs == 1:
            return self.simple_test(imgs[0], img_metas[0], **kwargs)
        else:
            return self.aug_test(imgs, img_metas, **kwargs)

    def simple_test(self, img, img_meta, rescale=False):

        # test_tensor = torch.ones(1,3,448,512).cuda()
        # x = self.extract_feat(test_tensor)
        x = self.extract_feat(img)

        outs = self.bbox_head(x, eval=True)

        mask_feat_pred = self.mask_feat_head(
            x[self.mask_feat_head.
                  start_level:self.mask_feat_head.end_level + 1])

        seg_inputs = outs + (mask_feat_pred, img_meta, self.test_cfg, rescale)

        seg_result = self.bbox_head.get_seg(*seg_inputs)
        return seg_result

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test function with test time augmentation."""
        raise NotImplementedError