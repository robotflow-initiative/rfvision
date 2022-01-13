# Copyright (c) OpenMMLab. All rights reserved.
import os
import numpy as np
from rfvision.datasets.builder import DATASETS
from .rhd2d_dataset import Rhd2DDataset


@DATASETS.register_module()
class Rhd3DDataset(Rhd2DDataset):
    def __init__(self,
                 ann_file,
                 img_prefix,
                 data_cfg,
                 pipeline,
                 dataset_info=None,
                 test_mode=False):
        super().__init__(ann_file=ann_file,
                         img_prefix=img_prefix,
                         data_cfg=data_cfg,
                         dataset_info=dataset_info,
                         test_mode=test_mode)

    def _get_db(self):
        """Load dataset."""
        gt_db = []
        bbox_id = 0
        num_joints = self.ann_info['num_joints']
        for img_id in self.img_ids:

            ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
            objs = self.coco.loadAnns(ann_ids)

            for obj in objs:
                if max(obj['keypoints']) == 0:
                    continue
                joints_3d = np.zeros((num_joints, 3), dtype=np.float32)
                joints_3d_visible = np.zeros((num_joints, 3), dtype=np.float32)

                keypoints = np.array(obj['keypoints']).reshape(-1, 3)
                joints_3d[:, :2] = keypoints[:, :2]
                joints_3d_visible[:, :2] = np.minimum(1, keypoints[:, 2:3])

                # the ori image is 224x224
                center, scale = self._xywh2cs(*obj['bbox'][:4], padding=1.25)

                image_file = os.path.join(self.img_prefix,
                                          self.id2name[img_id])
                gt_db.append({
                    'image_file': image_file,
                    'center': center,
                    'scale': scale,
                    'rotation': 0,
                    'joints_3d': joints_3d,
                    'joints_3d_visible': joints_3d_visible,
                    'dataset': self.dataset_name,
                    'bbox': obj['bbox'],
                    'bbox_score': 1,
                    'bbox_id': bbox_id,
                    'joints_cam': np.array(obj['joint_cam'])
                })
                bbox_id = bbox_id + 1
        gt_db = sorted(gt_db, key=lambda x: x['bbox_id'])

        return gt_db
if __name__ == '__main__':
    import rflib
    import numpy as np
    # get depth bound in rhd
    anno_file = rflib.load('/hdd0/data/rhd/RHD_published_v2/annotations/rhd_train.json')
    all_depth = np.hstack([np.array(i['joint_cam'])[:, 2] for i in anno_file['annotations']])
    max_bound = all_depth.max()
    min_bound = all_depth.min()