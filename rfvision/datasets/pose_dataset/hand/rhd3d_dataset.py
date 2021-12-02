# Copyright (c) OpenMMLab. All rights reserved.
import os
from collections import OrderedDict

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
        super().__init__(ann_file,
                         img_prefix,
                         data_cfg,
                         pipeline,
                         dataset_info,
                         test_mode)
        self.ann_info['heatmap3d_depth_bound'] = data_cfg[
            'heatmap3d_depth_bound']
        self.ann_info['heatmap_size_root'] = data_cfg['heatmap_size_root']
        self.ann_info['root_depth_bound'] = data_cfg['root_depth_bound']

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
                # For train set, joints_z in range (-52.540000915527344, 1182.0)
                # For test set, joints_z in range (-48.76000213623047, 994.0)
                # For train set, joints_z (root_relative, root_joint_id: 0) in range (-326.9000244140625, 294.6999816894531)
                # For test set, joints_z (root_relative, root_joint_id: 0) in range (-199.5999755859375, 189.99996948242188)
                joints_xyz = np.array(obj['joint_cam'])
                joints_xyz_rel = joints_xyz - joints_xyz[0]  # root relative, root_joint_id: 0

                joints_3d = np.zeros((num_joints, 3), dtype=np.float32)
                joints_3d_visible = np.zeros((num_joints, 3), dtype=np.float32)
                keypoints = np.array(obj['keypoints']).reshape(-1, 3)

                # joints_3d : joints_uvd
                joints_3d[:, :2] = keypoints[:, :2]
                joints_3d[:, 2] = joints_xyz_rel[:, 2]
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

                    'focal': self.coco.imgs[img_id]['cam_param']['focal'],
                    'princpt': self.coco.imgs[img_id]['cam_param']['princpt'],
                    'rel_root_depth': joints_xyz[0, 2],  # abs depth of root joint
                    'rel_root_valid': 1,
                    'hand_type': self._encode_handtype(obj['hand_type']),
                    'hand_type_valid': 1,
                    'joints_xyz': joints_xyz
                })
                bbox_id = bbox_id + 1
        gt_db = sorted(gt_db, key=lambda x: x['bbox_id'])

        return gt_db

    def evaluate(self, outputs, res_folder='./', metric='PCK', **kwargs):
        """Evaluate rhd keypoint results. The pose prediction results will be
        saved in `${res_folder}/result_keypoints.json`.

        Note:
            batch_size: N
            num_keypoints: K
            heatmap height: H
            heatmap width: W

        Args:
            outputs (list(preds, boxes, image_path, output_heatmap))
                :preds (np.ndarray[N,K,3]): The first two dimensions are
                    coordinates, score is the third dimension of the array.
                :boxes (np.ndarray[N,6]): [center[0], center[1], scale[0]
                    , scale[1],area, score]
                :image_paths (list[str]): For example
                    , ['training/rgb/00031426.jpg']
                :output_heatmap (np.ndarray[N, K, H, W]): model outpus.

            res_folder (str): Path of directory to save the results.
            metric (str | list[str]): Metric to be performed.
                Options: 'PCK', 'AUC', 'EPE'.

        Returns:
            dict: Evaluation results for evaluation metric.
        """
        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['PCK', 'AUC', 'EPE']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        res_file = os.path.join(res_folder, 'result_keypoints.json')

        kpts = []
        for output in outputs:
            preds = output['preds']
            boxes = output['boxes']
            image_paths = output['image_paths']
            bbox_ids = output['bbox_ids']

            batch_size = len(image_paths)
            for i in range(batch_size):
                image_id = self.name2id[image_paths[i][len(self.img_prefix):]]

                kpts.append({
                    'keypoints': preds[i][:, :3].tolist(),
                    'center': boxes[i][0:2].tolist(),
                    'scale': boxes[i][2:4].tolist(),
                    'area': float(boxes[i][4]),
                    'score': float(boxes[i][5]),
                    'image_id': image_id,
                    'bbox_id': bbox_ids[i]
                })
        kpts = self._sort_and_unique_bboxes(kpts)

        self._write_keypoint_results(kpts, res_file)
        info_str = self._report_metric(res_file, metrics)
        name_value = OrderedDict(info_str)

        return name_value

    @staticmethod
    def _encode_handtype(hand_type):
        if hand_type == 'right':
            return np.array([1, 0], dtype=np.float32)
        elif hand_type == 'left':
            return np.array([0, 1], dtype=np.float32)
        elif hand_type == 'interacting':
            return np.array([1, 1], dtype=np.float32)
        else:
            assert 0, f'Not support hand type: {hand_type}'