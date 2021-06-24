import os
import robotflow.rflib
import numpy as np
from torch.utils.data import Dataset
import torch
from robotflow.rflearner.datasets import DATASETS
from robotflow.rflearner.datasets.pipelines import Compose
from robotflow.rflearner.bricks.utils.handtailor_utils import (hm_to_kp2d, uvd2xyz, get_pck_all)

@DATASETS.register_module()
class FreiHandDataset(Dataset):
    CLASSES = None

    def __init__(self,
                 data_root,
                 pipeline,
                 split='train',
                 **kwargs
                 ):
        assert split in ['train', 'test', 'val']
        self.data_root = data_root
        self.split = split
        self.training_root = os.path.join(self.data_root, 'training')
        # loading
        self.training_scale = robotflow.rflib.load(os.path.join(self.data_root, 'training_scale.json')) * 4
        self.training_mano = robotflow.rflib.load(os.path.join(self.data_root, 'training_mano.json')) * 4
        self.training_K = robotflow.rflib.load(os.path.join(self.data_root, 'training_K.json')) * 4
        self.training_xyz = robotflow.rflib.load(os.path.join(self.data_root, 'training_xyz.json')) * 4

        # split train / test
        split_ratio = 0.9
        split_id = tuple(range(len(self.training_xyz)))
        if split == 'train':
            self.length = int(len(self.training_xyz) * split_ratio)
            self.split_id = split_id[:self.length]
        elif split == 'test' or split == 'val':
            self.length = int(len(self.training_xyz) * (1 - split_ratio))
            self.split_id = split_id[self.length:]

        self.pipeline = Compose(pipeline)
        self._set_group_flag()

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        idx = self.split_id[idx]  # map index
        rgb = robotflow.rflib.imread(os.path.join(self.training_root, 'rgb', '%08d' % idx + '.jpg'))
        results = {
            'img': rgb,
            'scale': np.float32(self.training_scale[idx]),
            'mano': np.float32(self.training_mano[idx]),
            'joints_xyz': np.float32(self.training_xyz[idx]),
            'K': np.float32(self.training_K[idx]),
        }

        results = self.pipeline(results)
        return results

    def _set_group_flag(self):
        self.flag = np.zeros(len(self), dtype=np.uint8)

    def evaluate(self,
                 results,
                 metric='mAP',
                 logger=None,
                 **kwargs):
        gt_joints_xyzs, pred_joints_xyzs, pred_joints_xyzs_mano = (), (), ()

        for idx in range(len(self)):
            gt_result = self.__getitem__(idx)
            gt_result = {key: value.cpu() if key!='img_metas' in gt_result else value for key, value in gt_result.items()}
            pred_result = results[idx]
            pred_result = {key: value.cpu() if  key != 'img_metas' in gt_result else value for key, value in
                        pred_result.items()}
            gt_joints_uv = gt_result['joints_uv']
            pred_joints_uv = hm_to_kp2d(pred_result['heatmap']).squeeze(0)
            err_uv = float(torch.mean(torch.sqrt(torch.sum((gt_joints_uv - pred_joints_uv) ** 2, -1)), -1))

            gt_joints_uvd = gt_result['joints_uvd']
            pred_joints_uvd = pred_result['joints_uvd'].squeeze(0)
            err_uvd = float(torch.mean(torch.sqrt(torch.sum((gt_joints_uvd - pred_joints_uvd) ** 2, -1)), -1))

            # to tuple
            joint_root = gt_result['joint_root'].unsqueeze(0)
            joint_bone = gt_result['joint_bone']
            pred_joints_uvd = pred_result['joints_uvd']
            K = gt_result['K'].unsqueeze(0)
            gt_joints_xyzs += gt_result['joints_xyz'],
            pred_joints_xyzs += uvd2xyz(pred_joints_uvd, joint_root, joint_bone, K),
            pred_joints_xyzs_mano += (pred_result['joints_xyz'] * gt_result['joint_bone'] + gt_result['joint_root']),

        gt_joints_xyzs = torch.stack(gt_joints_xyzs)
        pred_joints_xyzs = torch.stack(pred_joints_xyzs).squeeze()
        pred_joints_xyzs_mano = torch.stack(pred_joints_xyzs_mano).squeeze()

        err_xyz_20 = get_pck_all(pred_joints_xyzs, gt_joints_xyzs, threshold=20)
        err_xyz_30 = get_pck_all(pred_joints_xyzs, gt_joints_xyzs, threshold=30)
        err_xyz_40 = get_pck_all(pred_joints_xyzs, gt_joints_xyzs, threshold=40)

        err_xyz_mano_20 = get_pck_all(pred_joints_xyzs_mano, gt_joints_xyzs, threshold=20)
        err_xyz_mano_30 = get_pck_all(pred_joints_xyzs_mano, gt_joints_xyzs, threshold=30)
        err_xyz_mano_40 = get_pck_all(pred_joints_xyzs_mano, gt_joints_xyzs, threshold=40)

        eval_dict = {'err_xyz_20': float(err_xyz_20),
                     'err_xyz_30': float(err_xyz_30),
                     'err_xyz_40': float(err_xyz_40),
                     'err_xyz_mano_20': float(err_xyz_mano_20),
                     'err_xyz_mano_30': float(err_xyz_mano_30),
                     'err_xyz_mano_40': float(err_xyz_mano_40),
                     'err_uv': float(err_uv),
                     'err_uvd': float(err_uvd)}

        return eval_dict



if __name__ == '__main__':
    dataset = FreiHandDataset(data_root='/disk2/data/hand3d/FreiHAND/FreiHAND_pub_v1')
    sample = dataset[0]
