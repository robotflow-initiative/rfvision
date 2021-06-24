import os
import numpy as np
import torch
import rflib
from . import DATASETS
from .custom3d import Custom3DDataset
from rfvision.components import IKNetBackbone

SNAP_PARENT = [
    0, # 0's parent
    0, # 1's parent
    1,
    2,
    3,
    0, # 5's parent
    5,
    6,
    7,
    0, # 9's parent
    9,
    10,
    11,
    0, # 13's parent
    13,
    14,
    15,
    0, # 17's parent
    17,
    18,
    19,
]


@DATASETS.register_module()
class INVKDataset(Custom3DDataset):
    CLASSES = None
    def __init__(self,
                 data_root='/hddisk1/data/IKdataset',
                 data_source = ["freihand_gt","GenData"],
                 split='train',
                 shuffle=False,
                 **kwargs):
        assert split in ('train', 'test', 'val', 'all')
        joints_xyz_all = ()
        quats_all = ()
        if 'freihand_gt' in data_source:
            freihand_gt_path = os.path.join(data_root, 'freihand_j_s_q.pkl')
            data = rflib.load(freihand_gt_path)
            joints_xyz_all += np.array(data['joints']),
            quats_all += np.array(data['quat']),
        if 'GenData' in data_source:
            gen_data_root = os.path.join(data_root, 'GenData')
            for i in os.listdir(gen_data_root):
                if i.endswith('.pkl'):
                    data = rflib.load(os.path.join(gen_data_root,i))
                    joints_xyz_all += data['joint_'],
                    quats_all += data['quat'],
        joints_xyz_all = np.concatenate(joints_xyz_all, axis=0)
        quats_all = np.concatenate(quats_all, axis=0)
        # split dataset
        split_ratio = 0.99
        if shuffle == True:
            all_idx = np.random.randint(0, len(joints_xyz_all), len(joints_xyz_all))
        else:
            all_idx = list(range(len(joints_xyz_all)))

        train_length = int(split_ratio * len(joints_xyz_all))
        train_idx = all_idx[:train_length]
        test_idx = all_idx[train_length:]

        if split == 'train':
            self.joints_xyz = joints_xyz_all[train_idx]
            self.quats = quats_all[train_idx]
        elif split == 'test' or split =='val':
            self.joints_xyz = joints_xyz_all[test_idx]
            self.quats = quats_all[test_idx]
        elif split == 'all':
            self.joints_xyz = joints_xyz_all
            self.quats = quats_all

        self.ref_bone_link = (0, 9)
        self._set_group_flag()

    def __len__(self):
        return len(self.joints_xyz)

    def __getitem__(self, index):
        joints_xyz = self.joints_xyz[index]
        quat = self.quats[index]

        results = {'joints_xyz': joints_xyz,
                   'quat': quat}
        results.update(self._preprocess_joint(joints_xyz))  # add preprocess info to results
        return results

    def _preprocess_joint(self, joints_xyz):
        joint_bone = np.linalg.norm(self.ref_bone_link[1] - self.ref_bone_link[0])
        joints_xyz_ik = joints_xyz / joint_bone
        kin_chain = np.array([joints_xyz_ik[i] - joints_xyz_ik[SNAP_PARENT[i]] for i in range(len(joints_xyz_ik))])
        kin_len = np.linalg.norm(kin_chain, ord=2, axis=-1, keepdims=True)

        results = {'joint_bone': joint_bone,
                   'joints_xyz_ik': joints_xyz_ik,
                   'kin_chain': kin_chain,
                   'kin_len': kin_len}
        return results

    def evaluate(self,
                 results,
                 metric='mean_loss',
                 logger=None,
                 ):
        loss_quat_l2 = 0
        loss_quat_cos = 0
        for i in range(len(self)):
            pred = results[i]
            gt = self.__getitem__(i)
            pred_quat = pred['quat'].squeeze(0).cuda()
            gt_quat = torch.tensor(gt['quat']).cuda()
            losses = IKNetBackbone.loss_ik(pred_quat, gt_quat)
            loss_quat_l2 += float(losses['loss_quat_l2'])
            loss_quat_cos += float(losses['loss_quat_cos'])

        eval_dict = {'mean_loss_quat_l2': loss_quat_l2 / len(self),
                     'mean_loss_quat_cos': loss_quat_cos / len(self)}
        return eval_dict

if __name__ == '__main__':
    dataset = INVKDataset()
    sample = dataset[0]
