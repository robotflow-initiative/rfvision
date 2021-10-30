import os
import numpy as np
from rfvision.datasets import DATASETS
from torch.utils.data import Dataset


import os
import numpy as np
import torch
import rflib
from rfvision.datasets import DATASETS
from rfvision.datasets.custom3d import Custom3DDataset

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
class IKDataset(Custom3DDataset):
    CLASSES = None
    def __init__(self,
                 data_root='/hddisk1/data/IKdataset',
                 data_source = [
                     "freihand_gt",
                     "GenData"
                 ],
                 split='all',
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
        joints_xyz = normalize_point_cloud(joints_xyz)[0]
        quat = self.quats[index]

        results = {'joints_xyz': joints_xyz,
                   'quat': quat}

        # results.update(self._preprocess_joint(joints_xyz))  # add preprocess info to results
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


def normalize_point_cloud(pc):
    centroid = pc[9]
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc_normalized = pc / m
    return pc_normalized, centroid, m


# @DATASETS.register_module()
# class IKDataset(Dataset):
#     '''
#     We use manotorch(https://github.com/lixiny/manotorch) to generate the INVKDataset
#     according to following joint sequence.
#         0: 'wrist',
#         1: 'thumb1',
#         2: 'thumb2',
#         3: 'thumb3',
#         4: 'thumb4',
#         5: 'forefinger1',
#         6: 'forefinger2',
#         7: 'forefinger3',
#         8: 'forefinger4',
#         9: 'middle_finger1',
#         10: 'middle_finger2',
#         11: 'middle_finger3',
#         12: 'middle_finger4',
#         13: 'ring_finger1',
#         14: 'ring_finger2',
#         15: 'ring_finger3',
#         16: 'ring_finger4',
#         17: 'pinky_finger1',
#         18: 'pinky_finger2',
#         19: 'pinky_finger3',
#         20: 'pinky_finger4'
#
#     INVKDataset includes two metas: joints_xyz (bz, 21, 3) and full_posesernions (bz, 48)
#     Note: if you download our dataset (without any processing such as normalization and root relative), bz = 1000000. For more details: https://github.com/lixiny/manotorch
#     '''
#
#     def __init__(self,
#                  data_root='/hddisk1/data/IKdataset',
#                  split='train',
#                  shuffle=False,
#                  **kwargs):
#
#         assert split in ('train', 'test', 'val', 'all')
#         joints_xyz_path = os.path.join(data_root, 'joints_xyz.npy')
#         full_poses_path = os.path.join(data_root, 'full_poses.npy')
#         joints_xyz = np.load(joints_xyz_path, allow_pickle=True)
#         full_poses = np.load(full_poses_path)
#
#         # split dataset
#         split_ratio = 0.8
#         if shuffle == True:
#             all_idx = np.random.randint(0, len(joints_xyz), len(joints_xyz))
#         else:
#             all_idx = list(range(len(joints_xyz)))
#
#         train_length = int(split_ratio * len(joints_xyz))
#         train_idx = all_idx[:train_length]
#         test_idx = all_idx[train_length:]
#
#         if split == 'train':
#             self.joints_xyz = joints_xyz[train_idx]
#             self.full_poses = full_poses[train_idx]
#         elif split == 'test' or split =='val':
#             self.joints_xyz = joints_xyz[test_idx]
#             self.full_poses = full_poses[test_idx]
#         elif split == 'all':
#             self.joints_xyz = joints_xyz
#             self.full_poses = full_poses
#
#         self._processing()
#
#     def _processing(self):
#         self.full_poses = self.full_poses.reshape(-1, 16, 3)
#
#     def __len__(self):
#         return len(self.joints_xyz)
#
#     def __getitem__(self, index):
#         joints_xyz = self.joints_xyz[index]
#         joints_xyz = normalize_point_cloud(joints_xyz)[0]
#         full_poses = self.full_poses[index]
#         results = {'joints_xyz': joints_xyz,
#                    'full_poses': full_poses}
#         return results


if __name__ == '__main__':
    pass
    # dataset = INVKDataset()
    # sample = dataset[0]



