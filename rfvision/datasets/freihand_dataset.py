import os
import rflib
import numpy as np
import torch
from torch.utils.data import Dataset
from rfvision.datasets import DATASETS
from rfvision.datasets.pipelines import Compose
from rfvision.core.evaluation_keypoints import keypoint_epe, keypoint_pck_accuracy,  keypoint_auc


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
        self.training_scale = rflib.load(os.path.join(self.data_root, 'training_scale.json')) * 4
        self.training_mano = rflib.load(os.path.join(self.data_root, 'training_mano.json')) * 4
        self.training_K = rflib.load(os.path.join(self.data_root, 'training_K.json')) * 4
        self.training_xyz = rflib.load(os.path.join(self.data_root, 'training_xyz.json')) * 4


        ##################### split setting ###################
        split_ratio = [0.998, 0.001, 0.001]  # train / test / val
        shuffle = True
        total_length = len(self.training_xyz)
        ###################################################
        np.random.seed(0)
        assert sum(split_ratio) == 1
        split_id = np.arange(total_length)
        if shuffle == True:
            np.random.seed(0)
            np.random.shuffle(split_id)

        if split == 'train':
            self.length = int(total_length * split_ratio[0])
            self.split_id = split_id[:self.length]
        elif split == 'test':
            self.length = int(total_length * (1 - split_ratio[0]))
            self.split_id = split_id[self.length:]
        elif split == 'val':
            self.length = int(total_length * (1 - split_ratio[0] - split_ratio[1]))
            self.split_id = split_id[self.length:]
        self.pipeline = Compose(pipeline)
        self._set_group_flag()

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        idx = self.split_id[idx]  # map index
        results = {
            'img_prefix': None,
            'img_info': {'filename': os.path.join(self.training_root, 'rgb', '%08d' % idx + '.jpg')},
            'idx': idx,
            'scale': np.float32(self.training_scale[idx]),
            'mano': np.float32(self.training_mano[idx]),
            'joints_xyz': np.float32(self.training_xyz[idx]),
            'K': np.float32(self.training_K[idx]),
        }

        results = self.pipeline(results)
        return results

    def _set_group_flag(self):
        self.flag = np.zeros(len(self), dtype=np.uint8)


    def evaluate(self, results, res_folder, metric, **kwargs):
        # notes:
        # kps: keypoints
        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['PCK', 'AUC', 'EPE']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        gt_kps, pred_kps = (), ()
        gt_kps_visible = ()
        for i in range(len(results)):
            gt_result = self[i]
            pred_result = results[i]

            gt_kps_visible += (gt_result['kps_visible'],)
            gt_kps += (gt_result['kps'],)
            pred_kps += (pred_result['kps'],)

        gt_kps_visible = np.array(gt_kps_visible)
        gt_kps = np.array(gt_kps)
        pred_kps = np.array(torch.cat(pred_kps).cpu())

        normalize = np.ones((len(results), 2))

        score = {}
        for metric in metrics:
            if metric == 'EPE':
                score['EPE'] = keypoint_epe(pred_kps, gt_kps, gt_kps_visible)
            elif metric == 'AUC':
                score['AUC'] = keypoint_auc(pred_kps, gt_kps, gt_kps_visible,
                                            normalize=normalize)
            elif metric == 'PCK':
                score['PCK'] = keypoint_pck_accuracy(pred_kps, gt_kps, gt_kps_visible,
                                                     thr=0.2,
                                                     normalize=normalize)[1]

        rflib.dump(score, os.path.join(res_folder, 'result_keypoints.json'))

        return score


if __name__ == '__main__':
    dataset = FreiHandDataset(data_root='/disk2/data/hand3d/FreiHAND/FreiHAND_pub_v1')
    sample = dataset[0]
