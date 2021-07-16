import os
import rflib
import numpy as np
from torch.utils.data import Dataset
from rfvision.datasets import DATASETS
from rfvision.datasets.pipelines import Compose

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

    def evaluate(self,
                 results,
                 metric='mAP',
                 logger=None,
                 **kwargs):
        pass



if __name__ == '__main__':
    dataset = FreiHandDataset(data_root='/disk2/data/hand3d/FreiHAND/FreiHAND_pub_v1')
    sample = dataset[0]
