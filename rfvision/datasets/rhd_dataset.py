import os
import robotflow.rflib
import numpy as np
from torch.utils.data import Dataset
from robotflow.rflearner.datasets import DATASETS
from robotflow.rflearner.datasets.pipelines import Compose



@DATASETS.register_module()
class RHDDataset(Dataset):
    CLASSES = None
    def __init__(self,
                 data_root,
                 pipeline,
                 split='train',
                 **kwargs
                 ):
        assert split in ['train', 'test', 'val']
        # initalize some infos
        self.data_root = data_root
        self.split = split
        # loading
        if self.split == 'train':
            self.split_root = os.path.join(os.path.join(self.data_root, 'training'))
            self.annos = robotflow.rflib.load(os.path.join(self.data_root, 'training', 'anno_training.pickle'))
        else:
            self.split_root = os.path.join(os.path.join(self.data_root, 'evaluation'))
            self.annos = robotflow.rflib.load(os.path.join(self.data_root, 'evaluation', 'anno_training.pickle'))

        self.pipeline = Compose(pipeline)
        self._set_group_flag()

    def __len__(self):
        return len(self.annos)

    def __getitem__(self, index):
        rgb = robotflow.rflib.imread(os.path.join(self.split_root, 'color', '%05d' % index + '.png'))
        results = {
                   'img': rgb,
                   'uv_vis': np.float32(self.annos[index]['uv_vis']),
                   'joints_xyz': np.float32(self.annos[index]['xyz']),
                   'K': np.float32(self.annos[index]['K']),
                   }

        results = self.pipeline(results)
        return results

    def _set_group_flag(self):
        self.flag = np.zeros(len(self), dtype=np.uint8)

    def evaluate(self,
                 results,
                 metric='mAP',
                 logger=None):
        pass



if __name__ == '__main__':
    dataset = RHDDataset(data_root='/disk2/data/hand3d/RHD/RHD_published_v2')
    sample = dataset[0]
