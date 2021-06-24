import os
import h5py
import numpy as np
from . import DATASETS
from .pipelines import Compose
from .custom3d import Custom3DDataset

@DATASETS.register_module()
class ShapeNetCoreV2ForSkeletonMerger:
    def __init__(self,
                 data_root,
                 pipeline,
                 split='all',
                 test_mode=False):
        assert split in ['all', 'test', 'val', 'train']
        with open(os.path.join(data_root, 'shape_names.txt')) as f:
            self.CLASSES = f.readlines()
        self.points, self.labels = (), ()
        for filename in os.listdir(data_root):
            if filename.endswith('.h5') and (split in filename or split == 'all'):
                f = h5py.File(os.path.join(data_root, filename), 'r')
                points = np.array(f['data'])
                labels = np.array(f['label'])
                self.points += (points,)
                self.labels += (labels,)
                f.close()
            else:
                continue
        self.points = np.concatenate(self.points)
        self.labels = np.concatenate(self.labels)
        self.labels_onehot = np.eye(len(self.CLASSES))[self.labels.flatten()]
        
        self._set_group_flag()
        self.pipeline = Compose(pipeline)
        
    def __len__(self,):
        return len(self.points)
    
    def __getitem__(self, index):
        results = {'points':self.points[index],
                   'labels':self.labels[index],
                   'labels_onehot':self.labels_onehot[index]
                    }
        results = self.pipeline(results)
        return results
        
    def _set_group_flag(self):
        self.flag = np.zeros(len(self), dtype=np.uint8)


