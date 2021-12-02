from .loading3d import LoadPointsFromFile
from rfvision.datasets.builder import PIPELINES
import numpy as np
import rflib
@PIPELINES.register_module()
class LoadPointsFromFilePointFormer(LoadPointsFromFile):
    def _load_points(self, pts_filename):
        # pts_filename = './data/alfred/alfred_instance_data/' \
        #     + pts_filename.split('/')[-1][:-4] + '_vert.npy'
        rflib.check_file_exist(pts_filename)
        if pts_filename.endswith('.npy'):
            points = np.load(pts_filename)
        else:
            points = np.fromfile(pts_filename, dtype=np.float32)
        return points