from torch.utils.data import Dataset
import numpy as np
import random
from .pipelines import Compose
from .builder import DATASETS


@DATASETS.register_module()
class YCBVideoDataset(Dataset):
    CLASSES = ('002_master_chef_can',
               '003_cracker_box',
               '004_sugar_box',
               '005_tomato_soup_can',
               '006_mustard_bottle',
               '007_tuna_fish_can',
               '008_pudding_box',
               '009_gelatin_box',
               '010_potted_meat_can',
               '011_banana',
               '019_pitcher_base',
               '021_bleach_cleanser',
               '024_bowl',
               '025_mug',
               '035_power_drill',
               '036_wood_block',
               '037_scissors',
               '040_large_marker',
               '051_large_clamp',
               '052_extra_large_clamp',
               '061_foam_brick',
               )

    def __init__(self, ann_file,
                 pipeline,
                 img_prefix,
                 noise_trans=0.03,
                 refine=False,
                 add_noise=True,
                 num_pt=1000,
                 test_mode=False,
                 **kwargs):
        self.path = ann_file
        self.num_pt = num_pt
        self.img_prefix = img_prefix
        self.add_noise = add_noise
        self.noise_trans = noise_trans
        self.test_mode = test_mode
        self.list = []
        self.real = []
        self.syn = []
        input_file = open(self.path)
        while 1:
            input_line = input_file.readline()
            if not input_line:
                break
            if input_line[-1:] == '\n':
                input_line = input_line[:-1]
            if input_line[:5] == 'data/':
                self.real.append(input_line)
            else:
                self.syn.append(input_line)
            self.list.append(input_line)
        input_file.close()

        self.length = len(self.list)
        self.len_real = len(self.real)
        self.len_syn = len(self.syn)

        class_id = 1
        self.cld = {}
        for CLASS in self.CLASSES:
            input_file = open('{0}/models/{1}/points.xyz'.format(self.img_prefix, CLASS))
            self.cld[class_id] = []
            while 1:
                input_line = input_file.readline()
                if not input_line:
                    break
                input_line = input_line[:-1].split(' ')
                self.cld[class_id].append([float(input_line[0]), float(input_line[1]), float(input_line[2])])
            self.cld[class_id] = np.array(self.cld[class_id])
            input_file.close()

            class_id += 1

        self.cam_cx_1 = 312.9869
        self.cam_cy_1 = 241.3109
        self.cam_fx_1 = 1066.778
        self.cam_fy_1 = 1067.487

        self.cam_cx_2 = 323.7872
        self.cam_cy_2 = 279.6921
        self.cam_fx_2 = 1077.836
        self.cam_fy_2 = 1078.189

        if not self.test_mode:
            self._set_group_flag()

        self.pipeline = Compose(pipeline)

    def pre_pipeline(self, results):
        results['img_prefix'] = self.img_prefix

        color_path = '{0}/{1}-color.png'.format(self.img_prefix, results['sample_name'])
        depth_path = '{0}/{1}-depth.png'.format(self.img_prefix, results['sample_name'])
        label_path = '{0}/{1}-label.png'.format(self.img_prefix, results['sample_name'])
        meta_path = '{0}/{1}-meta.mat'.format(self.img_prefix, results['sample_name'])

        if results['sample_name'][:8] != 'data_syn' and int(results['sample_name'][5:9]) >= 60:
            cam_cx = self.cam_cx_2
            cam_cy = self.cam_cy_2
            cam_fx = self.cam_fx_2
            cam_fy = self.cam_fy_2
        else:
            cam_cx = self.cam_cx_1
            cam_cy = self.cam_cy_1
            cam_fx = self.cam_fx_1
            cam_fy = self.cam_fy_1

        cam_params = (cam_cx, cam_cy, cam_fx, cam_fy)
        if not self.test_mode:
            random_seeds = [random.choice(self.syn) for _ in range(5)]
        else:
            random_seeds = []
        real_seed = random.choice(self.real)

        results.update(color_path=color_path,
                       depth_path=depth_path,
                       label_path=label_path,
                       meta_path=meta_path,
                       cam_params=cam_params,
                       random_seeds=random_seeds,
                       real_seed=real_seed,
                       cld=self.cld)

    def prepare_train_sample(self, idx):
        sample_name = self.list[idx]
        results = dict(sample_name=sample_name)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_test_sample(self, idx):
        sample_name = self.sample_list[idx]
        results = dict(sample_name=sample_name)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_sample(idx)
        else:
            return self.prepare_train_sample(idx)

    def __len__(self):
        return self.length

    def _set_group_flag(self):
        """Set flag to 1
        """
        self.flag = np.ones(len(self), dtype=np.uint8)

    def get_sym_list(self):
        return self.symmetry_obj_idx

    def get_num_points_mesh(self):
        if self.refine:
            return self.num_pt_mesh_large
        else:
            return self.num_pt_mesh_small
