'''
ARTI is a dataset created by Liu Liu (@liuliu66)
'''
import os
import numpy as np
import open3d as o3d
import robotflow.rflib as rflib
import json
from torch.utils.data import Dataset
import copy
from .pipelines import Compose
from .builder import DATASETS

@DATASETS.register_module()
class ArtiSynDataset(Dataset):
    CLASSES = None

    def __init__(self, ann_file,
                 pipeline,
                 img_prefix,
                 intrinsics_path,
                 test_mode=False,
                 domain='real',
                 n_parts=3,
                 is_gen=False,
                 **kwargs):
        self.is_gen = is_gen
        self.n_parts = n_parts
        self.img_prefix = img_prefix
        self.domain = domain
        self.test_mode = test_mode
        self.camera_intrinsic = o3d.io.read_pinhole_camera_intrinsic(intrinsics_path)
        self.annotation_path = os.path.join(self.img_prefix, 'annotations')

        self.sample_list = rflib.list_from_file(ann_file)

        self.norm_factors, self.corner_pts = self.fetch_factors_nocs()
        self.all_joint_ins = self.fetch_joints_params()

        if not self.test_mode:
            self._set_group_flag()

        self.pipeline = Compose(pipeline)

    def pre_pipeline(self, results):
        results['camera_intrinsic'] = self.camera_intrinsic
        results['img_prefix'] = self.img_prefix
        results['domain'] = self.domain

        scene, h5_file = results['sample_name'].split('/')
        filename, instance_id = h5_file.split('.h5')[0].split('_')
        instance_id = int(instance_id)
        data_info = json.load(open(os.path.join(self.annotation_path, scene, filename + '.json')))
        instance_info = data_info['instances'][instance_id]
        img_width = data_info['width']
        img_height = data_info['height']
        bbox = instance_info['bbox']

        urdf_id = instance_info['urdf_id']
        joint_ins = self.all_joint_ins[urdf_id]
        category_id = instance_info['category_id']
        norm_factors = self.norm_factors[urdf_id]
        corner_pts = self.corner_pts[urdf_id]

        results.update(dict(instance_info=instance_info,
                            img_width=img_width,
                            img_height=img_height,
                            bbox=bbox,
                            category_id=category_id,
                            joint_ins=joint_ins,
                            norm_factors=norm_factors,
                            corner_pts=corner_pts))

    def prepare_train_sample(self, idx):
        sample_name = self.sample_list[idx]
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
        return len(self.sample_list)

    def _set_group_flag(self):
        """Set flag to 1
        """
        self.flag = np.ones(len(self), dtype=np.uint8)

    def fetch_factors_nocs(self):
        norm_factors = {}
        corner_pts = {}
        urdf_metas = json.load(open(self.img_prefix + '/urdf_metas.json'))['urdf_metas']
        for urdf_meta in urdf_metas:
            norm_factors[urdf_meta['id']] = np.array(urdf_meta['norm_factors'])
            corner_pts[urdf_meta['id']] = np.array(urdf_meta['corner_pts'])

        return norm_factors, corner_pts

    def fetch_joints_params(self):
        joint_ins = {}
        urdf_metas = json.load(open(self.img_prefix + '/urdf_metas.json'))['urdf_metas']
        for urdf_meta in urdf_metas:
            if urdf_meta == []:
                continue
            joint_ins[urdf_meta['id']] = dict(xyz=[], axis=[], type=[], parent=[], child=[])

            for n in range(self.n_parts - 1):
                if n == 0:
                    joint_ins[urdf_meta['id']]['xyz'].append([0., 0., 0.])
                    joint_ins[urdf_meta['id']]['axis'].append([0., 0., 0.])
                    joint_ins[urdf_meta['id']]['type'].append(None)
                    joint_ins[urdf_meta['id']]['parent'].append(None)
                    joint_ins[urdf_meta['id']]['child'].append(None)
                    continue
                x, y, z = urdf_meta['joint_xyz'][n-1][::-1]
                joint_ins[urdf_meta['id']]['xyz'].append([y, x, z])
                r, p, y = urdf_meta['joint_rpy'][n - 1][::-1]
                joint_ins[urdf_meta['id']]['axis'].append([p, -r, y])
                joint_ins[urdf_meta['id']]['type'].append(urdf_meta['joint_types'][n-1])
                joint_ins[urdf_meta['id']]['parent'].append(urdf_meta['joint_parents'][n-1])
                joint_ins[urdf_meta['id']]['child'].append(urdf_meta['joint_children'][n-1])

        return joint_ins

@DATASETS.register_module()
class ArtiRealDataset(Dataset):
    CLASSES = None

    def __init__(self, ann_file,
                 pipeline,
                 img_prefix,
                 intrinsics_path,
                 test_mode=False,
                 domain='real',
                 n_parts=3,
                 is_gen=False,
                 **kwargs):
        self.is_gen = is_gen
        self.n_parts = n_parts
        self.img_prefix = img_prefix
        self.domain = domain
        self.test_mode = test_mode
        self.camera_intrinsic = o3d.io.read_pinhole_camera_intrinsic(intrinsics_path)
        self.annotation_path = os.path.join(self.img_prefix, 'annotations')
        self.sample_list = rflib.list_from_file(ann_file)

        self.norm_factors, self.corner_pts = self.fetch_factors_nocs()
        self.all_joint_ins = self.fetch_joints_params()

        if not self.test_mode:
            self._set_group_flag()

        self.pipeline = Compose(pipeline)

    def pre_pipeline(self, results):
        results['camera_intrinsic'] = self.camera_intrinsic
        results['img_prefix'] = self.img_prefix
        results['domain'] = self.domain

        scene, video, h5_file = results['sample_name'].split('/')
        filename, instance_id = h5_file.split('.h5')[0].split('_')
        instance_id = int(instance_id)
        data_info = json.load(open(os.path.join(self.annotation_path, scene, video, filename + '.json')))
        instance_info = data_info['instances'][instance_id]

        img_width = data_info['width']
        img_height = data_info['height']
        bbox = instance_info['bbox']

        urdf_id = instance_info['urdf_id']
        joint_ins = self.all_joint_ins[urdf_id]
        category_id = instance_info['category_id']
        norm_factors = self.norm_factors[urdf_id]
        corner_pts = self.corner_pts[urdf_id]

        results.update(dict(instance_info=instance_info,
                            img_width=img_width,
                            img_height=img_height,
                            bbox=bbox,
                            category_id=category_id,
                            joint_ins=joint_ins,
                            norm_factors=norm_factors,
                            corner_pts=corner_pts))

    def prepare_train_sample(self, idx):
        sample_name = self.sample_list[idx]
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
        return len(self.sample_list)

    def _set_group_flag(self):
        """Set flag to 1
        """
        self.flag = np.ones(len(self), dtype=np.uint8)

    def fetch_factors_nocs(self):
        norm_factors = {}
        corner_pts = {}
        urdf_metas = json.load(open(self.img_prefix + '/urdf_metas.json'))['urdf_metas']
        for urdf_meta in urdf_metas:
            norm_factors[urdf_meta['id']] = np.array(urdf_meta['norm_factors'])
            corner_pts[urdf_meta['id']] = np.array(urdf_meta['corner_pts'])

        return norm_factors, corner_pts

    def fetch_joints_params(self):
        joint_ins = {}
        urdf_metas = json.load(open(self.img_prefix + '/urdf_metas.json'))['urdf_metas']
        for urdf_meta in urdf_metas:
            if urdf_meta == []:
                continue
            joint_ins[urdf_meta['id']] = dict(xyz=[], axis=[], type=[], parent=[], child=[])

            for n in range(self.n_parts - 1):
                if n == 0:
                    joint_ins[urdf_meta['id']]['xyz'].append([0., 0., 0.])
                    joint_ins[urdf_meta['id']]['axis'].append([0., 0., 0.])
                    joint_ins[urdf_meta['id']]['type'].append(None)
                    joint_ins[urdf_meta['id']]['parent'].append(None)
                    joint_ins[urdf_meta['id']]['child'].append(None)
                    continue
                x, y, z = urdf_meta['joint_xyz'][n-1][::-1]
                joint_ins[urdf_meta['id']]['xyz'].append([y, x, z])
                r, p, y = urdf_meta['joint_rpy'][n - 1][::-1]
                joint_ins[urdf_meta['id']]['axis'].append([p, -r, y])
                joint_ins[urdf_meta['id']]['type'].append(urdf_meta['joint_types'][n-1])
                joint_ins[urdf_meta['id']]['parent'].append(urdf_meta['joint_parents'][n-1])
                joint_ins[urdf_meta['id']]['child'].append(urdf_meta['joint_children'][n-1])

        return joint_ins
    
    
@DATASETS.register_module()
class ArtiImgDataset(Dataset):
    CLASSES = None

    def __init__(self, ann_file,
                 pipeline,
                 img_prefix,
                 intrinsics_path,
                 test_mode=False,
                 domain='real',
                 n_parts=3,
                 is_gen=False,
                 **kwargs):
        self.is_gen = is_gen
        self.n_parts = n_parts
        self.img_prefix = img_prefix
        self.domain = domain
        self.test_mode = test_mode
        self.camera_intrinsic = o3d.io.read_pinhole_camera_intrinsic(intrinsics_path)
        self.annotation_path = os.path.join(self.img_prefix, 'annotations')

        self.sample_list = rflib.list_from_file(ann_file)

        self.norm_factors, self.corner_pts = self.fetch_factors_nocs()
        self.all_joint_ins = self.fetch_joints_params()

        if not self.test_mode:
            self._set_group_flag()

        self.pipeline = Compose(pipeline)

    def pre_pipeline(self, results):
        results['camera_intrinsic'] = self.camera_intrinsic
        results['img_prefix'] = self.img_prefix
        results['domain'] = self.domain

        scene, h5_file = results['sample_name'].split('/')
        filename, instance_id = h5_file.split('.h5')[0].split('_')
        instance_id = int(instance_id)
        data_info = copy.copy(json.load(open(os.path.join(self.annotation_path, scene, filename + '.json'))))
        instance_info = data_info['instances'][instance_id]
        color_path = data_info['color_path']
        depth_path = data_info['depth_path']
        img_width = data_info['width']
        img_height = data_info['height']
        bbox = instance_info['bbox']

        urdf_id = instance_info['urdf_id']
        joint_ins = self.all_joint_ins[urdf_id]
        category_id = instance_info['category_id']
        norm_factors = self.norm_factors[urdf_id]
        corner_pts = self.corner_pts[urdf_id]

        results.update(dict(instance_info=instance_info,
                            color_path=color_path,
                            depth_path=depth_path,
                            img_width=img_width,
                            img_height=img_height,
                            bbox=bbox,
                            category_id=category_id,
                            joint_ins=joint_ins,
                            norm_factors=norm_factors,
                            corner_pts=corner_pts))

    def prepare_train_sample(self, idx):
        sample_name = self.sample_list[idx]
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
        return len(self.sample_list)

    def _rand_another(self, idx):
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    def _set_group_flag(self):
        """Set flag to 1
        """
        self.flag = np.ones(len(self), dtype=np.uint8)

    def fetch_factors_nocs(self):
        norm_factors = {}
        corner_pts = {}
        urdf_metas = json.load(open(self.img_prefix + '/urdf_metas.json'))['urdf_metas']
        for urdf_meta in urdf_metas:
            norm_factors[urdf_meta['id']] = np.array(urdf_meta['norm_factors'])
            corner_pts[urdf_meta['id']] = np.array(urdf_meta['corner_pts'])

        return norm_factors, corner_pts

    def fetch_joints_params(self):
        joint_ins = {}
        urdf_metas = json.load(open(self.img_prefix + '/urdf_metas.json'))['urdf_metas']
        for urdf_meta in urdf_metas:
            if urdf_meta == []:
                continue
            joint_ins[urdf_meta['id']] = dict(xyz=[], axis=[], type=[], parent=[], child=[])

            for n in range(self.n_parts - 1):
                if n == 0:
                    joint_ins[urdf_meta['id']]['xyz'].append([0., 0., 0.])
                    joint_ins[urdf_meta['id']]['axis'].append([0., 0., 0.])
                    joint_ins[urdf_meta['id']]['type'].append(None)
                    joint_ins[urdf_meta['id']]['parent'].append(None)
                    joint_ins[urdf_meta['id']]['child'].append(None)
                    continue
                x, y, z = urdf_meta['joint_xyz'][n-1]
                joint_ins[urdf_meta['id']]['xyz'].append([y, z, x])
                r, p, y = urdf_meta['joint_rpy'][n - 1]
                joint_ins[urdf_meta['id']]['axis'].append([p, y, r])
                joint_ins[urdf_meta['id']]['type'].append(urdf_meta['joint_types'][n-1])
                joint_ins[urdf_meta['id']]['parent'].append(urdf_meta['joint_parents'][n-1])
                joint_ins[urdf_meta['id']]['child'].append(urdf_meta['joint_children'][n-1])

        return joint_ins
