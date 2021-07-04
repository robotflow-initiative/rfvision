'''
ARTI is a dataset created by Liu Liu (@liuliu66)
'''
import os
import numpy as np
import open3d as o3d
import rflib
import json
from torch.utils.data import Dataset
import copy
from .pipelines import Compose
from .builder import DATASETS

INSTANCE_CLASSES = ('BG', 'box', 'stapler', 'cutter', 'drawer', 'scissor')
PART_CLASSES = {'box': ('BG', 'base_link', 'link1'),
                'stapler': ('BG', 'base_link', 'link1'),
                'cutter': ('BG', 'base_link', 'link1'),
                'drawer': ('BG', 'base_link', 'link1', 'link2', 'link3'),
                'scissor': ('BG', 'link1', 'link2')}
PART_LABEL_MAPS = {'box': (0, 1, 2),
                   'stapler': (0, 3, 4),
                   'cutter': (0, 5, 6),
                   'drawer': (0, 7, 8, 9, 10),
                   'scissor': (0, 11, 12)} # every category contains 0 for BG


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

        # color_path = os.path.join(scene, video, filename + '.jpg')
        # depth_path = os.path.join(scene, video, filename + '.png')

        results.update(dict(instance_info=instance_info,
                            img_width=img_width,
                            img_height=img_height,
                            bbox=bbox,
                            category_id=category_id,
                            joint_ins=joint_ins,
                            norm_factors=norm_factors,
                            corner_pts=corner_pts,
                            # color_path=color_path,
                            # depth_path=depth_path
                            ))

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
                 n_max_parts=3,
                 is_gen=False,
                 **kwargs):
        self.is_gen = is_gen
        self.n_max_parts = n_max_parts
        self.img_prefix = img_prefix
        self.domain = domain
        self.test_mode = test_mode
        if os.path.isdir(intrinsics_path):
            self.camera_intrinsic_path = dict()
            for scene in os.listdir(intrinsics_path):
                camera_intrinsic_path = os.path.join(intrinsics_path,
                                                     scene,
                                                     'camera_intrinsic.json')
                self.camera_intrinsic_path.update({scene: camera_intrinsic_path})
        else:
            self.camera_intrinsic_path = intrinsics_path
        self.annotation_path = os.path.join(self.img_prefix, 'annotations')

        self.sample_list = rflib.list_from_file(ann_file)

        self.norm_factors, self.corner_pts = self.fetch_factors_nocs()
        self.all_joint_ins = self.fetch_joints_params()

        if not self.test_mode:
            self._set_group_flag()

        self.pipeline = Compose(pipeline)

        '''
        if not self.test_mode:
            print('loading annotations...')
            self.data_infos = {}
            scenes = os.listdir(self.annotation_path)
            for scene in scenes:
                img_list = os.listdir(os.path.join(self.annotation_path, scene))
                for img in img_list:
                    name = img.split('.')[0]
                    with open(os.path.join(self.annotation_path, scene, name + '.json')) as fp:
                        self.data_infos[scene + '/' + name] = json.load(fp)
        '''

    def pre_pipeline(self, results):
        results['img_prefix'] = self.img_prefix
        results['domain'] = self.domain

        if self.domain == 'real':
            scene, video, h5_file = results['sample_name'].split('/')
        else:
            scene, h5_file = results['sample_name'].split('/')

        if isinstance(self.camera_intrinsic_path, dict):
            results['camera_intrinsic_path'] = self.camera_intrinsic_path[scene]
        else:
            results['camera_intrinsic_path'] = self.camera_intrinsic_path
        filename, instance_id = h5_file.split('.h5')[0].split('_')
        instance_id = int(instance_id)
        #if not self.test_mode:
            #data_info = self.data_infos[scene + '/' + filename]
        #else:
        if self.domain == 'real':
            data_info = copy.copy(json.load(open(os.path.join(self.annotation_path, scene, video, filename + '.json'))))
        else:
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
        category_name = INSTANCE_CLASSES[category_id]
        if category_name == 'drawer' and urdf_id not in (50, 51, 52, 53, 54, 55):
            n_parts = len(PART_CLASSES[category_name][:3])
            label_map = PART_LABEL_MAPS[INSTANCE_CLASSES[category_id]][:3]
        else:
            n_parts = len(PART_CLASSES[category_name])
            label_map = PART_LABEL_MAPS[INSTANCE_CLASSES[category_id]]
        norm_factors = self.norm_factors[urdf_id]
        corner_pts = self.corner_pts[urdf_id]
        # if 'scale' in instance_info.keys():
        #     norm_factors /= instance_info['scale']
        #     corner_pts /= instance_info['scale']

        results.update(dict(instance_info=instance_info,
                            color_path=color_path,
                            depth_path=depth_path,
                            img_width=img_width,
                            img_height=img_height,
                            bbox=bbox,
                            category_id=category_id,
                            label_map=label_map,
                            n_parts=n_parts,
                            joint_ins=joint_ins,
                            norm_factors=norm_factors,
                            corner_pts=corner_pts,
                            n_max_parts=self.n_max_parts))

    def prepare_train_sample(self, idx):
        #try:
        sample_name = self.sample_list[idx]
        results = dict(sample_name=sample_name)
            # ann_info = self.get_ann_info(idx)
            # results = dict(img_info=img_info, ann_info=ann_info)
        self.pre_pipeline(results)
        return self.pipeline(results)
        #except:
            #idx = self._rand_another(idx)
            #return self.prepare_train_sample(idx)

    def prepare_test_sample(self, idx):
        sample_name = self.sample_list[idx]
        results = dict(sample_name=sample_name)
        # img_info = self.img_infos[idx]
        # results = dict(img_info=img_info)
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
            joint_ins[urdf_meta['id']] = dict(xyz=[[0., 0., 0.]],
                                              axis=[[0., 0., 0.]],
                                              type=[None],
                                              parent=[None],
                                              child=[None])
            joint_types = urdf_meta['joint_types']
            joint_parents = urdf_meta['joint_parents']
            joint_children = urdf_meta['joint_children']
            joint_xyz = urdf_meta['joint_xyz']
            joint_rpy = urdf_meta['joint_rpy']
            assert len(joint_types) == len(joint_parents) == len(joint_children) == len(joint_xyz) == len(joint_rpy)

            num_joints = len(joint_types)
            for n in range(num_joints):
                x, y, z = joint_xyz[n]
                joint_ins[urdf_meta['id']]['xyz'].append([y, z, x])
                r, p, y = joint_rpy[n]
                joint_ins[urdf_meta['id']]['axis'].append([p, y, r])
                joint_ins[urdf_meta['id']]['type'].append(joint_types[n])
                joint_ins[urdf_meta['id']]['parent'].append(joint_parents[n])
                joint_ins[urdf_meta['id']]['child'].append(joint_children[n])

        return joint_ins