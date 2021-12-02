import json
from torch.utils.data import Dataset
from .pipelines_test import *
from .pipelines_train import *
from rfvision.datasets import DATASETS
from rfvision.datasets.pipelines import ToTensor, Compose, Collect

train_keys = ['parts_pts', 'parts_pts_feature', 'parts_cls', 'mask_array',
                    'nocs_p', 'nocs_g', 'offset_heatmap',
                    'offset_unitvec', 'joint_orient', 'joint_cls',
                    'joint_cls_mask', 'joint_params']
train_pipelines = [CreatePointDataTrain(),
                   LoadArtiNOCSData(),
                   LoadArtiJointData(),
                   CreateArtiJointGT(),
                   DownSampleTrain(),
                   ToTensor(keys=train_keys),
                   Collect(keys=train_keys,
                         meta_keys=['img_prefix', 'sample_name', 'norm_factors', 'corner_pts',
                                    'joint_ins'])]

test_keys = ['pts', 'pts_feature']
test_pipelines = [CreatePointData(), DownSample(), ToTensor(keys=test_keys)]




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
class ArticulationDataset(Dataset):
    CLASSES = None

    def __init__(self,
                 ann_file,
                 img_prefix,
                 intrinsics_path,
                 n_max_parts=13,
                 is_gen=False,
                 test_mode=False,
                 **kwargs):

        if test_mode==True:
            pipeline = test_pipelines
        else:
            pipeline = train_pipelines


        self.is_gen = is_gen
        self.n_max_parts = n_max_parts
        self.img_prefix = img_prefix
        self.camera_intrinsic_path = intrinsics_path
        self.annotation_path = os.path.join(self.img_prefix, 'annotations')

        self.sample_list = []
        with open(ann_file, 'r') as f:
            for line in f:
                self.sample_list.append(line.rstrip('\n'))

        self.norm_factors, self.corner_pts = self.fetch_factors_nocs()
        self.all_joint_ins = self.fetch_joints_params()

        self._set_group_flag()

        self.pipeline = Compose(pipeline)
        print('total samples: {}'.format(len(self)))

    def pre_pipeline(self, results):
        results['img_prefix'] = self.img_prefix
        filename, instance_id = results['sample_name'].split('_')
        instance_id = int(instance_id)
        data_info = copy.copy(json.load(open(os.path.join(self.annotation_path, filename + '.json'))))
        results['camera_intrinsic_path'] = self.camera_intrinsic_path

        instance_info = data_info['instances'][instance_id]
        color_path = os.path.join(self.img_prefix, 'color', data_info['color_path'])
        depth_path = os.path.join(self.img_prefix, 'depth', data_info['depth_path'])
        img_width = data_info['width']
        img_height = data_info['height']
        bbox = instance_info['bbox']

        urdf_id = instance_info['urdf_id']
        joint_ins = self.all_joint_ins[urdf_id]
        category_id = instance_info['category_id']
        category_name = INSTANCE_CLASSES[category_id]

        n_parts = len(PART_CLASSES[category_name])
        label_map = PART_LABEL_MAPS[INSTANCE_CLASSES[category_id]]
        norm_factors = self.norm_factors[urdf_id]
        corner_pts = self.corner_pts[urdf_id]

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
                # we need to transform (x,y,z) to (y,z,x) because unity coordinate system is different from our camera system
                joint_ins[urdf_meta['id']]['xyz'].append([y, z, x])
                r, p, y = joint_rpy[n]
                joint_ins[urdf_meta['id']]['axis'].append([p, y, r])
                joint_ins[urdf_meta['id']]['type'].append(joint_types[n])
                joint_ins[urdf_meta['id']]['parent'].append(joint_parents[n])
                joint_ins[urdf_meta['id']]['child'].append(joint_children[n])

        return joint_ins
