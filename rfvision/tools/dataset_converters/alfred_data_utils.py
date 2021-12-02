# Copyright (c) OpenMMLab. All rights reserved.
import rflib
import numpy as np
from concurrent import futures as futures
from os import path as osp


class AlfredData(object):
    """alfred data.

    Generate alfred infos for alfred_converter.

    Args:
        root_path (str): Root path of the raw data.
        split (str): Set split type of the data. Default: 'train'.
    """

    def __init__(self, root_path, split='train'):
        self.root_dir = root_path
        self.split = split
        self.split_dir = osp.join(root_path)
        self.classes = [
            'Apple',
            'Bowl',
            'Bread',
            'ButterKnife',
            'Cup',
            'DishSponge',
            'Egg',
            'Fork',
            'Knife',
            'Ladle',
            'Lettuce',
            'Mug',
            'Pan',
            'PepperShaker',
            'Plate',
            'Potato',
            'Pot',
            'SaltShaker',
            'SoapBottle',
            'Spatula',
            'Spoon',
            'Tomato',
            'Box',
            'CreditCard',
            'KeyChain',
            'Laptop',
            'Pillow',
            'RemoteControl',
            'Statue',
            'Vase',
            'Candle',
            'Cloth',
            'HandTowel',
            'Plunger',
            'ScrubBrush',
            'SoapBar',
            'SprayBottle',
            'ToiletPaper',
            'Towel',
            'Newspaper',
            'Watch',
            'Book',
            'CellPhone',
            'WateringCan',
            'Glassbottle',
            'PaperTowelRoll',
            'WineBottle',
            'Pencil',
            'Kettle',
            'Boots',
            'TissueBox',
            'Pen',
            'AlarmClock',
            'BasketBall',
            'CD',
            'TeddyBear',
            'TennisRacket',
            'BaseballBat',
            'Footstool'
        ]
        self.cat2label = {cat: self.classes.index(cat) for cat in self.classes}
        self.label2cat = {self.cat2label[t]: t for t in self.cat2label}

        assert split in ['train', 'val', 'test']
        if split == 'train':
            split_file = osp.join(self.root_dir, 'meta_data/train.txt')
            rflib.check_file_exist(split_file)
            self.sample_id_list = rflib.list_from_file(split_file)
            self.test_mode = False
        elif split == 'val':
            split_file1 = osp.join(self.root_dir, 'meta_data/valid_seen.txt')
            rflib.check_file_exist(split_file1)
            split_file2 = osp.join(self.root_dir, 'meta_data/valid_unseen.txt')
            rflib.check_file_exist(split_file2)
            self.sample_id_list = rflib.list_from_file(
                split_file1) + rflib.list_from_file(split_file2)
            self.test_mode = True
        elif split == 'test':
            split_file1 = osp.join(self.root_dir, 'meta_data/tests_seen.txt')
            rflib.check_file_exist(split_file1)
            split_file2 = osp.join(self.root_dir, 'meta_data/tests_unseen.txt')
            rflib.check_file_exist(split_file2)
            self.sample_id_list = rflib.list_from_file(
                split_file1) + rflib.list_from_file(split_file2)
            self.test_mode = True

    def __len__(self):
        return len(self.sample_id_list)

    def get_label(self, idx):
        label_file = osp.join(self.root_dir, 'alfred_instance_data',
                              f'{idx}_label.npy')
        rflib.check_file_exist(label_file)
        return np.load(label_file)

    def get_unaligned_bbox(self, idx):
        bbox_file = osp.join(self.root_dir, 'alfred_instance_data',
                             f'{idx}_bbox.npy')
        rflib.check_file_exist(bbox_file)
        return np.load(bbox_file)

    def get_infos(self, num_workers=4, has_label=True, sample_id_list=None):
        """Get data infos.

        This method gets information from the raw data.

        Args:
            num_workers (int): Number of threads to be used. Default: 4.
            has_label (bool): Whether the data has label. Default: True.
            sample_id_list (list[int]): Index list of the sample.
                Default: None.

        Returns:
            infos (list[dict]): Information of the raw data.
        """

        def process_single_scene(sample_idx):
            sample_idx = sample_idx.split('/')[-1]
            info = dict()
            pc_info = {'num_features': 6, 'lidar_idx': sample_idx}
            info['point_cloud'] = pc_info
            pts_filename = osp.join(self.root_dir, 'alfred_instance_data',
                                    f'{sample_idx}_vert.npy')
            points = np.load(pts_filename)
            assert np.isnan(points).any() == False
            assert points.shape[1] == 6
            rflib.mkdir_or_exist(osp.join(self.root_dir, 'points'))
            points.tofile(
                osp.join(self.root_dir, 'points', f'{sample_idx}.bin'))
            info['pts_path'] = osp.join('points', f'{sample_idx}.bin')

            if has_label:
                annotations = {}
                box_label = self.get_label(sample_idx)
                bbox = self.get_unaligned_bbox(sample_idx)
                assert np.isnan(bbox).any() == False
                annotations['gt_num'] = box_label.shape[0]
                if annotations['gt_num'] != 0:
                    classes = box_label
                    annotations['name'] = np.array([
                        classes[i] for i in range(annotations['gt_num'])
                    ])
                    # default names are given to aligned bbox for compatibility
                    # we also save unaligned bbox info with marked names
                    annotations['location'] = bbox[:, :3]
                    annotations['dimensions'] = bbox[:, 3:6]
                    annotations['gt_boxes_upright_depth'] = bbox
                    annotations['unaligned_location'] = bbox[:, :3]
                    annotations['unaligned_dimensions'] = bbox[:, 3:6]
                    annotations[
                        'unaligned_gt_boxes_upright_depth'] = bbox
                    annotations['index'] = np.arange(
                        annotations['gt_num'], dtype=np.int32)
                    annotations['class'] = np.array([
                        self.cat2label[classes[i]]
                        for i in range(annotations['gt_num'])
                    ])
                    assert np.isnan(annotations['class']).any() == False
                info['annos'] = annotations
            return info

        sample_id_list = sample_id_list if sample_id_list is not None \
            else self.sample_id_list
        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = executor.map(process_single_scene, sample_id_list)
        return list(infos)
