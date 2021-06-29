'''
ARTI is a dataset created by Liu Liu (@liuliu66)
'''

import os.path as osp
from collections.abc import Sequence

import rflib
import open3d as o3d
import numpy as np
import copy
import torch
import h5py
import pycocotools.mask as maskUtils

from rflib.parallel import DataContainer as DC

from rfvision.datasets.arti_utils import point_3d_offset_joint
from ..builder import PIPELINES

def estimateSimilarityUmeyama(SourceHom, TargetHom, rt_pre=None):
    # Copy of original paper is at: http://web.stanford.edu/class/cs273/refs/umeyama.pdf
    SourceCentroid = np.mean(SourceHom[:3, :], axis=1)
    TargetCentroid = np.mean(TargetHom[:3, :], axis=1)
    nPoints = SourceHom.shape[1]

    CenteredSource = SourceHom[:3, :] - np.tile(SourceCentroid, (nPoints, 1)).transpose()
    CenteredTarget = TargetHom[:3, :] - np.tile(TargetCentroid, (nPoints, 1)).transpose()

    CovMatrix = np.matmul(CenteredTarget, np.transpose(CenteredSource)) / nPoints

    if np.isnan(CovMatrix).any():
        print('nPoints:', nPoints)
        print(SourceHom.shape)
        print(TargetHom.shape)
        raise RuntimeError('There are NANs in the input.')

    U, D, Vh = np.linalg.svd(CovMatrix, full_matrices=True)
    d = (np.linalg.det(U) * np.linalg.det(Vh)) < 0.0
    if d:
        D[-1] = -D[-1]
        U[:, -1] = -U[:, -1]
    if rt_pre is not None:
        Rotation = rt_pre[:3, :3].T
    else:
        Rotation = np.matmul(U, Vh).T # Transpose is the one that works

    varP = np.var(SourceHom[:3, :], axis=1).sum()
    ScaleFact = 1/varP * np.sum(D) # scale factor
    Scales = np.array([ScaleFact, ScaleFact, ScaleFact])
    ScaleMatrix = np.diag(Scales)

    Translation  = TargetHom[:3, :].mean(axis=1) - SourceHom[:3, :].mean(axis=1).dot(ScaleFact*Rotation)

    OutTransform = np.identity(4)
    OutTransform[:3, :3] = ScaleMatrix @ Rotation.T
    OutTransform[:3, 3]  = Translation
    return Scales, Rotation, Translation, OutTransform

epsilon = 10e-8
thres_r = 0.2

INSTANCE_CLASSES = ['BG', 'box', 'stapler', 'cutter', 'drawer', 'scissor']
PART_CLASSES = {'box': ['BG', 'base_link', 'link1'],
                'stapler': ['BG', 'base_link', 'link1'],
                'cutter': ['BG', 'base_link', 'link1'],
                'drawer': ['BG', 'base_link', 'link1'],
                'scissor': ['BG', 'link1', 'link2']}


def rgbd2pc(rgb_path, depth_path, camera_intrinsic, vis=False, save_pcd=False):
    rgb_path = o3d.geometry.Image(rgb_path)
    depth_path = o3d.geometry.Image(depth_path)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_path,
                                                                    depth_path,
                                                                    1000.0,
                                                                    20.0,
                                                                    convert_rgb_to_intensity=False)


    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        camera_intrinsic
    )
    if vis:
        o3d.visualization.draw_geometries([pcd])
    if save_pcd:
        basename = osp.basename(rgb_path)
        pcd_save_name = basename.split('.png')[0] + '.pcd'
        o3d.io.write_point_cloud(pcd_save_name, pcd)

    return pcd


def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.
    """
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, Sequence) and not rflib.is_str(data):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError('type {} cannot be converted to tensor.'.format(
            type(data)))


@PIPELINES.register_module()
class CreatePointData:
    def __init__(self, downsample_voxel=0.005):
        self.downsample_voxel = downsample_voxel

    def __call__(self, results):
        category = INSTANCE_CLASSES[results['category_id']]
        n_parts = len(PART_CLASSES[category])
        instance_info = results['instance_info']

        parts_map = [[instance_info['links'][l]['link_category_id']]
                     for l in range(n_parts - 1)]
        # print(results['sample_name'], parts_map)
        joint_part = [None] + results['joint_ins']['parent']
        n_total_points = 0
        parts_pts = [None] * n_parts
        parts_pts_feature = [None] * n_parts
        parts_gts = [None] * n_parts
        parts_cls = [None] * n_parts
        parts_parent_joint = [None] * n_parts
        parts_child_joint = [None] * n_parts

        img_prefix = results['img_prefix']
        img_height = results['img_height']
        img_width = results['img_width']
        x1, y1, x2, y2 = results['bbox']

        mask_crop = np.zeros((img_height, img_width, 3)).astype(np.uint8)
        mask_crop[y1:y2, x1:x2, :] = 1
        part_masks = []

        color_image = o3d.io.read_image(osp.join(img_prefix, 'color', results['color_path']))
        depth_image = o3d.io.read_image(osp.join(img_prefix, 'depth', results['depth_path']))

        for j in range(1, n_parts):

            part_seg = instance_info['links'][j-1]['segmentation']

            try:
                rle = maskUtils.frPyObjects(part_seg, img_height, img_width)
                part_mask = np.sum(maskUtils.decode(rle), axis=2).clip(max=1).astype(np.uint8)

                part_color = color_image * np.repeat(part_mask[..., np.newaxis], 3, 2)
                part_depth = depth_image * part_mask
                part_masks.append(part_mask)

                part_pc = rgbd2pc(part_color * mask_crop, part_depth * mask_crop[:, :, 0], results['camera_intrinsic'])
                if self.downsample_voxel > 0:
                    part_pc = o3d.geometry.PointCloud.voxel_down_sample(part_pc, self.downsample_voxel)

                parts_pts[j] = np.array(part_pc.points)
                parts_pts_feature[j] = np.array(part_pc.colors)

                part_tran = np.array(instance_info['links'][j-1]['transformation'])
                part_pc_copy = copy.deepcopy(part_pc)
                part_coord = part_pc_copy.transform(np.linalg.inv(np.array(part_tran)))
                part_coord = np.array(part_coord.points)
                parts_gts[j] = part_coord
                parts_cls[j] = j * np.ones((parts_pts[j].shape[0]), dtype=np.float32)

                n_total_points += parts_pts[j].shape[0]
            except:
                parts_pts[j] = np.array([0.,0.,0.]).reshape(1, 3)
                parts_pts_feature[j] = np.array([0.,0.,0.]).reshape(1, 3)
                parts_gts[j] = np.array([0.,0.,0.]).reshape(1, 3)
                parts_cls[j] = np.array([0.])

            parts_parent_joint[j] = parts_map[j-1][0]
            parts_child_joint[j] = [ind for ind, x in enumerate(joint_part) if x == parts_map[j-1][-1]]

        fg_mask = sum(part_masks)
        bg_mask = (fg_mask == 0).astype(np.uint8)
        bg_color = color_image * np.repeat(bg_mask[..., np.newaxis], 3, 2)
        bg_depth = depth_image * bg_mask
        bg_pc = rgbd2pc(bg_color * mask_crop, bg_depth * mask_crop[:, :, 0], results['camera_intrinsic'])
        if self.downsample_voxel > 0:
            bg_pc = o3d.geometry.PointCloud.voxel_down_sample(bg_pc, self.downsample_voxel)

        parts_pts[0] = np.array(bg_pc.points)
        parts_pts_feature[0] = np.array(bg_pc.colors)
        parts_cls[0] = np.zeros((parts_pts[0].shape[0]), dtype=np.float32)
        parts_gts[0] = np.zeros((parts_pts[0].shape[0], 3))

        n_total_points += parts_pts[0].shape[0]

        if n_total_points == 0:
            print(results['bbox'])
            print(results['color_path'], instance_info['id'])
            print(p.shape[0] for p in parts_pts)

        results['n_parts'] = n_parts
        results['parts_pts'] = parts_pts
        results['parts_pts_feature'] = parts_pts_feature
        results['parts_gts'] = parts_gts
        results['parts_cls'] = parts_cls
        results['parts_parent_joint'] = parts_parent_joint
        results['parts_child_joint'] = parts_child_joint
        results['n_total_points'] = n_total_points

        return results


@PIPELINES.register_module()
class LoadArtiPointData:

    def __call__(self, results):
        category = INSTANCE_CLASSES[results['category_id']]
        n_parts = len(PART_CLASSES[category])
        instance_info = results['instance_info']
        part_names = PART_CLASSES[category][1:]

        parts_map = [[instance_info['links'][l]['link_category_id']]
                     for l in range(n_parts - 1)]
        parts_pts = [None] * n_parts
        parts_pts_feature = [None] * n_parts
        parts_gts = [None] * n_parts
        parts_cls = [None] * n_parts
        parts_parent_joint = [None] * n_parts
        parts_child_joint = [None] * n_parts

        n_total_points = 0
        joint_part = [None] + results['joint_ins']['parent']

        h5_path = osp.join(results['img_prefix'], 'hdf5', results['sample_name'])
        with h5py.File(h5_path, 'r') as handle:
            for i in range(len(part_names)):
                part_id = instance_info['links'][i]['link_category_id']
                parts_pts[part_id] = handle['parts_pts'][str(part_id)][()]
                parts_pts_feature[part_id] = handle['parts_pts_feature'][str(part_id)][()]
                parts_gts[part_id] = handle['parts_gts'][str(part_id)][()]
                parts_cls[part_id] = handle['parts_cls'][str(part_id)][()]

                parts_parent_joint[part_id] = parts_map[i][0]
                parts_child_joint[part_id] = [ind for ind, x in enumerate(joint_part) if x == parts_map[i][-1]]

                n_total_points += parts_pts[part_id].shape[0]

            parts_pts[0] = handle['parts_pts']['0'][()]
            parts_pts_feature[0] = handle['parts_pts_feature']['0'][()]
            parts_cls[0] = handle['parts_cls']['0'][()]
            parts_gts[0] = np.zeros((parts_pts[0].shape[0], 3))

            n_total_points += parts_pts[0].shape[0]
        results['n_parts'] = n_parts
        results['parts_pts'] = parts_pts
        results['parts_pts_feature'] = parts_pts_feature
        results['parts_gts'] = parts_gts
        results['parts_cls'] = parts_cls
        results['parts_parent_joint'] = parts_parent_joint
        results['parts_child_joint'] = parts_child_joint
        results['n_total_points'] = n_total_points
        return results

    def __repr__(self):
        return self.__class__.__name__


@PIPELINES.register_module()
class DownSamplePointData:
    def __init__(self, num_points=1024):
        self.num_points = num_points

    def __call__(self, results):
        if results['n_total_points'] < self.num_points:
            tile_n = int(self.num_points / results['n_total_points']) + 1
            results['n_total_points'] = tile_n * results['n_total_points']
            for j in range(results['n_parts']):
                results['parts_pts'][j] = np.concatenate([results['parts_pts'][j]] * tile_n, axis=0)
                results['parts_pts_feature'][j] = np.concatenate([results['parts_pts_feature'][j]] * tile_n, axis=0)
                results['parts_gts'][j] = np.concatenate([results['parts_gts'][j]] * tile_n, axis=0)
                results['parts_cls'][j] = np.concatenate([results['parts_cls'][j]] * tile_n, axis=0)

        points_per_part = [results['parts_pts'][i].shape[0] for i in range(results['n_parts'])]
        points_total_indexes = []
        for j in range(results['n_parts']):
            points_total_indexes.append(sum(points_per_part[:j+1]))
        points_total_indexes.insert(0, 0)
        sample_indexes = np.random.permutation(results['n_total_points'])[:self.num_points]

        for j in range(results['n_parts']):
            inds = sample_indexes[(sample_indexes >= points_total_indexes[j]) * (sample_indexes < points_total_indexes[j + 1])] \
                   - points_total_indexes[j]
            results['parts_pts'][j] = results['parts_pts'][j][inds]
            results['parts_pts_feature'][j] = results['parts_pts_feature'][j][inds]
            results['parts_gts'][j] = results['parts_gts'][j][inds]
            results['parts_cls'][j] = results['parts_cls'][j][inds]

        results['n_total_points'] = self.num_points
        return results

    def __repr__(self):
        return self.__class__.__name__


@PIPELINES.register_module()
class LoadArtiNOCSData:

    def __call__(self, results):
        nocs_p = [None] * results['n_parts']
        nocs_g = [None] * results['n_parts']
        for j in range(results['n_parts']):
            if j == 0:
                nocs_p[j] = np.zeros((results['parts_pts'][0].shape[0], 3))
                nocs_g[j] = np.zeros((results['parts_pts'][0].shape[0], 3))
                continue

            norm_factor = results['norm_factors'][j]
            norm_corner = results['corner_pts'][j]
            nocs_p[j] = (results['parts_gts'][j][:, :3] - norm_corner[0]) * norm_factor + \
                        np.array([0.5, 0.5, 0.5]).reshape(1, 3) - 0.5 * (
                                    norm_corner[1] - norm_corner[0]) * norm_factor

            norm_factor = results['norm_factors'][0]
            norm_corner = results['corner_pts'][0]
            nocs_g[j] = (results['parts_gts'][j][:, :3] - norm_corner[0]) * norm_factor + \
                        np.array([0.5, 0.5, 0.5]).reshape(1, 3) - 0.5 * (
                                    norm_corner[1] - norm_corner[0]) * norm_factor

        results['nocs_p'] = nocs_p
        results['nocs_g'] = nocs_g

        return results

    def __repr__(self):
        return self.__class__.__name__


@PIPELINES.register_module()
class LoadArtiJointData:

    def __call__(self, results):
        parts_offset_joint = [[] for _ in range(results['n_parts'])]
        parts_joints = [[] for _ in range(results['n_parts'])]
        joint_index = [[] for _ in range(results['n_parts'])]
        joint_xyz = results['joint_ins']['xyz']
        joint_rpy = results['joint_ins']['axis']
        joint_type = results['joint_ins']['type']
        joint_params = np.zeros((results['n_parts'], 7))

        for j in range(results['n_parts']):
            if j == 0:
                continue

            norm_factor = results['norm_factors'][0]
            norm_corner = results['corner_pts'][0]

            if j > 1:
                joint_P0 = np.array(joint_xyz[j - 1])
                joint_P0 = (joint_P0 - norm_corner[0]) * norm_factor + \
                           np.array([0.5, 0.5, 0.5]).reshape(1, 3) - 0.5 * (
                                   norm_corner[1] - norm_corner[0]) * norm_factor
                joint_l = np.array(joint_rpy[j - 1])
                if joint_type[j-1] == 'prismatic':
                    orth_vect = np.ones_like(np.array([0, 0, 0]).reshape(1, 3)) * 0.5 * thres_r
                else:
                    orth_vect = point_3d_offset_joint([joint_P0, joint_l], np.array([0, 0, 0]).reshape(1, 3))
                joint_params[j, 0:3] = joint_l
                joint_params[j, 6] = np.linalg.norm(orth_vect)
                joint_params[j, 3:6] = orth_vect / joint_params[j, 6]

            if results['parts_parent_joint'][j] != 1:
                joint_P0 = np.array(joint_xyz[results['parts_parent_joint'][j] - 1])
                joint_P0 = (joint_P0 - norm_corner[0]) * norm_factor + \
                           np.array([0.5, 0.5, 0.5]).reshape(1, 3) - 0.5 * (
                                   norm_corner[1] - norm_corner[0]) * norm_factor
                joint_l = np.array(joint_rpy[j - 1])
                if joint_type[j-1] == 'prismatic':
                    offset_arr = np.ones_like(results['nocs_g'][j]) * 0.5 * thres_r
                else:
                    offset_arr = point_3d_offset_joint([joint_P0, joint_l], results['nocs_g'][j])
                parts_offset_joint[j].append(offset_arr)
                parts_joints[j].append([joint_P0, joint_l])
                joint_index[j].append(results['parts_parent_joint'][j] - 1)

            if results['parts_child_joint'][j] is not None:
                for m in results['parts_child_joint'][j]:
                    joint_P0 = np.array(joint_xyz[m-1])
                    joint_P0 = (joint_P0 - norm_corner[0]) * norm_factor + \
                               np.array([0.5, 0.5, 0.5]).reshape(1, 3) - 0.5 * (
                                       norm_corner[1] - norm_corner[0]) * norm_factor
                    joint_l = np.array(joint_rpy[m-1])
                    if joint_type[j-1] == 'prismatic':
                        offset_arr = np.ones_like(results['nocs_g'][j]) * 0.5 * thres_r
                    else:
                        offset_arr = point_3d_offset_joint([joint_P0, joint_l], results['nocs_g'][j])
                    parts_offset_joint[j].append(offset_arr)
                    parts_joints[j].append([joint_P0, joint_l])
                    joint_index[j].append(m-1)

        results['parts_offset_joint'] = parts_offset_joint
        results['parts_joints'] = parts_joints
        results['joint_index'] = joint_index
        results['joint_params'] = joint_params

        return results

    def __repr__(self):
        return self.__class__.__name__


@PIPELINES.register_module()
class CreateArtiJointGT:

    def __call__(self, results):
        offset_heatmap = [None] * results['n_parts']
        offset_unitvec = [None] * results['n_parts']
        joint_orient = [None] * results['n_parts']
        joint_cls = [None] * results['n_parts']
        joint_axis_gt = [None] * results['n_parts']
        for j, offsets in enumerate(results['parts_offset_joint']):
            if j == 0:
                offset_heatmap[j] = np.zeros((results['parts_pts'][0].shape[0]))
                offset_unitvec[j] = np.zeros((results['parts_pts'][0].shape[0], 3))
                joint_orient[j] = np.zeros((results['parts_pts'][0].shape[0], 3))
                joint_cls[j] = np.zeros((results['parts_pts'][0].shape[0]))
                continue
            offset_heatmap[j] = np.zeros((results['parts_gts'][j].shape[0]))
            offset_unitvec[j] = np.zeros((results['parts_gts'][j].shape[0], 3))
            joint_orient[j] = np.zeros((results['parts_gts'][j].shape[0], 3))
            joint_cls[j] = np.zeros((results['parts_gts'][j].shape[0]))
            for k, offset in enumerate(offsets):
                heatmap = np.linalg.norm(offset, axis=1)
                unitvec = offset / (heatmap.reshape(-1, 1) + epsilon)
                idc = np.where(heatmap < thres_r)[0]
                offset_heatmap[j][idc] = 1 - heatmap[idc] / thres_r
                offset_unitvec[j][idc, :] = unitvec[idc, :]
                joint_orient[j][idc, :] = results['parts_joints'][j][k][1]
                joint_axis_gt[j] = results['parts_joints'][j][k][1]
                joint_cls[j][idc] = results['joint_index'][j][k]

        cls_arr = np.concatenate(results['parts_cls'], axis=0)
        pts_arr = np.concatenate(results['parts_pts'], axis=0)
        pts_feature_arr = np.concatenate(results['parts_pts_feature'], axis=0)
        offset_heatmap = np.concatenate(offset_heatmap, axis=0)
        offset_unitvec = np.concatenate(offset_unitvec, axis=0)
        joint_orient = np.concatenate(joint_orient, axis=0)
        joint_cls = np.concatenate(joint_cls, axis=0)
        if results['nocs_p'][0] is not None:
            p_arr = np.concatenate(results['nocs_p'], axis=0)
        if results['nocs_g'][0] is not None:
            g_arr = np.concatenate(results['nocs_g'], axis=0)

        results['parts_cls'] = cls_arr
        results['parts_pts'] = pts_arr.astype(np.float32)
        results['parts_pts_feature'] = pts_feature_arr.astype(np.float32)
        results['offset_heatmap'] = offset_heatmap
        results['offset_unitvec'] = offset_unitvec
        results['joint_axis_gt'] = joint_axis_gt
        results['joint_orient'] = joint_orient
        results['joint_cls'] = joint_cls
        results['cls_arr'] = cls_arr
        results['nocs_p'] = p_arr.astype(np.float32)
        results['nocs_g'] = g_arr.astype(np.float32)
        return results

    def __repr__(self):
        return self.__class__.__name__


@PIPELINES.register_module()
class CreatePartRelationGT:
    def __init__(self, joint_types=('none', 'revolute', 'prismatic')):
        self.joint_types = joint_types

    def __call__(self, results):
        n_parts = results['n_parts']
        n_possible_joint = n_parts ** 2

        joint_type = results['joint_ins']['type']
        joint_parent = results['joint_ins']['parent']
        joint_child = results['joint_ins']['child']
        n_joints = len(joint_parent)

        part_relation = np.zeros([results['n_total_points'], n_possible_joint], dtype=np.float32)
        for j in range(n_joints):
            if j == 0:
                continue
            parent_id = joint_parent[j]
            child_id = joint_child[j]
            type_id = self.joint_types.index(joint_type[j])

            idc = np.where(results['parts_cls'] == parent_id)[0]

            part_relation[idc, parent_id * n_parts + child_id] = type_id
            part_relation[idc, child_id * n_parts + parent_id] = type_id

        results['part_relation'] = part_relation

        return results


@PIPELINES.register_module()
class CreatePartMask:

    def __call__(self, results):
        results['parts_cls'] = results['parts_cls'].reshape(-1, 1).astype(np.float32)
        num_points = results['parts_cls'].shape[0]

        results['parts_pts'] = results['parts_pts'] * results['norm_factors'][0]

        results['offset_heatmap'] = results['offset_heatmap'].reshape(num_points, 1).astype(np.float32)
        results['offset_unitvec'] = results['offset_unitvec'].astype(np.float32)
        results['joint_orient'] = results['joint_orient'].astype(np.float32)
        results['joint_cls'] = results['joint_cls'].reshape(num_points, 1).astype(np.float32)
        joint_cls_mask = np.zeros((results['joint_cls'].shape[0]), dtype=np.float32).reshape(num_points, 1)
        id_valid = np.where(results['joint_cls'] > 0)[0]
        joint_cls_mask[id_valid] = 1.0

        mask_array = np.zeros([num_points, results['n_parts']], dtype=np.float32)
        mask_array[np.arange(num_points), results['parts_cls'][:, 0].astype(np.int8)] = 1.0
        results['mask_array'] = mask_array

        results['joint_cls_mask'] = joint_cls_mask
        return results

    def __repr__(self):
        return self.__class__.__name__


@PIPELINES.register_module()
class DownSampleArti:
    def __init__(self, num_points=1024, linspace=False):
        self.num_points = num_points
        self.linspace = linspace

    def __call__(self, results):
        if results['n_total_points'] < self.num_points:
            tile_n = int(self.num_points / results['n_total_points']) + 1
            results['n_total_points'] = tile_n * results['n_total_points']
            results['parts_cls'] = np.concatenate([results['parts_cls']] * tile_n, axis=0)
            results['parts_pts'] = np.concatenate([results['parts_pts']] * tile_n, axis=0)
            results['parts_pts_feature'] = np.concatenate([results['parts_pts_feature']] * tile_n, axis=0)
            results['offset_heatmap'] = np.concatenate([results['offset_heatmap']] * tile_n, axis=0)
            results['offset_unitvec'] = np.concatenate([results['offset_unitvec']] * tile_n, axis=0)
            results['joint_orient'] = np.concatenate([results['joint_orient']] * tile_n, axis=0)
            results['joint_cls']  = np.concatenate([results['joint_cls']] * tile_n, axis=0)
            if results['nocs_p'][0] is not None:
                results['nocs_p'] = np.concatenate([results['nocs_p']] * tile_n, axis=0)
            if results['nocs_g'][0] is not None:
                results['nocs_g'] = np.concatenate([results['nocs_g']] * tile_n, axis=0)

        mask_array = np.zeros([self.num_points, results['n_parts']], dtype=np.float32)
        if self.linspace:
            perm = np.linspace(0, results['n_total_points']-1, self.num_points).astype(np.uint16)
        else:
            perm = np.random.permutation(results['n_total_points'])
        results['parts_cls'] = results['parts_cls'][perm[:self.num_points]].reshape(self.num_points, 1).astype(np.float32)

        results['parts_pts'] = results['parts_pts'][perm[:self.num_points]]
        results['parts_pts_feature'] = results['parts_pts_feature'][perm[:self.num_points]]
        results['offset_heatmap'] = results['offset_heatmap'][perm[:self.num_points]].reshape(self.num_points, 1).astype(np.float32)
        results['offset_unitvec'] = results['offset_unitvec'][perm[:self.num_points]].astype(np.float32)
        results['joint_orient'] = results['joint_orient'][perm[:self.num_points]].astype(np.float32)
        results['joint_cls'] = results['joint_cls'][perm[:self.num_points]].reshape(self.num_points, 1).astype(np.float32)
        if 'part_relation' in results.keys():
            results['part_relation'] = results['part_relation'][perm[:self.num_points]]
        joint_cls_mask = np.zeros((results['joint_cls'].shape[0]), dtype=np.float32).reshape(self.num_points, 1)
        id_valid = np.where(results['joint_cls'] > 0)[0]
        joint_cls_mask[id_valid] = 1.0
        mask_array[np.arange(self.num_points), results['parts_cls'][:, 0].astype(np.int8)] = 1.0
        results['mask_array'] = mask_array
        if results['nocs_p'][0] is not None:
            results['nocs_p'] = results['nocs_p'][perm[:self.num_points]]
        if results['nocs_g'][0] is not None:
            results['nocs_g'] = results['nocs_g'][perm[:self.num_points]]

        results['joint_cls_mask'] = joint_cls_mask
        return results

    def __repr__(self):
        return self.__class__.__name__


@PIPELINES.register_module()
class DefaultFormatBundleArti:
    """Default formatting bundle.

    It simplifies the pipeline of formatting common fields, including "img",
    "proposals", "gt_bboxes", "gt_labels", "gt_masks" and "gt_semantic_seg".
    These fields are formatted as follows.

    - img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    - proposals: (1)to tensor, (2)to DataContainer
    - gt_bboxes: (1)to tensor, (2)to DataContainer
    - gt_bboxes_ignore: (1)to tensor, (2)to DataContainer
    - gt_labels: (1)to tensor, (2)to DataContainer
    - gt_masks: (1)to tensor, (2)to DataContainer (cpu_only=True)
    - gt_semantic_seg: (1)unsqueeze dim-0 (2)to tensor,
                       (3)to DataContainer (stack=True)
    """

    def __call__(self, results):
        if 'img' in results:
            img = np.ascontiguousarray(results['img'].transpose(2, 0, 1))
            results['img'] = DC(to_tensor(img), stack=True)
        for key in ['parts_pts', 'parts_pts_feature', 'parts_cls', 'mask_array',
                    'nocs_p', 'nocs_g', 'offset_heatmap',
                    'offset_unitvec', 'joint_orient', 'joint_cls',
                    'joint_cls_mask', 'joint_params']:
            if key not in results:
                continue
            results[key] = DC(to_tensor(results[key]), stack=True, pad_dims=1)
        return results

    def __repr__(self):
        return self.__class__.__name__