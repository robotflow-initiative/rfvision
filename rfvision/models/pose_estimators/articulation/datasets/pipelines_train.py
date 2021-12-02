import os
import copy
import open3d as o3d
import numpy as np
import pycocotools.mask as maskUtils
from .utils import point_3d_offset_joint

INSTANCE_CLASSES = ['BG', 'box', 'stapler', 'cutter', 'drawer', 'scissor']
PART_CLASSES = {'box': ['BG', 'base_link', 'link1'],
                'stapler': ['BG', 'base_link', 'link1'],
                'cutter': ['BG', 'base_link', 'link1'],
                'drawer': ['BG', 'base_link', 'link1', 'link2', 'link3'],
                'scissor': ['BG', 'link1', 'link2']}

epsilon = 10e-8
thres_r = 0.2


def rgbd2pc(rgb_path, depth_path, camera_intrinsic):
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
    return pcd



class CreatePointDataTrain(object):
    def __init__(self, downsample_voxel=0.005):
        self.downsample_voxel = downsample_voxel

    def __call__(self, results):
        n_parts = results['n_parts']
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

        color_image = o3d.io.read_image(os.path.join(img_prefix, 'color', results['color_path']))
        depth_image = o3d.io.read_image(os.path.join(img_prefix, 'depth', results['depth_path']))
        camera_intrinsic = o3d.io.read_pinhole_camera_intrinsic(results['camera_intrinsic_path'])

        for j in range(1, n_parts):
            # color = copy.copy(color_image)
            # depth = copy.copy(depth_image)

            part_seg = instance_info['links'][j-1]['segmentation']

            try:
                rle = maskUtils.frPyObjects(part_seg, img_height, img_width)
                part_mask = np.sum(maskUtils.decode(rle), axis=2).clip(max=1).astype(np.uint8)

                part_color = color_image * np.repeat(part_mask[..., np.newaxis], 3, 2)
                part_depth = depth_image * part_mask
                part_masks.append(part_mask)

                part_pc = rgbd2pc(part_color * mask_crop, part_depth * mask_crop[:, :, 0], camera_intrinsic)
                if self.downsample_voxel > 0:
                    part_pc = o3d.geometry.PointCloud.voxel_down_sample(part_pc, self.downsample_voxel)

                parts_pts[j] = np.array(part_pc.points)
                parts_pts_feature[j] = np.array(part_pc.colors)

                part_tran = np.array(instance_info['links'][j-1]['transformation'])
                part_pc_copy = copy.deepcopy(part_pc)
                part_coord = part_pc_copy.transform(np.linalg.inv(np.array(part_tran)))
                part_coord = np.array(part_coord.points)
                parts_gts[j] = part_coord
                if 'label_map' in results.keys():
                    parts_cls[j] = results['label_map'][j] * np.ones((parts_pts[j].shape[0]), dtype=np.float32)
                else:
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
        bg_pc = rgbd2pc(bg_color * mask_crop, bg_depth * mask_crop[:, :, 0], camera_intrinsic)
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

        results['parts_pts'] = parts_pts
        results['parts_pts_feature'] = parts_pts_feature
        results['parts_gts'] = parts_gts
        results['parts_cls'] = parts_cls
        results['parts_parent_joint'] = parts_parent_joint
        results['parts_child_joint'] = parts_child_joint
        results['n_total_points'] = n_total_points

        return results


class LoadArtiNOCSData(object):

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


class LoadArtiJointData(object):
    def __init__(self):
        self.joint_type_dict = {'prismatic': 1, 'revolute': 2}

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
                joint_index[j].append(results['parts_parent_joint'][j])
                # plot_arrows(results['nocs_g'][j], offset_arr, [joint_P0, joint_l], title_name='NOCS {} to joint {}'.format(j, j))

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
                    joint_index[j].append(m)
                    # plot_arrows(results['nocs_g'][j], offset_arr, [joint_P0, joint_l], title_name='NOCS {} to joint {}'.format(j, m))

        results['parts_offset_joint'] = parts_offset_joint
        results['parts_joints'] = parts_joints
        results['joint_index'] = joint_index
        results['joint_params'] = joint_params

        return results

    def __repr__(self):
        return self.__class__.__name__


class CreateArtiJointGT(object):

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
                if 'label_map' in results.keys():
                    joint_cls[j][idc] = results['label_map'][results['joint_index'][j][k]]
                else:
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


class DownSampleTrain(object):
    def __init__(self, num_points=1024, linspace=False, point_norm=True):
        self.num_points = num_points
        self.linspace = linspace
        self.point_norm = point_norm

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

        # if results['n_total_points'] > self.num_points:
        mask_array = np.zeros([self.num_points, results['n_max_parts']], dtype=np.float32)
        if self.linspace:
            perm = np.linspace(0, results['n_total_points']-1, self.num_points).astype(np.uint16)
        else:
            perm = np.random.permutation(results['n_total_points'])
        results['parts_cls'] = results['parts_cls'][perm[:self.num_points]].reshape(self.num_points, 1).astype(np.float32)

        # results['parts_pts'] = results['parts_pts'][perm[:self.num_points]] * results['norm_factors'][0]
        results['parts_pts'] = results['parts_pts'][perm[:self.num_points]]
        if self.point_norm:
            results['points_mean'] = results['parts_pts'].mean(axis=0)
            results['parts_pts'] -= results['points_mean']
        results['parts_pts_feature'] = results['parts_pts_feature'][perm[:self.num_points]]
        results['offset_heatmap'] = results['offset_heatmap'][perm[:self.num_points]].reshape(self.num_points, 1).astype(np.float32)
        results['offset_unitvec'] = results['offset_unitvec'][perm[:self.num_points]].astype(np.float32)
        results['joint_orient'] = results['joint_orient'][perm[:self.num_points]].astype(np.float32)
        results['joint_cls'] = results['joint_cls'][perm[:self.num_points]].reshape(self.num_points, 1).astype(np.float32)
        if 'part_relation' in results.keys():
            results['part_relation'] = results['part_relation'][perm[:self.num_points]]
        # print('joint_cls_arr has shape: ', joint_cls_arr.shape)
        joint_cls_mask = np.zeros((results['joint_cls'].shape[0]), dtype=np.float32).reshape(self.num_points, 1)
        id_valid = np.where(results['joint_cls'] > 0)[0]
        joint_cls_mask[id_valid] = 1.0
        mask_array[np.arange(self.num_points), results['parts_cls'][:, 0].astype(np.int8)] = 1.0
        results['mask_array'] = mask_array
        id_object = np.where(results['parts_cls'] > 0)[0]

        if results['nocs_p'][0] is not None:
            results['nocs_p'] = results['nocs_p'][perm[:self.num_points]]
        if results['nocs_g'][0] is not None:
            results['nocs_g'] = results['nocs_g'][perm[:self.num_points]]

        results['joint_cls_mask'] = joint_cls_mask
        return results

    def __repr__(self):
        return self.__class__.__name__
