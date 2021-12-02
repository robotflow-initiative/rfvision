import collections
import cv2
import open3d as o3d
import torch
import numpy as np


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
    # pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    return pcd


class CreatePointData(object):
    def __init__(self, downsample_voxel=0.005):
        self.downsample_voxel = downsample_voxel

    def __call__(self, results):
        x1, y1, x2, y2 = results['bbox']
        color_image = o3d.io.read_image(results['color_path'])
        depth_image = o3d.io.read_image(results['depth_path'])
        img_height, img_width, _ = np.array(color_image).shape
        bbox_crop = np.zeros((img_height, img_width, 3)).astype(np.uint8)
        bbox_crop[y1:y2, x1:x2, :] = 1
        camera_intrinsic = o3d.io.read_pinhole_camera_intrinsic(results['camera_intrinsic_path'])

        pc = rgbd2pc(color_image * bbox_crop, depth_image * bbox_crop[:, :, 0], camera_intrinsic)
        if self.downsample_voxel > 0:
            pc = o3d.geometry.PointCloud.voxel_down_sample(pc, self.downsample_voxel)
        pts = np.array(pc.points)
        pts_feature = np.array(pc.colors)
        n_total_points = pts.shape[0]

        results['pts'] = pts.astype(np.float32)
        results['pts_feature'] = pts_feature.astype(np.float32)
        results['n_total_points'] = n_total_points

        return results


class DownSample(object):
    def __init__(self, num_points=1024, point_norm=True):
        self.num_points = num_points
        self.point_norm = point_norm

    def __call__(self, results):
        if results['n_total_points'] < self.num_points:
            tile_n = int(self.num_points / results['n_total_points']) + 1
            results['n_total_points'] = tile_n * results['n_total_points']
            results['pts'] = np.concatenate([results['pts']] * tile_n, axis=0)
            results['pts_feature'] = np.concatenate([results['pts_feature']] * tile_n, axis=0)
        perm = np.random.permutation(results['n_total_points'])
        results['pts'] = results['pts'][perm[:self.num_points]]
        results['pts_feature'] = results['pts_feature'][perm[:self.num_points]]
        if self.point_norm:
            results['points_mean'] = results['pts'].mean(axis=0)
            results['pts'] -= results['points_mean']

        return results



