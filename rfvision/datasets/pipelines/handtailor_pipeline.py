import numpy as np
import random
from torchvision.transforms import functional
import cv2
import robotflow.rflib
from robotflow.rflearner.bricks.utils.handtailor_utils import (get_affine_transform,
                    gen_cam_param, gen_heatmap, transform_coords, DEPTH_RANGE, DEPTH_MIN)
from robotflow.rflearner.datasets import PIPELINES


@PIPELINES.register_module()
class HandTailorPipeline:
    def __init__(self,
                 num_joints=21,
                 img_shape=(256, 256),
                 heatmap_shape=(64, 64),
                 heatmap_sigma=2,
                 test_mode=False):
        self.rand = np.random.RandomState(seed=random.randint(0, 1024))
        self.test_mode = test_mode
        self.img_shape = img_shape
        self.heatmap_shape = heatmap_shape
        self.num_joints = num_joints
        self.heatmap_sigma = heatmap_sigma

    def get_random_rotmat(self,):
        rot = self.rand.uniform(low=-self.max_rot, high=self.max_rot)
        rot_mat = np.float32([
            [np.cos(rot), -np.sin(rot), 0],
            [np.sin(rot),  np.cos(rot), 0],
            [0,            0,           1]],)
        return rot_mat


    def normalize(self, img, mean=[0.5, 0.5, 0.5], std=[1, 1, 1]):
        img = functional.to_tensor(img).float()
        img = functional.normalize(img, mean, std)
        return img

    def get_joints_uv_center(self, joints_uv):
        min_u, min_v = joints_uv.min(0)
        max_u, max_v = joints_uv.max(0)
        c_u = int((max_u + min_u) / 2)
        c_v = int((max_v + min_v) / 2)
        joints_center_uv = np.int32([c_u, c_v])
        return joints_center_uv
    def get_joints_uv_scale(self, joints_uv, scale_factor=2):
        """
        Retreives the size of the square we want to crop by taking the
        maximum of vertical and horizontal span of the hand and multiplying
        it by the scale_factor to add some padding around the hand
        """
        min_x, min_y = joints_uv.min(0)
        max_x, max_y = joints_uv.max(0)
        delta_x = max_x - min_x
        delta_y = max_y - min_y
        max_delta = max(delta_x, delta_y)
        scale = max_delta * scale_factor
        return scale

    def get_joints_uv(self, joints_3d, K):
        joints_uv = joints_3d.dot(K.T)
        joints_uv = joints_uv[:, :2] / joints_uv[:, 2].reshape(-1, 1)
        return joints_uv

    def get_joints_d(self, joints_3d):
        joint_bone = np.linalg.norm(joints_3d[9] - joints_3d[0]) # int
        joint_root = joints_3d[9] # shape : (3,)
        joints_d = (joints_3d[:, 2] - joint_root[2]) / joint_bone # shape : (21,)
        joints_d = (joints_d - DEPTH_MIN) / DEPTH_RANGE # shape : (21,)
        return joints_d.reshape(-1, 1), np.array(joint_bone), np.array(joint_root)

    def aug_center(self, center, scale ,aug_center_factor = 0.1):
        center_offsets = (aug_center_factor * scale * self.rand.uniform(low=-1, high=1, size=2))
        center = center + center_offsets.astype(int)
        return center

    def aug_scale(self, scale, aug_scale_factor = 0.1):
        scale *= np.clip(aug_scale_factor * self.rand.randn() + 1,
                         1 - aug_scale_factor,
                         1 + aug_scale_factor,)
        return scale

    def __call__(self, results):
        img = results['img']
        joints_xyz = results['joints_xyz']
        K = results['K']
        # get some shared infos
        H, W = results['img'].shape[:2]
        joints_uv = self.get_joints_uv(joints_xyz, K)
        center_uv = self.get_joints_uv_center(joints_uv)
        scale_uv = self.get_joints_uv_scale(joints_uv)
        if self.test_mode == False:
            center_uv = self.aug_center(center_uv, scale_uv)
            scale_uv = self.aug_scale(scale_uv)
            rot = self.rand.uniform(low=-np.pi, high=np.pi)

            rot_mat = np.float32([
                [np.cos(rot), -np.sin(rot), 0],
                [np.sin(rot), np.cos(rot), 0],
                [0, 0, 1]], )
            affinetrans, post_rot_trans = get_affine_transform(center_uv, scale_uv, [H, W], rot=rot)
            joints_uv = transform_coords(joints_uv, affinetrans)
            joints_xyz = rot_mat.dot(joints_xyz.transpose(1, 0)).transpose()
            joints_d, joint_bone, joint_root = self.get_joints_d(joints_xyz)
            # normalize joints_uv
            joints_uv_normalized = joints_uv / np.array([W, H])
            # get uvd
            joints_uvd = np.hstack((joints_uv_normalized, joints_d))
            K_new = gen_cam_param(joints_xyz, joints_uv, mode='persp')

            # img
            affinetrans_for_cv = affinetrans[:2, :]  # shape (2, 3)
            img_processed = cv2.warpAffine(img, affinetrans_for_cv, (W, H))
            img_processed = robotflow.rflib.impad(img_processed, shape=self.img_shape)
            img_processed = self.normalize(img_processed)

            # heatmap
            heatmap = np.zeros((self.num_joints, self.heatmap_shape[0], self.heatmap_shape[1]), dtype='float32')
            heatmap_weight = np.ones((self.num_joints, 1), dtype='float32')
            joints_uv_for_heatmap = np.int32(joints_uv_normalized * self.heatmap_shape)
            for i in range(self.num_joints):
                heatmap[i] = gen_heatmap(heatmap[i], joints_uv_for_heatmap[i], self.heatmap_sigma)

            results['joints_uvd'] = joints_uvd
            results['joints_uv'] = joints_uv
            results['heatmap'] = heatmap
            results['heatmap_weight'] = heatmap_weight
            results['img'] = img_processed
            results['K'] = K_new
            results['joint_bone'] = joint_bone
            results['joint_root'] = joint_root
            return results
        else:
            rot = 0
            affinetrans, post_rot_trans = get_affine_transform(center_uv, scale_uv, [H, W], rot=rot)
            joints_uv = transform_coords(joints_uv, affinetrans)
            joints_d, joint_bone, joint_root = self.get_joints_d(joints_xyz)
            joints_uvd = np.hstack((joints_uv, joints_d))

            K_new = post_rot_trans.dot(K)

            # img
            affinetrans_for_cv = affinetrans[:2, :]  # shape (2, 3)
            img_processed = cv2.warpAffine(img, affinetrans_for_cv, (W, H))
            img_processed = robotflow.rflib.impad(img_processed, shape=self.img_shape)
            img_processed = self.normalize(img_processed)

            # heatmap
            joints_uv_normalized = joints_uv / np.array([W, H])

            heatmap = np.zeros((self.num_joints, self.heatmap_shape[0], self.heatmap_shape[1]), dtype='float32')
            heatmap_weight = np.ones((self.num_joints, 1), dtype='float32')
            joints_uv_for_heatmap = np.int32(joints_uv_normalized * self.heatmap_shape)
            for i in range(self.num_joints):
                heatmap[i] = gen_heatmap(heatmap[i], joints_uv_for_heatmap[i], self.heatmap_sigma)

            results['joints_uvd'] = joints_uvd
            results['joints_uv'] = joints_uv
            results['heatmap'] = heatmap
            results['heatmap_weight'] = heatmap_weight
            results['img'] = img_processed
            results['K'] = K_new
            results['joint_bone'] = joint_bone
            results['joint_root'] = joint_root
            return results
