import numpy as np
import cv2
from rfvision.components.utils import (xyz2uv, affine_transform, get_K, generate_heatmap_2d)
from rfvision.datasets import PIPELINES


@PIPELINES.register_module()
class GetJointsUV:
    # Compute joints uv toward joints xyz and K.
    def get_joints_bbox(self, joints_uv):
        # get bbox info of joints_uv
        min_u, min_v = joints_uv.min(0)
        max_u, max_v = joints_uv.max(0)
        joints_bbox_uv = np.array([min_u, min_v, max_u, max_v])  # bbox left-top uv , right_bottom uv
        # get the center minimum bounding rectangle as joints_uv center
        c_u = int((max_u + min_u) / 2)
        c_v = int((max_v + min_v) / 2)
        joints_center_uv = (c_u, c_v)
        return joints_bbox_uv, joints_center_uv

    def __call__(self, results):
        img_shape = np.array(results['img_shape'][:2]).reshape(-1, 2)
        joints_uv = xyz2uv(results['joints_xyz'], results['K'])
        # joints_uv_visible: joints_uv inside image boundary is visible (set to 1) and joints_uv out of image
        # boundary is invisible (set to 0)
        # Typically, joint_uv = (126, 262), image boundary wh = (224, 224), thus this joint is invisible. (Due to v (262) > h (224) )
        bool_matrix = joints_uv < img_shape
        joints_uv_visible = np.logical_and(bool_matrix[:, 0], bool_matrix[:, 1]).astype('int32')

        joints_bbox_uv, joints_center_uv = self.get_joints_bbox(joints_uv)

        results['joints_uv'] = joints_uv
        results['joints_uv_visible'] = joints_uv_visible
        results['joints_bbox_uv'] = joints_bbox_uv
        results['joints_center_uv'] = joints_center_uv
        return results


@PIPELINES.register_module()
class AffineCorp(object):
    """
    Centralize the joints to image center for better performance of
    models.

    Require keys : joints_xyz, K, img
    Generate keys : joints_uv, img_shape, rot_angle,
    Update keys : joints_xyz, K, img

    Args:
        img_outsize: image output size
        rot_angle_range: rotate angle range, uint: degree
    """
    def __init__(self,
                 centralize=True,
                 img_outsize=(256, 256),
                 rot_angle_range=(-180, 180),
                 ):

        # TODO: Add more argumentations.
        self.rot_angle_range = rot_angle_range
        self.img_outsize = img_outsize
        self.centralize = centralize

    def __call__(self, results):
        if self.rot_angle_range is None:
            rot_angle = 0
        else:
            rot_angle = np.random.randint(low=self.rot_angle_range[0],
                                          high=self.rot_angle_range[1])
        if self.centralize == True:
            joints_center_uv = results['joints_center_uv']
            # rotate first
            affine_matrix = cv2.getRotationMatrix2D(joints_center_uv, rot_angle, scale=1)
            img_center_uv = np.array(results['img_shape'][:2][::-1]) // 2
            # then shift joints_center_uv to img_center_uv
            delta = np.array(joints_center_uv) - img_center_uv
            affine_matrix[:, 2] -= delta
        else:
            center_uv = (self.img_outsize[1] // 2, self.img_outsize[0] // 2)
            affine_matrix = cv2.getRotationMatrix2D(center_uv, rot_angle, scale=1)

        # affine
        img_affine = cv2.warpAffine(results['img'], affine_matrix, self.img_outsize)
        joints_uv_affine = affine_transform(results['joints_uv'], affine_matrix)
        # rotate xy only
        joints_xy_affine = affine_transform(results['joints_xyz'][:, :2], affine_matrix)
        joints_xyz_affine = np.hstack((joints_xy_affine, results['joints_xyz'][:, 2:]))

        # compute new K
        K_new = get_K(joints_xyz_affine, joints_uv_affine)


        results['img'] = img_affine
        results['joints_xyz'] = joints_xyz_affine
        results['joints_uv'] = joints_uv_affine

        results['img_shape'] = self.img_outsize
        results['rot_angle'] = rot_angle
        results['K'] = K_new
        return results

@PIPELINES.register_module()
class GenerateHeatmap2D:
    '''
    Generate 2D heatmap
    Require keys: joints_uv, img_shape
    Generate keys: heatmap, heatmap weight
    Args:
        heatmap_shape: heatmap outsize.
        sigma: Gaussian sigma.
    '''
    def __init__(self,
                 heatmap_shape=(64, 64),
                 sigma=1):

        self.heatmap_shape = np.array(heatmap_shape)
        self.sigma = sigma

    def __call__(self, results):
        joints_uv_for_hm = results['joints_uv'] / results['img_shape'][:2] * self.heatmap_shape

        hm = np.array([generate_heatmap_2d(uv, self.heatmap_shape, self.sigma) \
                       if visible == 1 else np.zeros(self.heatmap_shape) \
                       for uv, visible in zip(np.int32(joints_uv_for_hm), results['joints_uv_visible'])])
        hm_weight = np.ones((results['joints_uv'].shape[0], 1))
        # for num_joints = 21
        # hm shape (21, 64, 64)
        # hm_weight shape (21, 1)
        results['heatmap'] = hm
        results['heatmap_weight'] = hm_weight
        return results

@PIPELINES.register_module()
class JointsUVNormalize:
    def __call__(self, results):
        joints_uv = results['joints_uv'] / results['img_shape'][:2][::-1] #  [::-1] img_shape (h, w) to img_shape (w, h)
        results['joints_uv'] = joints_uv
        return results

if __name__ == '__main__':
    pass