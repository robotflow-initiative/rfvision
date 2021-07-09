import numpy as np
import torch
import cv2
from rfvision.datasets import PIPELINES


def heatmap_to_uv(heatmap, mode='max'):
    '''

    Args:
        heatmap: -, shape (w, h), dim=2
        mode: -

    Returns:

    '''
    # TODO: Add mode 'average'
    assert mode in ('max',)
    assert heatmap.ndim == 2
    if mode == 'max':
        uv = torch.tensor(torch.where(heatmap == heatmap.max()))
    return uv


def generate_heatmap_2d(uv, heatmap_shape ,sigma=7):
    '''

    Args:
        uv: single pixel coordinate, shape (1, 2),
        heatmap_shape: -
        sigma:Gaussian sigma

    Returns:heatmap

    '''
    hm = np.zeros(heatmap_shape)
    hm[uv[1], uv[0]] = 1
    hm = cv2.GaussianBlur(hm, (sigma, sigma), 0)
    hm /= hm.max()  # normalize hm to [0, 1]
    return hm


def get_K(xyz, uv):
    '''
    Compute K (camera instrinics) by using given xyz and uv
    :param xyz: point cloud coordinates, shape (n, 3)
    :param uv: pixel coordinates, shape (n, 2)
    :return: K, shape(3, 3)
    '''
    assert xyz.ndim == 2 and uv.ndim == 2
    assert xyz.shape[0] == uv.shape[0]
    assert xyz.shape[1] == 3 and uv.shape[1] ==2
    xy = xyz[:, :2] / xyz[:, 2:]
    I = np.ones((xyz.shape[0], 1))
    x = np.hstack((xy[:, 0].reshape(-1, 1), I))
    y = np.hstack((xy[:, 1].reshape(-1, 1), I))
    u = np.hstack((uv[:, 0].reshape(-1, 1), I))
    v = np.hstack((uv[:, 1].reshape(-1, 1), I))
    # use least square
    fx, cx = np.linalg.inv(x.T.dot(x)).dot(x.T).dot(u)[:, 0]
    fy, cy = np.linalg.inv(y.T.dot(y)).dot(y.T).dot(v)[:, 0]
    K = np.float32([[fx, 0, cx],
                    [0, fy, cy],
                    [0,  0,  1]])
    return K

def xyz2uv(xyz, K):

    xy = xyz / xyz[:, 2:]
    uv = xy.dot(K.T)[:, :2]
    return uv

def affine_transform(points, affine_matrix):
    '''
    Affine transform uv
    :param points:pixel coordinates uv or point cloud coordinates xy, shape (n, 2)
    :param affine_matrix: shape(2, 3)
    :return:affine-transformed uv
    '''
    points = np.hstack((points, np.ones((points.shape[0], 1))))
    points_affine = points.dot(affine_matrix.T)
    return points_affine


@PIPELINES.register_module()
class GetJointsUV:
    # Compute joints uv toward joints xyz and K.
    def __call__(self, results):
        results['joints_uv'] = xyz2uv(results['joints_xyz'], results['K'])
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

    def get_joints_center_uv(self, joints_uv):
        # get the center minimum bounding rectangle as joints_uv center
        min_u, min_v = joints_uv.min(0)
        max_u, max_v = joints_uv.max(0)
        c_u = int((max_u + min_u) / 2)
        c_v = int((max_v + min_v) / 2)
        joints_center_uv = (c_u, c_v)
        return joints_center_uv


    def __call__(self, results):
        if self.rot_angle_range is None:
            rot_angle = 0
        else:
            rot_angle = np.random.randint(low=self.rot_angle_range[0],
                                          high=self.rot_angle_range[1])
        if self.centralize == True:
            center_uv = (self.img_outsize[1] // 2, self.img_outsize[0] // 2)
        else:
            center_uv = self.get_joints_center_uv(results['joints_uv'])

        # affine
        affine_matrix = cv2.getRotationMatrix2D(center_uv, rot_angle, scale=1)
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

        self.heatmap_shape = heatmap_shape
        self.sigma = sigma

    def __call__(self, results):
        joints_uv_for_hm = results['joints_uv'] / results['img_shape'] * np.array(self.heatmap_shape)
        hm = np.array([generate_heatmap_2d(uv, self.heatmap_shape, self.sigma) for uv in np.int32(joints_uv_for_hm)])
        hm_weight = np.ones((results['joints_uv'].shape[0], 1))
        # for num_joints = 21
        # hm shape (21, 64, 64)
        # hm_weight shape (21, )
        results['heatmap'] = hm
        results['heatmap_weight'] = hm_weight
        return results

@PIPELINES.register_module()
class JointsNormalize:
    '''
    Normalize joints_z and joints_uv, this pipeline is specially used for handtailor now.
    Require keys : joints_xyz, joints_uv, img_shape
    Generate keys: joint_root, joint_bone, joints_uvd
    Update keys : joints_xyz
    '''
    def __init__(self):
        # DEPTH_MIN and DEPTH_RANGE is the empirical value
        self.DEPTH_MIN = -1.5
        self.DEPTH_RANGE = 3

    def __call__(self, results):
        ############# normalize joint_z ############
        joints_xyz = results['joints_xyz']
        joint_root = joints_xyz[0]
        joint_root_z = joint_root[2]
        joints_z = joints_xyz[:, 2:]

        # joint_bone: the Euler distance between joint No.9 and joint No.0
        joint_bone = np.linalg.norm(joints_xyz[9] - joint_root)    # int
        joints_z_normalized = (joints_z - joint_root_z) / joint_bone  # shape (21, 1)
        joints_z_normalized = (joints_z_normalized - self.DEPTH_MIN) / self.DEPTH_RANGE # shape (21, 1)

        joints_xyz[:, 2:] = joints_z_normalized

        ############# normalize joint_uv ############
        joints_uv_normalized = results['joints_uv'] / np.array(results['img_shape'])

        # combine uv and z to 'uvd'
        joints_uvd = np.hstack((joints_uv_normalized, joints_z_normalized))

        results['joints_uvd'] = joints_uvd
        results['joints_xyz'] = joints_xyz
        results['joint_root'] = joint_root
        results['joint_bone'] = joint_bone
        return results

if __name__ == '__main__':
    pass