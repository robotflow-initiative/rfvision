import numpy as np
import cv2
import torch

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
        uv: single pixel coordinate, shape (1, 2)
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
    points = np.hstack(points, np.ones(points.shape[[0], 1]))
    points_affine = points.dot(affine_matrix.T)
    return points_affine

class CentralizeJoints(object):
    """
    Centralize the joints to image center for better performance of
    models.

    Require keys : joints_xyz, K, img
    Generate keys : joints_uv, img_shape, rot_angle,
    Update keys : joints_xyz, K, img

    Args:
        img_shape: image output size
        rot_angle_range: rotate angle range, uint: degree
    """
    def __init__(self,
                 img_shape=(256, 256),
                 rot_angle_range=(-180, 180),
                 ):

        # TODO: Add more argumentations.
        self.rot_angle_range = rot_angle_range
        self.img_shape = img_shape

    def get_joints_center_uv(self, joints_uv):
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
        joints_uv = xyz2uv(results['joints_xyz'], results['K'])
        center_uv = self.get_joints_center_uv(joints_uv)
        affine_matrix = cv2.getRotationMatrix2D(center_uv, rot_angle, scale=1)
        img_affine = cv2.warpAffine(results['img'], affine_matrix, self.img_shape)
        joints_uv_affine = affine_transform(joints_uv, affine_matrix)
        joints_uv_affine_normalized = joints_uv_affine / np.array(self.img_shape)
        # rotate xy only
        joints_xy_affine = affine_transform(results['joints_xyz'][:, :2], affine_matrix)
        joints_xyz_affine = np.hstack((joints_xy_affine, results['joints_xyz'][:, 2:]))

        # compute new K
        K_new = get_K(joints_xyz_affine, joints_uv_affine)


        results['img'] = img_affine
        results['joints_xyz'] = joints_xyz_affine
        results['joints_uv'] = joints_uv_affine_normalized

        results['img_shape'] = self.img_shape
        results['rot_angle'] = self.rot_angle
        results['K'] = K_new
        return results

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
        hm = np.array([generate_heatmap_2d(uv, self.heatmap_shape, self.sigma) for uv in joints_uv_for_hm])
        hm_weight = np.ones(results['joints_uv'].shape[0])
        # for num_joints = 21
        # hm shape (21, 64, 64)
        # hm_weight shape (21, )
        results['heatmap'] = hm
        results['heatmap_weight'] = hm_weight
        return results


class JointsZNormalize:

    def __init__(self):

        self.DEPTH_MIN = -1.5
        self.DEPTH_RANGE = 3

    def __call__(self, results):
        ############# normalize joint_z ############
        joints_xyz = results['joints_xyz']
        root_joint = joints_xyz[0]
        root_joint_z = root_joint[2]
        joints_z = joints_xyz[:, 2:]

        # joint_bone: the Euler distance between joint No.9 and joint No.0
        joint_bone = np.linalg.norm(joints_xyz[9] - root_joint)    # int
        joints_z_normalized = (joints_z - root_joint_z) / joint_bone  # shape (21, 1)
        joints_z_normalized = (joints_z_normalized - self.DEPTH_MIN) / self.DEPTH_RANGE # shape (21, 1)

        joints_xyz[:, 2] = joints_z_normalized

        results['joints_xyz'] = joints_xyz
        return results

if __name__ == '__main__':
    hm = torch.rand(1, 21, 64, 64)
    hm[:,:,32,32]=100
    import time
    from rfvision.components.utils.handtailor_utils import hm_to_kp2d
    start = time.time()
    torch.cat(tuple(heatmap_to_uv(hm) for hm in hm[0]))
    duration = time.time() - start