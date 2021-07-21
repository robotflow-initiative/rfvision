import torch
import numpy as np
import cv2
import torch.nn.functional as F
import scipy.spatial.transform.rotation as rot

def batch_argmax(tensor):
    '''
    Locate uv of max value in a tensor in dim 2 and dim 3.
    The tensor.ndim should be 4 (typically tensor.shape = (16, 3, 64, 64)).

    Args:
        tensor: torch.tensor

    Returns: uv of max value with batch.

    '''
    # if tensor.shape = (16, 3, 64, 64)
    assert tensor.ndim == 4, 'The tensor.ndim should be 4 (typically tensor.shape = (16, 3, 64, 64)).'
    b, c, h, w = tensor.shape
    tensor = tensor.reshape(b, c, -1)  # flatten the tensor to (16, 3, 4096)
    idx = tensor.argmax(dim=2)
    v = idx // w    # shape (16, 3)
    u = idx - v * w  # shape (16, 3)
    u, v = u.unsqueeze(-1), v.unsqueeze(-1)  # reshape u, v to (16, 3, 1) for cat
    uv = torch.cat((u, v), dim=-1)  # shape (16, 3, 2)
    return uv


def heatmap_to_uv(hm, mode='max'):
    '''
    Locate single keypoint pixel coordinate uv in heatmap.

    Args:
        hm:  shape (w, h), dim=2
        mode: -

    Returns: keypoint pixel coordinate uv, shape  (1, 2)

    '''

    assert mode in ('max', 'average')
    if mode == 'max':
        uv = batch_argmax(hm)
    elif mode == 'average':
        b, c, h, w = hm.shape
        hm = hm.reshape(b, c, -1)
        hm = hm / torch.sum(hm, dim=-1, keepdim=True)
        v_map, u_map = torch.meshgrid(torch.arange(h), torch.arange(w))
        u_map = u_map.reshape(1, 1, -1).float().to(hm.device)
        v_map = v_map.reshape(1, 1, -1).float().to(hm.device)
        u = torch.sum(u_map * hm, -1, keepdim=True)
        v = torch.sum(v_map * hm, -1, keepdim=True)
        uv = torch.cat((u, v), dim=-1)
    return uv


def generate_heatmap_2d(uv, heatmap_shape, sigma=7):
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
    Compute K (camera instrinics) by using given xyz and uv.

    Args:
        xyz: point cloud coordinates, shape (n, 3)
        uv: pixel coordinates shape (n, 2)

    Returns: camera instrinics, shape (3, 3)

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
    '''

    Args:
        xyz: point cloud coordinates xyz, shape (n, 3)
        K: camera instrincs, shape (3, 3)

    Returns:  pixel coordinates uv, shape (n, 2)

    '''

    xy = xyz / xyz[:, 2:]
    uv = xy @ K.T[:, :2]
    return uv

def uv2xyz(uv, K , depth):
    '''

    Args:
        uv: pixel coordinates shape (n, 2)
        K: camera instrincs, shape (3, 3)
        depth: depth values of uv, shape (n, 1)

    Returns: point cloud coordinates xyz, shape (n, 3)

    '''
    assert depth.ndim == 2, f'depth shape should be (n, 1) instead of {depth.shape}'
    assert uv.ndim == 2, f'uv shape should be (n, 2) instead of {uv.shape}'
    '''
    # Another form
    u = uv[:, 0]
    v = uv[:, 1]
    fx,fy=K[0,0],K[1,1]
    cx,cy=K[0,2],K[1,2]
    x = (u - cx) / fx 
    y = (v - cy) / fy  
    xyz = np.hstack((x.reshape(-1, 1) * depth, y.reshape(-1, 1) * depth, depth))
    '''
    xy = cv2.undistort(np.float32(uv), np.float32(K), distCoeffs=np.zeros(5))
    xyz = np.hstack((xy * depth, depth))
    return xyz

def batch_uv2xyz(uv, K, depth):
    '''

    Args:
        uv: pixel coordinates shape (b, n, 2)
        K: camera instrincs, shape (b, 3, 3)
        depth: depth values of uv, shape (b, n, 1)

    Returns: point cloud coordinates xyz, shape (b, n, 3)

    '''

    # u, v shape (b, n)  -- n: num of points
    u, v = uv[:, :, 0], uv[:, :, 1]
    # fx, fy, cx, cy shape (b, 1)
    fx, fy = K[:, 0, 0].unsqueeze(-1), K[:, 1, 1].unsqueeze(-1)
    cx, cy = K[:, 0, 2].unsqueeze(-1), K[:, 1, 2].unsqueeze(-1)
    x = (u - cx) / fx
    y = (v - cy) / fy
    xyz = torch.cat((x.unsqueeze(-1) * depth, y.unsqueeze(-1) * depth, depth), dim=-1)
    return xyz

def affine_transform(points, affine_matrix):

    '''
    Args:
        points: pixel coordinates uv coordinates xy, shape (n, 2) or point cloud coordinates xyz shape (n, 3)
        affine_matrix: shape (2, 3)

    Returns: affine-transformed coordinates

    '''
    points = np.hstack((points, np.ones((points.shape[0], 1))))
    points_affine = points.dot(affine_matrix.T)
    return points_affine


def normalize_point_cloud(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc_normalized = pc / m
    return pc_normalized, centroid, m


def depth_map_to_point_cloud(depth_map, K, depth_scale=1, flatten=True):
    '''
    Convert depth_map to point_cloud

    Args:
        depth_map: shape (h, w)
        K: camera instrincs, shape (3, 3)
        depth_scale: depth scale factor
        flatten: if True, output point cloud shape (h * w, 3) else (h, w, 3)

    Returns:

    '''
    assert depth_map.ndim == 2
    assert K.ndim == 2
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    h_map, w_map = np.mgrid[0:depth_map.shape[0], 0:depth_map.shape[1]]
    z = depth_map / depth_scale
    x = (w_map - cx) * z / fx
    y = (h_map - cy) * z / fy
    pc = np.dstack((x, y, z)) if flatten == False else np.dstack((x, y, z)).reshape(-1, 3)

    # cv2.rgbd.depthTo3d(depth_map.astype('float32'), K.astype('float32'))
    return pc

def normalize_quaternion(quaternion, eps=1e-12):
    return F.normalize(quaternion, p=2, dim=-1, eps=eps)


def my_atan2(y, x):
    pi = torch.from_numpy(np.array([np.pi])).to(y.device, y.dtype)
    ans = torch.atan(y / x)
    ans = torch.where(((y > 0).float() * (x < 0).float()).bool(), ans + pi, ans)
    ans = torch.where(((y < 0).float() * (x < 0).float()).bool(), ans + pi, ans)
    return ans


def quaternion_to_angle_axis(quaternion):
    q1 = quaternion[..., 1]
    q2 = quaternion[..., 2]
    q3 = quaternion[..., 3]
    sin_squared_theta = q1 * q1 + q2 * q2 + q3 * q3

    sin_theta = torch.sqrt(sin_squared_theta)
    cos_theta = quaternion[..., 0]
    two_theta = 2.0 * torch.where(
        cos_theta < 0.0, my_atan2(-sin_theta, -cos_theta),
        my_atan2(sin_theta, cos_theta))

    k_pos = two_theta / sin_theta
    k_neg = 2.0 * torch.ones_like(sin_theta)
    k = torch.where(sin_squared_theta > 0.0, k_pos, k_neg)

    angle_axis = quaternion[..., 1:] * k.unsqueeze(2)

    return angle_axis


def quaternion_inv(q):
    """
    inverse quaternion(s) q
    The quaternion should be in (x, y, z, w) format.
    Expects  tensors of shape (*, 4), where * denotes any number of dimensions.
    Returns q*r as a tensor of shape (*, 4).
    """
    assert q.shape[-1] == 4

    q_conj = q[..., :3] * -1.0
    q_conj = torch.cat((q_conj, q[..., 3:]), dim=-1)
    q_norm = torch.norm(q, dim=-1, keepdim=True)
    return q_conj / q_norm


def quaternion_mul(q, r):
    """
    Multiply quaternion(s) q with quaternion(s) r.
    The quaternion should be in (x, y, z, w) format.
    Expects two equally-sized tensors of shape (*, 4), where * denotes any number of dimensions.
    Returns q*r as a tensor of shape (*, 4).
    """
    assert q.shape[-1] == 4
    assert r.shape[-1] == 4

    original_shape = q.shape

    # Compute outer product
    # terms; ( * , 4, 4)
    terms = torch.bmm(r.view(-1, 4, 1), q.view(-1, 1, 4))

    w = terms[:, 3, 3] - terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2]
    x = terms[:, 3, 0] + terms[:, 0, 3] + terms[:, 1, 2] - terms[:, 2, 1]
    y = terms[:, 3, 1] - terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0]
    z = terms[:, 3, 2] + terms[:, 0, 1] - terms[:, 1, 0] + terms[:, 2, 3]
    return torch.stack((x, y, z, w), dim=1).view(original_shape)


if __name__ == '__main__':
    from scipy.spatial.transform import Rotation as rot
    p = rot.from_euler('z', 45, degrees=True)
