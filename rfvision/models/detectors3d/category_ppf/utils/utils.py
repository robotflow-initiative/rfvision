import numpy as np
import math
import open3d as o3d
import torch


def backproject(depth, intrinsics, instance_mask):
    intrinsics_inv = np.linalg.inv(intrinsics)
    image_shape = depth.shape
    width = image_shape[1]
    height = image_shape[0]

    x = np.arange(width)
    y = np.arange(height)

    # non_zero_mask = np.logical_and(depth > 0, depth < 5000)
    non_zero_mask = (depth > 0)
    final_instance_mask = np.logical_and(instance_mask, non_zero_mask)

    idxs = np.where(final_instance_mask)
    grid = np.array([idxs[1], idxs[0]])

    # shape: height * width
    # mesh_grid = np.meshgrid(x, y) #[height, width, 2]
    # mesh_grid = np.reshape(mesh_grid, [2, -1])
    length = grid.shape[1]
    ones = np.ones([1, length])
    uv_grid = np.concatenate((grid, ones), axis=0)  # [3, num_pixel]

    xyz = intrinsics_inv @ uv_grid  # [3, num_pixel]
    xyz = np.transpose(xyz)  # [num_pixel, 3]

    z = depth[idxs[0], idxs[1]]

    # print(np.amax(z), np.amin(z))
    pts = xyz * z[:, np.newaxis] / xyz[:, -1:]
    pts[:, 0] = -pts[:, 0]
    pts[:, 1] = -pts[:, 1]

    return pts, idxs


def fibonacci_sphere(samples):
    points = []
    phi = math.pi * (3. - math.sqrt(5.))  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = math.cos(theta) * radius
        z = math.sin(theta) * radius

        points.append((x, y, z))

    return points


def rotz(a):
    return np.array([[np.cos(a), np.sin(a), 0, 0],
                     [-np.sin(a), np.cos(a), 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])


def roty(a):
    return np.array([[np.cos(a), 0, -np.sin(a), 0],
                     [0, 1, 0, 0],
                     [np.sin(a), 0, np.cos(a), 0],
                     [0, 0, 0, 1]])


def rotx(a):
    return np.array([[1, 0, 0, 0],
                     [0, np.cos(a), -np.sin(a), 0],
                     [0, np.sin(a), np.cos(a), 0],
                     [0, 0, 0, 1]])


shapenet_obj_scales = {
    '02946921': [0.128, 0.18],
    '02876657': [0.2300, 0.4594],
    '02880940': [0.1851, 0.2381],
    '02942699': [0.1430, 0.2567],
    '03642806': [0.3862, 0.5353],
    '03797390': [0.1501, 0.1995]
}


tr_ranges = {
    1: [0.25, 0.25],
    2: [0.12, 0.12],
    3: [0.15, 0.15],
    4: [0.1, 0.1],
    5: [0.3, 0.3],
    6: [0.12, 0.12]
}

scale_ranges = {
    1: [0.05, 0.15, 0.05],
    2: [0.07, 0.03, 0.07],
    3: [0.05, 0.05, 0.07],
    4: [0.037, 0.055, 0.037],
    5: [0.13, 0.1, 0.15],
    6: [0.06, 0.05, 0.045]
}


def pc_downsample(pc: np.ndarray, voxel_size=0.05):
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pc))
    return np.array(pcd.voxel_down_sample(voxel_size).points)

def estimate_normals(pc, knn):
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pc))
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn))
    return np.array(pcd.normals)


def real2prob(val, max_val, num_bins, circular=False):
    is_torch = isinstance(val, torch.Tensor)
    if is_torch:
        res = torch.zeros((*val.shape, num_bins), dtype=val.dtype).to(val.device)
    else:
        res = np.zeros((*val.shape, num_bins), dtype=val.dtype)

    if not circular:
        interval = max_val / (num_bins - 1)
        if is_torch:
            low = torch.clamp(torch.floor(val / interval).long(), max=num_bins - 2)
        else:
            low = np.clip(np.floor(val / interval).astype(np.int64), a_min=None, a_max=num_bins - 2)
        high = low + 1
        # assert torch.all(low >= 0) and torch.all(high < num_bins)

        # huge memory
        if is_torch:
            res.scatter_(-1, low[..., None], torch.unsqueeze(1. - (val / interval - low), -1))
            res.scatter_(-1, high[..., None], 1. - torch.gather(res, -1, low[..., None]))
        else:
            np.put_along_axis(res, low[..., None], np.expand_dims(1. - (val / interval - low), -1), -1)
            np.put_along_axis(res, high[..., None], 1. - np.take_along_axis(res, low[..., None], -1), -1)
        # res[..., low] = 1. - (val / interval - low)
        # res[..., high] = 1. - res[..., low]
        # assert torch.all(0 <= res[..., low]) and torch.all(1 >= res[..., low])
        return res
    else:
        interval = max_val / num_bins
        if is_torch:
            val_new = torch.clone(val)
        else:
            val_new = val.copy()
        val_new[val < interval / 2] += max_val
        res = real2prob(val_new - interval / 2, max_val, num_bins + 1)
        res[..., 0] += res[..., -1]
        return res[..., :-1]

def generate_target(pc, pc_normal, up_sym=False, right_sym=False, z_right=False, subsample=200000):
    if subsample is None:
        xv, yv = np.meshgrid(np.arange(pc.shape[1]), np.arange(pc.shape[1]))
        point_idxs = np.stack([yv, xv], -1).reshape(-1, 2)
    else:
        point_idxs = np.random.randint(0, pc.shape[0], size=[subsample, 2])

    a = pc[point_idxs[:, 0]]
    b = pc[point_idxs[:, 1]]
    pdist = a - b
    pdist_unit = pdist / (np.linalg.norm(pdist, axis=-1, keepdims=True) + 1e-7)
    proj_len = np.sum(a * pdist_unit, -1)
    oc = a - proj_len[..., None] * pdist_unit
    dist2o = np.linalg.norm(oc, axis=-1)
    target_tr = np.stack([proj_len, dist2o], -1)

    up = np.array([0, 1, 0])
    down = np.array([0, -1, 0])
    if z_right:
        right = np.array([0, 0, 1])
        left = np.array([0, 0, -1])
    else:
        right = np.array([1, 0, 0])
        left = np.array([-1, 0, 0])
    up_cos = np.arccos(np.sum(pdist_unit * up, -1))
    if up_sym:
        up_cos = np.minimum(up_cos, np.arccos(np.sum(pdist_unit * down, -1)))
    right_cos = np.arccos(np.sum(pdist_unit * right, -1))
    if right_sym:
        right_cos = np.minimum(right_cos, np.arccos(np.sum(pdist_unit * left, -1)))
    target_rot = np.stack([up_cos, right_cos], -1)

    pairwise_normals = pc_normal[point_idxs[:, 0]]
    pairwise_normals[np.sum(pairwise_normals * pdist_unit, -1) < 0] *= -1
    target_rot_aux = np.stack([
        np.sum(pairwise_normals * up, -1) > 0,
        np.sum(pairwise_normals * right, -1) > 0
    ], -1).astype(np.float32)
    return target_tr.astype(np.float32).reshape(-1, 2), target_rot.astype(np.float32).reshape(), target_rot_aux.reshape(
        -1, 2), point_idxs.astype(np.int64)-1,



category_cfgs = {1: {'res': 0.004, 'npoint_max': 10000, 'regress_right': False, 'tr_num_bins': 32, 'rot_num_bins': 36, 'up_sym': True, 'right_sym': False, 'z_right': False, 'knn': 60, 'cls_bins': True,
                     'K' :np.float32([[591.0125, 0, 320],[0, 590.16775, 240],[0, 0, 1]]), 'num_rots':72, 'n_threads': 512},
                 2: {'res': 0.004, 'npoint_max': 10000, 'regress_right': False, 'tr_num_bins': 32, 'rot_num_bins': 36, 'up_sym': False, 'right_sym': False, 'z_right': False, 'knn': 60,'cls_bins': True,
                     'K' :np.float32([[591.0125, 0, 320],[0, 590.16775, 240],[0, 0, 1]]), 'num_rots':72, 'n_threads': 512},
                 3: {'res': 0.004, 'npoint_max': 10000, 'regress_right': True, 'tr_num_bins': 32, 'rot_num_bins': 36, 'up_sym': False, 'right_sym': False, 'z_right': False, 'knn': 60,'cls_bins': True,
                     'K' :np.float32([[591.0125, 0, 320],[0, 590.16775, 240],[0, 0, 1]]), 'num_rots':72, 'n_threads': 512},
                 4: {'res': 0.004, 'npoint_max': 10000, 'regress_right': False, 'tr_num_bins': 32, 'rot_num_bins': 36, 'up_sym': True, 'right_sym': False, 'z_right': False, 'knn': 60,'cls_bins': True,
                     'K' :np.float32([[591.0125, 0, 320],[0, 590.16775, 240],[0, 0, 1]]), 'num_rots':72, 'n_threads': 512},
                 5: {'res': 0.01, 'npoint_max': 10000, 'regress_right': True, 'tr_num_bins': 32, 'rot_num_bins': 36, 'up_sym': False, 'right_sym': False, 'z_right': False, 'knn': 60,'cls_bins': True,
                     'K' :np.float32([[591.0125, 0, 320],[0, 590.16775, 240],[0, 0, 1]]), 'num_rots':72, 'n_threads': 512},
                 6: {'res': 0.004, 'npoint_max': 10000, 'regress_right': True, 'tr_num_bins': 32, 'rot_num_bins': 36, 'up_sym': True, 'right_sym': False, 'z_right': False, 'knn': 60,'cls_bins': True,
                     'K' :np.float32([[591.0125, 0, 320],[0, 590.16775, 240],[0, 0, 1]]), 'num_rots':72, 'n_threads': 512}}