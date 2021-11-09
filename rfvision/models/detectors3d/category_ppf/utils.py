import numpy as np
import cupy as cp
from .voting import ppf_kernel
import visdom
import cv2
import math

# vis = visdom.Visdom(server='10.52.28.4', port=22)

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

def visualize(vis, *pcs, **opts):
    vis_pc = np.concatenate(pcs)
    vis_label = np.ones((sum([p.shape[0] for p in pcs])), np.int64)
    a = 0
    for i, pc in enumerate(pcs):
        vis_label[a:a+pc.shape[0]] = i + 1
        a += pc.shape[0]
    vis.scatter(vis_pc, vis_label, **opts)


def validation(vertices, outputs, probs, res, point_idxs, n_ppfs, num_rots=36, visualize=True):
    with cp.cuda.Device(0):
        block_size = (vertices.shape[0] ** 2 + 512 - 1) // 512

        corners = np.stack([np.min(vertices, 0), np.max(vertices, 0)])
        grid_res = ((corners[1] - corners[0]) / res).astype(np.int32) + 1
        grid_obj = cp.asarray(np.zeros(grid_res, dtype=np.float32))
        ppf_kernel(
            (block_size, 1, 1),
            (512, 1, 1),
            (
                cp.asarray(vertices).astype(cp.float32), cp.asarray(outputs).astype(cp.float32),
                cp.asarray(probs).astype(cp.float32), cp.asarray(point_idxs).astype(cp.int32), grid_obj,
                cp.asarray(corners[0]), cp.float32(res),
                n_ppfs, num_rots, grid_obj.shape[0], grid_obj.shape[1], grid_obj.shape[2]
            )
        )

        grid_obj = grid_obj.get()

        # cand = np.array(np.unravel_index([np.argmax(grid_obj, axis=None)], grid_obj.shape)).T[::-1]
        # grid_obj[cand[-1][0]-20:cand[-1][0]+20, cand[-1][1]-20:cand[-1][1]+20, cand[-1][2]-20:cand[-1][2]+20] = 0
        if visualize:
            vis.heatmap(cv2.rotate(grid_obj.max(0), cv2.ROTATE_90_COUNTERCLOCKWISE), win=3, opts=dict(title='front'))
            vis.heatmap(cv2.rotate(grid_obj.max(1), cv2.ROTATE_90_COUNTERCLOCKWISE), win=4, opts=dict(title='bird'))
            vis.heatmap(cv2.rotate(grid_obj.max(2), cv2.ROTATE_90_COUNTERCLOCKWISE), win=5, opts=dict(title='side'))

        cand = np.array(np.unravel_index([np.argmax(grid_obj, axis=None)], grid_obj.shape)).T[::-1]
        cand_world = corners[0] + cand * res
        # print(cand_world[-1])
        return grid_obj, cand_world