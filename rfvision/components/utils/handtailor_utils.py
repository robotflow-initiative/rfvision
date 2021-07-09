import torch
import torch.nn.functional as F
import numpy as np

DEPTH_RANGE = 3.0
DEPTH_MIN = -1.5

def hm_to_kp2d(hm):
    b, c, w, h = hm.size()
    hm = hm.view(b, c, -1)
    hm = hm / torch.sum(hm, -1, keepdim=True)
    coord_map_x = torch.arange(0, w).view(-1, 1).repeat(1, h).to(hm.device)
    coord_map_y = torch.arange(0, h).view(1, -1).repeat(w, 1).to(hm.device)
    coord_map_x = coord_map_x.view(1, 1, -1).float()
    coord_map_y = coord_map_y.view(1, 1, -1).float()
    x = torch.sum(coord_map_x * hm, -1, keepdim=True)
    y = torch.sum(coord_map_y * hm, -1, keepdim=True)
    kp_2d = torch.cat((y, x), dim=-1)
    return kp_2d

# used for handtailor.py(detector3d)
def hm_to_uvd(hm3d):
    b, c, w, h = hm3d.size()
    hm2d = hm3d[:, :21, ...]
    depth = hm3d[:, 21:, ...]
    uv = hm_to_kp2d(hm2d) / w
    hm2d = hm2d.view(b, 1, c // 2, -1)
    depth = depth.view(b, 1, c // 2, -1)
    hm2d = hm2d / torch.sum(hm2d, -1, keepdim=True)
    d = torch.sum(depth * hm2d, -1).permute(0, 2, 1)
    joint = torch.cat((uv, d), dim=-1)
    return joint


# used for manonet.py
def uvd2xyz(uvd, joint_root, joint_bone, intr=None, inp_res=256,):
    uv = uvd[:, :, :2] * inp_res  # 0~256
    depth = (uvd[:, :, 2] * DEPTH_RANGE) + DEPTH_MIN
    root_depth = joint_root[:, -1].unsqueeze(1)  # (B, 1)
    z = depth * joint_bone.expand_as(uvd[:, :, 2]) + \
        root_depth.expand_as(uvd[:, :, 2])  # B x M

    '''2. uvd->xyz'''
    camparam = torch.cat((intr[:, 0:1, 0], intr[:, 1:2, 1], intr[:, 0:1, 2], intr[:, 1:2, 2]), 1)
    camparam = camparam.unsqueeze(1).repeat(1, uvd.size(1), 1)  # B x M x 4
    xy = ((uv - camparam[:, :, 2:4]) / camparam[:, :, :2]) * \
         z.unsqueeze(2).expand_as(uv)  # B x M x 2
    return torch.cat((xy, z.unsqueeze(2)), -1)  # B x M x 3


# used for iknet.py
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


def get_affine_trans_no_rot(center, scale, res):
    affinet = np.zeros((3, 3))
    affinet[0, 0] = float(res[1]) / scale
    affinet[1, 1] = float(res[0]) / scale
    affinet[0, 2] = res[1] * (-float(center[0]) / scale + .5)
    affinet[1, 2] = res[0] * (-float(center[1]) / scale + .5)
    affinet[2, 2] = 1
    return affinet


def get_affine_transform(center, scale, res, rot=0):
    rot_mat = np.zeros((3, 3))
    sn, cs = np.sin(rot), np.cos(rot)
    rot_mat[0, :2] = [cs, -sn]
    rot_mat[1, :2] = [sn, cs]
    rot_mat[2, 2] = 1
    # Rotate center to obtain coordinate of center in rotated image
    origin_rot_center = rot_mat.dot(center.tolist() + [1])[:2]
    # Get center for transform with verts rotated around optical axis
    # (through pixel center, smthg like 128, 128 in pixels and 0,0 in 3d world)
    # For this, rotate the center but around center of image (vs 0,0 in pixel space)
    t_mat = np.eye(3)
    t_mat[0, 2] = -res[1] / 2
    t_mat[1, 2] = -res[0] / 2
    t_inv = t_mat.copy()
    t_inv[:2, 2] *= -1
    transformed_center = t_inv.dot(rot_mat).dot(t_mat).dot(center.tolist() + [
        1,
    ])
    post_rot_trans = get_affine_trans_no_rot(origin_rot_center, scale, res)
    total_trans = post_rot_trans.dot(rot_mat)
    # check_t = get_affine_transform_bak(center, scale, res, rot)
    affinetrans_post_rot = get_affine_trans_no_rot(transformed_center[:2],
                                                   scale, res)
    return total_trans.astype(np.float32), affinetrans_post_rot.astype(
        np.float32)


def gen_cam_param(joint, kp2d, mode='ortho'):
    if mode in ['persp', 'perspective']:
        kp2d = kp2d.reshape(-1)[:, np.newaxis]  # (42, 1)
        joint = joint / joint[:, 2:]
        joint = joint[:, :2]
        jM = np.zeros((42, 2), dtype="float32")
        for i in range(joint.shape[0]):  # 21
            jM[2 * i][0] = joint[i][0]
            jM[2 * i + 1][1] = joint[i][1]
        pad2 = np.array(range(42))
        pad2 = (pad2 % 2)[:, np.newaxis]
        pad1 = (1 - pad2)

        jM = np.concatenate([jM, pad1, pad2], axis=1)  # (42, 4)
        jMT = jM.transpose()  # (4, 42)
        jMTjM = np.matmul(jMT, jM)  # (4,4)
        jMTb = np.matmul(jMT, kp2d)
        cam_param = np.matmul(np.linalg.inv(jMTjM), jMTb)
        cam_param = cam_param.reshape(-1)
        cam_param = np.float32([
            [cam_param[0], 0.0, cam_param[2]],
            [0.0, cam_param[1], cam_param[3]],
            [0.0, 0.0, 1.0], ])

        return cam_param
    elif mode in ['ortho', 'orthogonal']:
        # ortho only when
        assert np.sum(np.abs(joint[0, :])) == 0
        joint = joint[:, :2]  # (21, 2)
        joint = joint.reshape(-1)[:, np.newaxis]
        kp2d = kp2d.reshape(-1)[:, np.newaxis]
        pad2 = np.array(range(42))
        pad2 = (pad2 % 2)[:, np.newaxis]
        pad1 = (1 - pad2)
        jM = np.concatenate([joint, pad1, pad2], axis=1)  # (42, 3)
        jMT = jM.transpose()  # (3, 42)
        jMTjM = np.matmul(jMT, jM)
        jMTb = np.matmul(jMT, kp2d)
        cam_param = np.matmul(np.linalg.inv(jMTjM), jMTb)
        cam_param = cam_param.reshape(-1)
        cam_param = np.float32([
            [cam_param[0], 0.0, cam_param[2]],
            [0.0, cam_param[1], cam_param[3]],
            [0.0, 0.0, 1.0], ])
        return cam_param
    else:
        raise Exception("Unkonwn mode type. should in ['persp', 'orth']")


def gen_heatmap(img, pt, sigma):
    """generate heatmap based on pt coord.

    :param img: original heatmap, zeros
    :type img: np (H,W) float32
    :param pt: keypoint coord.
    :type pt: np (2,) int32
    :param sigma: guassian sigma
    :type sigma: float
    :return
    - generated heatmap, np (H, W) each pixel values id a probability
    - flag 0 or 1: indicate wheather this heatmap is valid(1)

    """

    pt = pt.astype(np.int32)
    # Check that any part of the gaussian is in-bounds
    ul = [int(pt[0] - 3 * sigma), int(pt[1] - 3 * sigma)]
    br = [int(pt[0] + 3 * sigma + 1), int(pt[1] + 3 * sigma + 1)]
    if (
            ul[0] >= img.shape[1]
            or ul[1] >= img.shape[0]
            or br[0] < 0
            or br[1] < 0
    ):
        # If not, just return the image as is
        return img, 0

    # Generate gaussian
    size = 6 * sigma + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    return img


def transform_coords(pts, affine_trans, invert=False):
    """
    Args:
        pts(np.ndarray): (point_nb, 2)
    """
    if invert:
        affine_trans = np.linalg.inv(affine_trans)
    hom2d = np.concatenate([pts, np.ones([np.array(pts).shape[0], 1])], 1)
    transformed_rows = affine_trans.dot(hom2d.transpose()).transpose()[:, :2]
    return transformed_rows.astype(int)


def get_pck_all(pred_joints_xyzs, gt_joints_xyzs, threshold):
    # pred_joints_xyzs tensor shape : (n, 21, 3)
    # gt_joints_xyzs tensor shape : (n, 21, 3)
    assert pred_joints_xyzs.shape == gt_joints_xyzs.shape and len(pred_joints_xyzs.shape) == len(gt_joints_xyzs.shape) == 3
    dist = torch.sqrt(torch.sum((gt_joints_xyzs * 1000 - pred_joints_xyzs * 1000) ** 2, -1))  # shape (21, n)
    pck = (torch.mean(dist, -2) <= threshold).float()  # shape (21, 1)
    return pck.mean()


