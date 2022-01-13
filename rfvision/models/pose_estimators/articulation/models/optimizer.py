import os
import json
import numpy as np

from .utils import ransac, joint_transformation_estimator, joint_transformation_verifier, get_3d_bbox

INSTANCE_CLASSES = ('BG', 'box', 'stapler', 'cutter', 'drawer', 'scissor')
label_maps = {'box': (0, 1, 2),
              'stapler': (0, 3, 4),
              'cutter': (0, 5, 6),
              'drawer': (0, 7, 8, 9, 10),
              'scissor': (0, 11, 12)}  # every category contains 0 for BG


def fetch_factors_nocs(root_dset):
    norm_factors = {}
    corner_pts = {}
    urdf_metas = json.load(open(root_dset + '/urdf_metas.json'))['urdf_metas']
    for urdf_meta in urdf_metas:
        norm_factors[urdf_meta['id']] = np.array(urdf_meta['norm_factors'])
        corner_pts[urdf_meta['id']] = np.array(urdf_meta['corner_pts']).squeeze(2)

    return norm_factors, corner_pts


def fetch_joints_params(root_dset):
    joint_ins = {}
    urdf_metas = json.load(open(root_dset + '/urdf_metas.json'))['urdf_metas']
    for urdf_meta in urdf_metas:
        if urdf_meta == []:
            continue
        joint_ins[urdf_meta['id']] = dict(xyz=[], axis=[], type=[], parent=[], child=[])
        joint_types = urdf_meta['joint_types']
        joint_parents = urdf_meta['joint_parents']
        joint_children = urdf_meta['joint_children']
        joint_xyz = urdf_meta['joint_xyz']
        joint_rpy = urdf_meta['joint_rpy']
        assert len(joint_types) == len(joint_parents) == len(joint_children) == len(joint_xyz) == len(joint_rpy)

        num_joints = len(joint_types)
        for n in range(num_joints):
            x, y, z = joint_xyz[n]
            joint_ins[urdf_meta['id']]['xyz'].append([y, z, x])
            r, p, y = joint_rpy[n]
            joint_ins[urdf_meta['id']]['axis'].append([p, y, r])
            joint_ins[urdf_meta['id']]['type'].append(joint_types[n])
            joint_ins[urdf_meta['id']]['parent'].append(joint_parents[n])
            joint_ins[urdf_meta['id']]['child'].append(joint_children[n])

    return joint_ins


def optimize_pose(result):
    refined_result = dict(Rotation={}, Translation={}, Box3D={}, Joint={})
    cls_per_pt_pred = result['W'][0].cpu().numpy()
    cls_per_pt_pred = np.argmax(cls_per_pt_pred, axis=1)
    joint_cls_pred = result['index_per_point'][0].cpu().numpy()
    joint_cls_pred = np.argmax(joint_cls_pred, axis=1)

    category_name = INSTANCE_CLASSES[result['category_id']]
    label_map = label_maps[category_name][1:]
    num_parts = len(label_map)
    partidx = [None] * num_parts
    joint_idx = [None] * num_parts
    for idx, j in enumerate(label_map):
        partidx[idx] = np.where(cls_per_pt_pred == j)[0]
    for idx, j in enumerate(label_map):
        if j == 0:
            continue
        joint_idx[idx] = np.where(joint_cls_pred == j)[0]

    P = result['pts'].cpu().numpy()
    if 'points_mean' in result.keys():
        P += result['points_mean']
    nocs_pred = result['nocs_per_point'][0].cpu().numpy()
    all_norm_factors, all_corner_pts = fetch_factors_nocs('demo')
    urdf_id = result['urdf_id']

    base_part_id = label_map[0]
    for idx, j in enumerate(label_map[1:]):
        source0 = nocs_pred[partidx[0], 3 * 1:3 * (1 + 1)]
        target0 = P[partidx[0], :]
        source1 = nocs_pred[partidx[idx+1], 3 * j:3 * (j + 1)]
        target1 = P[partidx[idx+1], :3]

        joint_axis_per_points = result['joint_axis_per_point'][0].cpu().numpy()
        jt_axis = np.median(joint_axis_per_points[joint_idx[idx + 1], 3 * j:3 * (j + 1)], 0)

        dataset = dict()
        dataset['source0'] = source0
        dataset['target0'] = target0
        dataset['nsource0'] = source0.shape[0]
        dataset['source1'] = source1
        dataset['target1'] = target1
        dataset['nsource1'] = source1.shape[0]
        dataset['joint_direction'] = jt_axis

        niter = 200
        inlier_th = 0.1
        best_model, best_inliers = ransac(dataset, joint_transformation_estimator,
                                          joint_transformation_verifier, inlier_th, niter)

        # pose (4x4 transformation) estimation
        if idx == 0:
            offset0 = all_corner_pts[urdf_id][1].mean(axis=0)
            translation0_pred_recover = best_model['scale0'] * np.matmul(best_model['rotation0'],
                                                                         np.array([0.5, 0.5, 0.5]).reshape(1, 3).T) \
                                        + best_model['translation0'].reshape(3, 1) \
                                        - np.matmul(best_model['rotation0'], offset0.reshape(3, 1))
            translation0_pred_recover = translation0_pred_recover.T[0]

            refined_result['Rotation'][base_part_id] = best_model['rotation0']
            refined_result['Translation'][base_part_id] = translation0_pred_recover

        offset1 = all_corner_pts[urdf_id][idx + 2].mean(axis=0)
        translation1_pred_recover = best_model['scale1'] * np.matmul(best_model['rotation1'],
                                                                     np.array([0.5, 0.5, 0.5]).reshape(1, 3).T) \
                                    + best_model['translation1'].reshape(3, 1) \
                                    - np.matmul(best_model['rotation1'], offset1.reshape(3, 1))
        translation1_pred_recover = translation1_pred_recover.T[0]

        refined_result['Rotation'][j] = best_model['rotation1']
        refined_result['Translation'][j] = translation1_pred_recover

        # 3D bounding box estimation
        if idx == 0:
            centered_nocs0 = source0 - 0.5
            s_pred0 = 2 * np.max(abs(centered_nocs0), axis=0)
            bbox3d_pred0 = get_3d_bbox(s_pred0, shift=np.array([1 / 2, 1 / 2, 1 / 2])).transpose()
            bbox3d_pred0 = best_model['scale0'] * np.matmul(best_model['rotation0'], bbox3d_pred0.T) \
                           + best_model['translation0'].reshape(3, 1)
            bbox3d_pred0 = bbox3d_pred0.T

            refined_result['Box3D'][base_part_id] = bbox3d_pred0

        centered_nocs1 = source1 - 0.5
        s_pred1 = 2 * np.max(abs(centered_nocs1), axis=0)
        bbox3d_pred1 = get_3d_bbox(s_pred1, shift=np.array([1 / 2, 1 / 2, 1 / 2])).transpose()
        bbox3d_pred1 = best_model['scale1'] * np.matmul(best_model['rotation1'], bbox3d_pred1.T) \
                       + best_model['translation1'].reshape(3, 1)
        bbox3d_pred1 = bbox3d_pred1.T

        refined_result['Box3D'][j] = bbox3d_pred1

        # joint estimation
        gocs_pred = result['gocs_per_point'][0].cpu().numpy()
        # if idx == 0:
        x = gocs_pred[partidx[0], 3 * 1:3 * (1 + 1)]  # N * 3
        y = nocs_pred[partidx[0], 3 * 1:3 * (1 + 1)]  # N * 3
        scale_g_to_p = np.std(np.mean(y, axis=1)) / np.std(np.mean(x, axis=1))
        translation_g_to_p = np.mean(y - scale_g_to_p * x, axis=0)

        # x = gocs_pred[partidx[idx+1], 3 * j:3 * (j + 1)]  # N * 3
        # y = nocs_pred[partidx[idx+1], 3 * j:3 * (j + 1)]  # N * 3
        # scale1 = np.std(np.mean(y, axis=1)) / np.std(np.mean(x, axis=1))
        # translation1 = np.mean(y - scale1 * x, axis=0)

        heatmap_pred = result['heatmap_per_point'][0].cpu().numpy()
        heatmap_pred = heatmap_pred[:, j]
        unitvec_pred = result['unitvec_per_point'][0].cpu().numpy()
        unitvec_pred = unitvec_pred[:, j * 3:(j + 1) * 3]
        gocs_pred_fuse = np.zeros((gocs_pred.shape[0], 3))
        for i, p in enumerate(label_map):
            gocs_pred_fuse[partidx[i], :] = gocs_pred[partidx[i], 3 * p:3 * (p + 1)]

        thres_r = 0.2
        offset = unitvec_pred * (1 - heatmap_pred.reshape(-1, 1)) * thres_r
        joint_pts = gocs_pred_fuse + offset
        joint_pt = np.mean(joint_pts[joint_idx[idx+1]], axis=0)

        joint_pt = joint_pt * scale_g_to_p + translation_g_to_p
        joint_cam = dict()
        joint_cam['p'] = np.dot(best_model['scale0'] * joint_pt.reshape(1, 3), best_model['rotation0'].T) + best_model['translation0']
        joint_cam['l'] = np.dot(jt_axis.reshape(1, 3), best_model['rotation0'].T)

        refined_result['Joint'][j] = joint_cam

    return refined_result