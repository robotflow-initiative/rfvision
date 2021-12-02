import os
import json
import open3d as o3d
import cv2
import numpy as np

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


def compose_rt(rotation, translation):
    aligned_RT = np.zeros((4, 4), dtype=np.float32)
    aligned_RT[:3, :3] = rotation[:3, :3]
    aligned_RT[:3, 3]  = translation
    aligned_RT[3, 3]   = 1
    return aligned_RT


def transform_coordinates_3d(coordinates, RT):
    assert coordinates.shape[0] == 3
    coordinates = np.vstack([coordinates, np.ones((1, coordinates.shape[1]), dtype=np.float32)])
    new_coordinates = RT @ coordinates
    new_coordinates = new_coordinates[:3, :] / new_coordinates[3, :]
    return new_coordinates


def calculate_2d_projections(coordinates_3d, intrinsics):
    projected_coordinates = intrinsics @ coordinates_3d
    projected_coordinates = projected_coordinates[:2, :] / projected_coordinates[2, :]
    projected_coordinates = projected_coordinates.transpose()
    projected_coordinates = np.array(projected_coordinates, dtype=np.int32)

    return projected_coordinates


def vis_pose(refined_results, input_data, out_dir='./'):
    color_path = input_data['color_path']
    camera_intrinsic_path = input_data['camera_intrinsic_path']
    all_norm_factors, all_corner_pts = fetch_factors_nocs('demo')
    urdf_id = input_data['urdf_id']

    color_image = cv2.imread(color_path)
    for refined_result in refined_results:
        camera_intrinsic = o3d.io.read_pinhole_camera_intrinsic(camera_intrinsic_path)

        category_name = INSTANCE_CLASSES[input_data['category_id']]
        label_map = label_maps[category_name][1:]

        bbox = np.array(refined_result['bbox']).astype(np.int32) + np.array(
            [-20, -20, 20, 20])  # enlarge 20 pixels for best view
        left_top = (bbox[0], bbox[1])
        right_bottom = (bbox[2], bbox[3])
        cv2.rectangle(
            color_image, left_top, right_bottom, (0, 255, 0), thickness=2)
        cv2.putText(color_image, category_name, (bbox[0], bbox[1] - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1., (0, 255, 0), 2)

        for idx, j in enumerate(label_map):
            transformation = compose_rt(refined_result['Rotation'][j], refined_result['Translation'][j])

            xyz_axis = 0.05 * np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]]).transpose()
            corner_pt = all_corner_pts[urdf_id][idx + 1]
            xyz_offset = corner_pt.mean(axis=0)
            xyz_axis += xyz_offset.reshape(3, 1)

            transformed_axes = transform_coordinates_3d(xyz_axis, transformation)
            projected_axes = calculate_2d_projections(transformed_axes, camera_intrinsic.intrinsic_matrix)

            cv2.line(color_image, tuple(projected_axes[0]), tuple(projected_axes[1]), (0, 0, 255), 3)
            cv2.line(color_image, tuple(projected_axes[0]), tuple(projected_axes[3]), (255, 0, 0), 3)
            cv2.line(color_image, tuple(projected_axes[0]), tuple(projected_axes[2]), (0, 255, 0), 3)  ## y last

            box_3d = refined_result['Box3D'][j].transpose()
            projected_box = calculate_2d_projections(box_3d, camera_intrinsic.intrinsic_matrix)

            cv2.line(color_image, tuple(projected_box[0]), tuple(projected_box[1]), (0, 255, 255), 2)
            cv2.line(color_image, tuple(projected_box[0]), tuple(projected_box[2]), (0, 255, 255), 2)
            cv2.line(color_image, tuple(projected_box[0]), tuple(projected_box[4]), (0, 255, 255), 2)
            cv2.line(color_image, tuple(projected_box[1]), tuple(projected_box[3]), (0, 255, 255), 2)
            cv2.line(color_image, tuple(projected_box[1]), tuple(projected_box[5]), (0, 255, 255), 2)
            cv2.line(color_image, tuple(projected_box[2]), tuple(projected_box[6]), (0, 255, 255), 2)
            cv2.line(color_image, tuple(projected_box[2]), tuple(projected_box[3]), (0, 255, 255), 2)
            cv2.line(color_image, tuple(projected_box[3]), tuple(projected_box[7]), (0, 255, 255), 2)
            cv2.line(color_image, tuple(projected_box[4]), tuple(projected_box[6]), (0, 255, 255), 2)
            cv2.line(color_image, tuple(projected_box[4]), tuple(projected_box[5]), (0, 255, 255), 2)
            cv2.line(color_image, tuple(projected_box[5]), tuple(projected_box[7]), (0, 255, 255), 2)
            cv2.line(color_image, tuple(projected_box[6]), tuple(projected_box[7]), (0, 255, 255), 2)

            if idx > 0:
                joint_pt = refined_result['Joint'][j]['p'].reshape(-1)
                joint_l = refined_result['Joint'][j]['l'].reshape(-1) * 0.15

                line_start = [(joint_pt[i] + joint_l[i]) for i in range(joint_pt.shape[0])]
                line_end = [(joint_pt[i] - joint_l[i]) for i in range(joint_pt.shape[0])]

                joint = np.array([line_start, line_end]).transpose()
                projected_joint = calculate_2d_projections(joint, camera_intrinsic.intrinsic_matrix)
                cv2.arrowedLine(color_image, tuple(projected_joint[1]), tuple(projected_joint[0]), (255, 0, 255), 2,
                                cv2.LINE_AA, 0, tipLength=0.08)

    cv2.imwrite(os.path.join(out_dir, 'out' + os.path.basename(color_path)), color_image)
