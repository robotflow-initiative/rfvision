"""
This file contains some demo for you to load data into your CNN.
I provide a file named `source_directories.txt`, which includes all directories containing .ply file and .json file.
Note that the directory is a relative path under ALFRED_ROOT/data/json_2.1.0/
You should focus on `unityExportOpt.ply`, `traj_data.json` and `object_annotations.json`. File `unityExportOpt.ply`
contains point cloud file, `traj_data.json` contains natural language instructions and high/low level actions, and
`object_annotations.json` includes each object's id and its corresponding 3D bbox.

The following functions will show you how to read the above data.
"""

import os
import open3d as o3d
import numpy as np
import json
import copy


source_directories_file = 'source_directories.txt'

# FIXME: You need to modify this variable to your local alfred root
ALFRED_ROOT = '/home/sjtu/inscs/mmdetection3d/'

sub_dir = 'data/alfred/json_2.1.0'

instruction_annotation_file = 'traj_data.json'
object_annotation_file = 'object_annotations.json'
point_cloud_file = 'unityExportOpt.ply'


def get_all_absolute_directories():
    """
    This function will return all absolute directories which contains .pcd and .json into a list.
    """
    abs_dirs = []
    with open(source_directories_file, 'r') as f:
        contents = f.readlines()
        for line in contents:
            relative_dir = line.split()[0]
            abs_dir = os.path.join(ALFRED_ROOT, sub_dir, relative_dir)
            abs_dirs.append(abs_dir)

    return abs_dirs


def get_3d_bbox(abs_dir):
    """
    This function needs an absolute directory as input, and returns a dict. The keys include each object's id, and the
    value is object's 3D bounding box wrapped into a numpy array in shape (8, 3). Note that each object's id is in
    format `ObjectName|x|y|z`.
    """
    annotation_file = os.path.join(abs_dir, object_annotation_file)
    raw_annotation = dict()
    with open(annotation_file, 'r') as f:
        raw_annotation = json.load(f)

    object_ids = list(raw_annotation.keys())
    annotation_object_ids = object_ids
    bbox_annotation = dict()

    for object_id in annotation_object_ids:
        bbox = raw_annotation[object_id]['3dBbox']
        points = []
        for point in bbox:
            # Here we use use negative x to align axis.
            points.append([-point['x'], point['y'], point['z']])
        bbox_annotation[object_id] = np.array(points)

    return bbox_annotation


def get_point_cloud(abs_dir):
    """
    This function needs an absolute directory as input, and returns a numpy array in shape (N, 6).
    """
    pcd_file = os.path.join(abs_dir, point_cloud_file)
    point_cloud = o3d.io.read_point_cloud(pcd_file)

    xyz = np.asarray(point_cloud.points)
    rgb = np.asarray(point_cloud.colors)

    return np.concatenate((xyz, rgb), axis=1)


def get_task_descriptions(abs_dir):
    """
    This function needs an absolute directory as input, and returns a list of strings containing this task's natural
    language description. Note that the list length is 3.
    """
    traj_path = os.path.join(abs_dir, instruction_annotation_file)
    traj_annotation = dict()
    with open(traj_path, 'r') as f:
        traj_annotation = json.load(f)

    task_descriptions = []
    raw_descriptions = traj_annotation['turk_annotations']['anns']
    for desc in raw_descriptions:
        task_descriptions.append(desc['task_desc'])

    return task_descriptions


def get_high_desctiptions(abs_dir):
    """
    This function needs an absolute directory as input, and returns a list of `string list` containing this task's
    natural language step-by-step instructions. Note that the list length is 3, while each `string list`'s length
    is not fixed.
    """
    traj_path = os.path.join(abs_dir, instruction_annotation_file)
    traj_annotation = dict()
    with open(traj_path, 'r') as f:
        traj_annotation = json.load(f)

    task_instructions = []
    raw_descriptions = traj_annotation['turk_annotations']['anns']
    for desc in raw_descriptions:
        task_instructions.append(desc['high_descs'])

    return task_instructions


def get_high_level_action_sequence(abs_dir):
    """
    This function needs an absolute directory as input, and returns a list of high-level actions as ground truth.
    Note that for `test_seen` and `test_unseen` folders, there's no ground truth action sequences.
    """
    traj_path = os.path.join(abs_dir, instruction_annotation_file)
    traj_annotation = dict()
    with open(traj_path, 'r') as f:
        traj_annotation = json.load(f)

    return copy.deepcopy(traj_annotation['plan']['high_pddl'])


def get_low_level_action_sequence(abs_dir):
    """
    This function needs an absolute directory as input, and returns a list of low-level actions as ground truth.
    Note that for `test_seen` and `test_unseen` folders, there's no ground truth action sequences.
    """
    traj_path = os.path.join(abs_dir, instruction_annotation_file)
    traj_annotation = dict()
    with open(traj_path, 'r') as f:
        traj_annotation = json.load(f)

    return copy.deepcopy(traj_annotation['plan']['low_actions'])


def object_type2color(object_type):
    return [0, 1, 0]


def vis_bbox(abs_dir):
    bbox_annotation = get_3d_bbox(abs_dir)
    pc_npy = get_point_cloud(abs_dir)
    o3d_vis_contents = []

    lines = [
        [0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6],
        [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]
    ]
    # Draw bbox
    for object_id, bbox in bbox_annotation.items():
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(bbox),
            lines=o3d.utility.Vector2iVector(lines)
        )

        # TODO: Apply each object type to different color, i.e., optimize object_type2color() function
        object_type = object_id.split('|')[0]
        color = object_type2color(object_type)
        colors = [color for i in range(len(lines))]
        line_set.colors = o3d.utility.Vector3dVector(colors)
        o3d_vis_contents.append(line_set)

    # Draw point cloud
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pc_npy[:, :3])
    pc.colors = o3d.utility.Vector3dVector(pc_npy[:, 3:6])
    o3d_vis_contents.append(pc)

    o3d.visualization.draw_geometries(o3d_vis_contents, window_name="Open3D")


if __name__ == '__main__':
    dirs = get_all_absolute_directories()
    print(get_3d_bbox(dirs[0]))
    print(get_3d_bbox(dirs[0])['WateringCan|-00.21|00.00|+03.23'].shape)
    print(get_point_cloud(dirs[0]).shape)
    print(get_task_descriptions(dirs[0]))
    print(get_high_desctiptions(dirs[0]))
    print(get_high_level_action_sequence(dirs[0]))
    print(get_low_level_action_sequence(dirs[0]))
    vis_bbox(dirs[0])
