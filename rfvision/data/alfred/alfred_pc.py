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
import math
import open3d as o3d
import numpy as np
import json
import copy


class AlfredPC:
    def __init__(self, source_file_path, root_path):
        self.source_directories_file = source_file_path
        self.root = root_path
        self.instruction_annotation_file = 'traj_data.json'
        self.object_annotation_file = 'object_annotations.json'
        self.point_cloud_file = 'unityExportOpt.ply'
        self.all_abs_dirs = self.get_all_absolute_directories()

    def get_all_absolute_directories(self):
        """
            This function will return all absolute directories which contains .pcd and .json into a list.
            """
        abs_dirs = []
        with open(self.source_directories_file, 'r') as f:
            contents = f.readlines()
            for line in contents:
                try:
                    relative_dir = line.split()[0]
                    abs_dir = os.path.join(self.root, relative_dir)
                    abs_dirs.append(abs_dir)
                except:
                    pass

        return abs_dirs

    def get_object(self, abs_dir):
        """
        This function needs an absolute directory as input, and returns a dict. The keys include each object's id, and the
        value is object's annotations of `desc`.
        """
        annotation_file = os.path.join(abs_dir, self.object_annotation_file)
        raw_annotation = self._read_json(annotation_file)

        object_ids = list(raw_annotation.keys())
        annotation_object_ids = object_ids
        anno = []

        for object_id in annotation_object_ids:
            if raw_annotation[object_id]['3dBbox'] is None:
                continue
            anno.append(raw_annotation[object_id]['objectType'])

        return anno

    def get_3d_bbox(self, abs_dir):
        """
        This function needs an absolute directory as input, and returns a dict. The keys include each object's id, and the
        value is object's 3D bounding box wrapped into a numpy array in shape (7,). Note that each object's id is in
        format `ObjectName|x|y|z`.
        """
        annotation_file = os.path.join(abs_dir, self.object_annotation_file)
        raw_annotation = self._read_json(annotation_file)

        object_ids = list(raw_annotation.keys())
        annotation_object_ids = object_ids
        bbox_annotation = dict()

        for object_id in annotation_object_ids:
            bbox = raw_annotation[object_id]['3dBbox']
            if bbox is None:
                continue
            points = []
            for point in bbox:
                # Here we use use negative x to align axis.
                points.append([-point['x'], point['y'], point['z']])

            # Convert bbox to (x, y, z, l, w, h, yaw) format
            oriented_bbox = o3d.geometry.OrientedBoundingBox.create_from_points(
                o3d.utility.Vector3dVector(points))
            center = np.asarray(oriented_bbox.center)
            extent = np.asarray(oriented_bbox.extent)
            rotation_mat = np.asarray(oriented_bbox.R)
            yaw = np.asarray(
                [math.atan2(rotation_mat[1][0], rotation_mat[0][0])])

            bbox_annotation[object_id] = np.concatenate((center, extent, yaw))

        return bbox_annotation

    def get_point_cloud(self, abs_dir):
        """
        This function needs an absolute directory as input, and returns a numpy array in shape (N, 6) containing
        xyz information and rgb information. Note that xyz is the absolute coordinates, and rgb in range [0, 1].
        """
        pcd_file = os.path.join(abs_dir, self.point_cloud_file)
        point_cloud = o3d.io.read_point_cloud(pcd_file)

        xyz = np.asarray(point_cloud.points)
        rgb = np.asarray(point_cloud.colors)

        return np.concatenate((xyz, rgb), axis=1)

    def get_point_cloud_norgb(self, abs_dir):
        """
        This function needs an absolute directory as input, and returns a numpy array in shape (N, 3) containing
        only xyz information. Note that xyz is the absolute coordinates.
        """
        pcd_file = os.path.join(abs_dir, self.point_cloud_file)
        point_cloud = o3d.io.read_point_cloud(pcd_file)

        xyz = np.asarray(point_cloud.points)

        return xyz

    def get_task_descriptions(self, abs_dir):
        """
        This function needs an absolute directory as input, and returns a list of strings containing this task's natural
        language description. Note that the list length is 3.
        """
        traj_path = os.path.join(abs_dir, self.instruction_annotation_file)
        traj_annotation = self._read_json(traj_path)

        task_descriptions = []
        raw_descriptions = traj_annotation['turk_annotations']['anns']
        for desc in raw_descriptions:
            task_descriptions.append(desc['task_desc'])

        return task_descriptions

    def get_high_descriptions(self, abs_dir):
        """
        This function needs an absolute directory as input, and returns a list of `string list` containing this task's
        natural language step-by-step instructions. Note that the list length is 3, while each `string list`'s length
        is not fixed.
        """
        traj_path = os.path.join(abs_dir, self.instruction_annotation_file)
        traj_annotation = self._read_json(traj_path)

        task_instructions = []
        raw_descriptions = traj_annotation['turk_annotations']['anns']
        for desc in raw_descriptions:
            task_instructions.append(desc['high_descs'])

        return task_instructions

    def get_high_level_action_sequence(self, abs_dir):
        """
        This function needs an absolute directory as input, and returns a list of high-level actions as ground truth.
        Note that for `test_seen` and `test_unseen` folders, there's no ground truth action sequences.
        """
        traj_path = os.path.join(abs_dir, self.instruction_annotation_file)
        traj_annotation = self._read_json(traj_path)

        if 'plan' not in traj_annotation.keys():
            return None

        return copy.deepcopy(traj_annotation['plan']['high_pddl'])

    def get_low_level_action_sequence(self, abs_dir):
        """
        This function needs an absolute directory as input, and returns a list of low-level actions as ground truth.
        Note that for `test_seen` and `test_unseen` folders, there's no ground truth action sequences.
        """
        traj_path = os.path.join(abs_dir, self.instruction_annotation_file)
        traj_annotation = self._read_json(traj_path)

        if 'plan' not in traj_annotation.keys():
            return None

        return copy.deepcopy(traj_annotation['plan']['low_actions'])

    def vis_pc(self, abs_dir, bbox=False):
        """
        This function needs an absolute directory as input, and visualize the point cloud in the given directory. You
        can use parameter `bbox` to determine whether show bounding box.
        """
        bbox_annotation = self.get_3d_bbox(abs_dir)
        pc_npy = self.get_point_cloud(abs_dir)
        o3d_vis_contents = []

        if bbox:
            # Draw 3D bounding box
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

                object_type = object_id.split('|')[0]
                color = self._obj_type_2_color(object_type)
                colors = [color for i in range(len(lines))]
                line_set.colors = o3d.utility.Vector3dVector(colors)
                o3d_vis_contents.append(line_set)

        # Draw point cloud
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(pc_npy[:, :3])
        pc.colors = o3d.utility.Vector3dVector(pc_npy[:, 3:6])
        o3d_vis_contents.append(pc)

        o3d.visualization.draw_geometries(
            o3d_vis_contents, window_name="Open3D")

    def _read_json(self, json_file_path):
        """
        This function will parse a json file and return a dict.
        """
        with open(json_file_path, 'r') as f:
            json_dict = json.load(f)

        return copy.deepcopy(json_dict)

    def _obj_type_2_color(self, object_type):
        """
        This function will assign each object_type a unique color, for the visualization of bounding box.
        """
        # TODO: Apply each object type to different color, i.e., optimize object_type2color() function
        return [0, 1, 0]

