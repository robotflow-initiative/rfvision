# Modified from
# https://github.com/facebookresearch/votenet/blob/master/scannet/load_scannet_data.py
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Load Alfred scenes with vertices and ground truth labels for semantic and
instance segmentations.
"""

import numpy as np

def export(pc, abs_dir, output_file=None, need_rgb=True, test_mode=False):
    """Export original files to vert, ins_label, sem_label and bbox file.

    Args:
        mesh_file (str): Path of the mesh_file.
        output_file (str): Path of the output folder.
            Default: None.
        need_rgb (str): Whether rgb data is needed.
            Default: True.
        test_mode (bool): Whether is generating test data without labels.
            Default: False.

    It returns a tuple, which containts the the following things:
        np.ndarray: Vertices of points data.
        np.ndarray: Instance bboxes.
    """

    if need_rgb:
        mesh_vertices = pc.get_point_cloud(abs_dir)
    else:
        mesh_vertices = pc.get_point_cloud_norgb(abs_dir)
    
    bboxes = pc.get_3d_bbox(abs_dir)
    bboxes_arr = []
    for bbox in bboxes:
        bboxes_arr.append(bboxes[bbox])
    bbox = np.array(bboxes_arr)
    obj = np.array(pc.get_object(abs_dir))
    assert bbox.shape[0] == obj.shape[0]
    
    assert np.isnan(bbox).any() == False
    assert np.isnan(mesh_vertices).any() == False

    if output_file is not None:
        np.save(output_file + '_vert.npy', mesh_vertices)
        np.save(output_file + '_bbox.npy', bbox)
        np.save(output_file + '_label.npy', obj)

    return mesh_vertices, bboxes, obj
