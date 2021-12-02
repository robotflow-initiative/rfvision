# Modified from
# https://github.com/facebookresearch/votenet/blob/master/alfred/batch_load_alfred_data.py
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Batch mode in loading alfred scenes with vertices and ground truth labels for
semantic and instance segmentations.

Usage example: python ./batch_load_alfred_data.py
"""

import argparse
import argparse
import datetime
import inspect
import numpy as np
import os
from os import path as osp

from load_alfred_data import export
from alfred_pc import AlfredPC

currentdir = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe())))

DONOTCARE_CLASS_IDS = np.array([])
OBJ_CLASS_IDS = np.array(
    [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])


def batch_export(max_num_point,
                 output_folder,
                 alfred_names_file,
                 alfred_dir,
                 need_rgb,
                 test_mode=False):

    # Init the object of AlfredPC
    pc = AlfredPC(alfred_names_file, alfred_dir)
    abs_dirs = pc.get_all_absolute_directories()

    if test_mode and not os.path.exists(alfred_dir):
        # test data preparation is optional
        return
    if not os.path.exists(output_folder):
        print(f'Creating new data folder: {output_folder}')
        os.mkdir(output_folder)

    for abs_dir in abs_dirs:
        output_filename_prefix = osp.join(output_folder, abs_dir.split('/')[-1])
        if not need_rgb:
            output_filename_prefix = osp.join(output_filename_prefix, '_norgb')
        if osp.isfile(f'{output_filename_prefix}_vert.npy'):
            continue
        export(pc, abs_dir, output_filename_prefix, need_rgb, test_mode)
    
    print('Load success!')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--max_num_point',
        default=None,
        help='The maximum number of the points.')
    parser.add_argument(
        '--output_folder',
        default='./alfred_instance_data',
        help='output folder of the result.')
    parser.add_argument(
        '--alfred_dir', default='json_2.1.0', help='alfred data directory.')
    parser.add_argument(
        '--alfred_names_file',
        default='meta_data/train.txt',
        help='The path of the file that stores the alfred names.')
    parser.add_argument(
        '--need_rgb',
        default=False,
        help='Whether RGB data is needed.')
    parser.add_argument(
        '--test',
        default=False,
        help='Whether test data is needed.')
    args = parser.parse_args()
    batch_export(
        args.max_num_point,
        args.output_folder,
        args.alfred_names_file,
        args.alfred_dir,
        args.need_rgb,
        args.test
    )


if __name__ == '__main__':
    main()
