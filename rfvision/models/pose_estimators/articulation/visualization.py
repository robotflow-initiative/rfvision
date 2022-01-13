from rflib import load_checkpoint
import torch
import json
from rfvision.models.pose_estimators.articulation.datasets.articulation_dataset import test_pipelines
from rfvision.datasets.pipelines import Compose
from rfvision.models.pose_estimators.articulation.models.optimizer import optimize_pose
from rfvision.models.pose_estimators.articulation.models.vis_pose import vis_pose
from rfvision.models.pose_estimators.articulation import ArticulationEstimator

import argparse
import os
import json
import torch


def parse_args():
    parser = argparse.ArgumentParser(description='Articulation test Estimator')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('data_dir', help='demo data directory')
    parser.add_argument('det_bbox_file', help='detected bounding box file')
    parser.add_argument('--use_gpu', default=True, help='use gpu for inference')
    parser.add_argument('--n_max_parts', default=13, type=int, help='use rgb as point feature')
    parser.add_argument('--out', default='none', help='output result file')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    # build estimator
    estimator = ArticulationEstimator(in_channels=3,
                                      n_max_parts=args.n_max_parts)

    # load checkpoint
    load_checkpoint(estimator, args.checkpoint)
    if args.use_gpu:
        estimator.cuda()
    estimator.eval()

    # prepare data
    device = next(estimator.parameters()).device
    test_pipeline = Compose(test_pipelines)
    color_list = os.listdir(os.path.join(args.data_dir, 'color'))
    color_list.sort()
    depth_list = os.listdir(os.path.join(args.data_dir, 'depth'))
    depth_list.sort()
    camera_intrinsic_path = os.path.join(args.data_dir, 'camera_intrinsic.json')
    if not os.path.lexists(camera_intrinsic_path):
        raise FileNotFoundError('camera intrinsic {} not found, break!'.format(camera_intrinsic_path))

    test_id = 1
    color_image = color_list[test_id]
    depth_image = color_image.replace('.jpg', '.png')
    data = dict(color_path=os.path.join(args.data_dir, 'color', color_image),
                depth_path=os.path.join(args.data_dir, 'depth', depth_image),
                camera_intrinsic_path=camera_intrinsic_path)
    det_bboxes_total = json.load(open(args.det_bbox_file, 'r'))
    det_bboxes = [b for b in det_bboxes_total if b['image_file_name'] == color_image]
    if len(det_bboxes) == 0:
        raise AssertionError('sample {} do not contain any detected instances, break!'.format(color_image))

    refined_results = []
    for det_bbox in det_bboxes:
        data.update(dict(bbox=det_bbox['bbox'], category_id=det_bbox['category_id'], urdf_id=det_bbox['urdf_id']))
        data = test_pipeline(data)
        # forward the model
        if args.use_gpu:
            data['pts'] = data['pts'].cuda()
            data['pts_feature'] = data['pts_feature'].cuda()
        with torch.no_grad():
            result = estimator(return_loss=False, rescale=True, **data)
        result.update(data)

        refined_result = optimize_pose(result)
        refined_result.update(dict(bbox=det_bbox['bbox']))
        refined_results.append(refined_result)

    # visualization
    vis_pose(refined_results, data)


if __name__ == '__main__':
    main()