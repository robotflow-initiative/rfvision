from rfvision.models.detectors3d.category_ppf.utils.eval_utils import compute_degree_cm_mAP
from rfvision.models.detectors3d.category_ppf.category_ppf import CategoryPPF
from rflib.runner import load_checkpoint
import os
import glob
from tqdm import tqdm
import pickle
import numpy as np
import torch
import cv2

def get_nocs_pred(data_root):
    log_dir = os.path.join(data_root, 'real_test_20210511T2129')
    result_pkl_list = glob.glob(os.path.join(log_dir, 'results_*.pkl'))
    result_pkl_list = sorted(result_pkl_list)[:]

    final_results = []
    for pkl_path in tqdm(result_pkl_list):
        with open(pkl_path, 'rb') as f:
            result = pickle.load(f)
            if not 'gt_handle_visibility' in result:
                result['gt_handle_visibility'] = np.ones_like(result['gt_class_ids'])
                print('can\'t find gt_handle_visibility in the pkl.')
            else:
                assert len(result['gt_handle_visibility']) == len(result['gt_class_ids']), "{} {}".format(
                    result['gt_handle_visibility'], result['gt_class_ids'])

        if type(result) is list:
            final_results += result
        elif type(result) is dict:
            final_results.append(result)
        else:
            assert False
    return final_results

def get_model_for_all_categories(ppf_work_dir, epoch_id=200):
    model_for_all_categories = {}
    for i in range(1, 7):
        m = CategoryPPF(category=i)
        checkpoint_path= os.path.join(ppf_work_dir, f'category{str(i)}', f'epoch_{str(epoch_id)}.pth')
        load_checkpoint(m, checkpoint_path)
        model_for_all_categories[i] = m
    return model_for_all_categories


synset_names = ['BG', #0
                'bottle', #1
                'bowl', #2
                'camera', #3
                'can',  #4
                'laptop',#5
                'mug'#6
                ]

synset_names_inv = dict([(k, v) for v, k in enumerate(synset_names)])

if __name__ == '__main__':
    data_root = '/hdd0/data/ppf_dataset'
    final_results = get_nocs_pred(data_root)
    ppf_models = get_model_for_all_categories(ppf_work_dir='/home/hanyang/rfvision/work_dir/category_ppf',
                                              epoch_id=200)
    final_preds = {}

    for i , res in tqdm(enumerate(final_results)):
        res['image_path'] = data_root + res['image_path'][4:]
        img = cv2.imread(res['image_path'] + '_color.png')[:, :, ::-1]
        depth = cv2.imread(res['image_path'] + '_depth.png', -1)

        draw_img = img.copy()
        draw_image_bbox = img.copy()
        bboxs = res['pred_bboxes']
        masks = res['pred_masks']
        RTs = res['pred_RTs']
        scales = res['pred_scales']
        cls_ids = res['pred_class_ids']
        gt_RTs = res['gt_RTs']
        gt_cls_ids = res['gt_class_ids']


        ppf_pred_RT = []
        ppf_pred_scale = []

        for i, bbox in enumerate(bboxs):
            cv2.rectangle(draw_img,
                        (bbox[1], bbox[0]),
                        (bbox[3], bbox[2]),
                        (255, 0, 0), 2)
            draw_img[masks[:, :, i]] = np.array([0, 255, 0])
            cls_id = cls_ids[i]
            cfg = cfgs[cls_id]
            point_encoder = point_encoders[cls_id]
            ppf_encoder = ppf_encoders[cls_id]

            pc, idxs = backproject(depth, intrinsics, masks[:, :, i])
            pc /= 1000
            # augment
            pc = pc + np.clip(cfg.res / 4 * np.random.randn(*pc.shape), -cfg.res / 2, cfg.res / 2)

            pc[:, 0] = -pc[:, 0]
            pc[:, 1] = -pc[:, 1]
            pc, index = pc_downsample(pc, 0.004)
            pc = pc.astype('float32')
            # high_res_indices = ME.utils.sparse_quantize(np.ascontiguousarray(pc), return_index=True, quantization_size=cfg.res)[1]
            # pc = pc[high_res_indices].astype(np.float32)
            pc_normal = estimate_normals(pc, cfg.knn).astype(np.float32)

            if cls_id == 5:
                continue
                mask_idxs = np.where(masks[:, :, i])
                bbox = np.array([
                    [np.min(mask_idxs[0]), np.max(mask_idxs[0])],
                    [np.min(mask_idxs[1]), np.max(mask_idxs[1])]
                ])

                rgb_obj = np.zeros_like(img, dtype=np.float32)
                rgb_obj[mask_idxs[0], mask_idxs[1]] = img[mask_idxs[0], mask_idxs[1]] / 255.
                rgb_cropped = cv2.resize(rgb_obj[bbox[0][0]:bbox[0][1]+1, bbox[1][0]:bbox[1][1]+1], (224, 224))
                resize_scale = 224 / (bbox[:, 1] - bbox[:, 0])

                pc_xy = np.stack(idxs, -1)
                idxs_resized = np.clip(((pc_xy - bbox[:, 0]) * resize_scale).astype(np.int64), 0, 223)

                output = laptop_aux(torch.from_numpy(rgb_cropped[None]).cuda().permute(0, 3, 1, 2))['out']
                preds_laptop_aux = output[0].argmax(0).cpu().numpy()
                pc_img_indices = idxs_resized[high_res_indices]
                preds_laptop_aux = preds_laptop_aux[pc_img_indices[:, 0], pc_img_indices[:, 1]]
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pc[preds_laptop_aux == 0])
                if (preds_laptop_aux == 0).sum() < 10:
                    laptop_up = None
                else:
                    plane, inlier = pcd.segment_plane(distance_threshold=0.02,
                                                ransac_n=3,
                                                num_iterations=100)
                    laptop_up = plane[:3]

            pcs = torch.from_numpy(pc[None]).cuda()
            pc_normals = torch.from_numpy(pc_normal[None]).cuda()
            point_idxs = np.random.randint(0, pc.shape[0], (100000, 2))

            ppf_preds = ppf_models[i].forward_test(pc, pc_normal)
            ppf_pred_RT.append(ppf_preds['pred_RT'])
            ppf_pred_scale.append(ppf_preds['pred_scale'])
        ppf_pred_RT = np.array(ppf_pred_RT)
        ppf_pred_scale = np.array(ppf_pred_scale)

        final_results[i]['pred_RTs'] = ppf_pred_RT
        final_results[i]['pred_scales'] = ppf_pred_scale
    compute_degree_cm_mAP(final_results, synset_names, out_dir + '_plots',
                          degree_thresholds=[5, 10, 15],  # range(0, 61, 1),
                          shift_thresholds=[5, 10, 15],  # np.linspace(0, 1, 31)*15,
                          iou_3d_thresholds=np.linspace(0, 1, 101),
                          iou_pose_thres=0.1,
                          use_matches_for_pose=True)