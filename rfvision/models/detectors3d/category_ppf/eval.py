from rfvision.models.detectors3d.category_ppf.category_ppf import CategoryPPF
from rfvision.models.detectors3d.category_ppf.utils.utils import pc_downsample, estimate_normals, backproject
from rfvision.models.detectors3d.category_ppf.utils.eval_utils import compute_degree_cm_mAP
from rflib.runner import load_checkpoint
import os
import glob
from tqdm import tqdm
import pickle
import cv2
import numpy as np
import torch

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
        m = CategoryPPF(category=i).cuda().eval()
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
    data_root = '/disk1/data/ppf_dataset'
    final_results = get_nocs_pred(data_root)
    ppf_models = get_model_for_all_categories(ppf_work_dir='/home/hanyang/rfvision/work_dir/category_ppf',
                                              epoch_id=200)
    intrinsics = np.float32([[591.0125, 0, 320],[0, 590.16775, 240],[0, 0, 1]])
    out_dir = './'
    final_preds = {}
    ppf_model = ppf_models[1]
    for i , res in enumerate(tqdm(final_results)):
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
            # ppf_model = ppf_models[cls_id]

            pc, idxs = backproject(depth, intrinsics, masks[:, :, i])
            pc /= 1000
            # augment
            # pc = pc + np.clip(cfg.res / 4 * np.random.randn(*pc.shape), -cfg.res / 2, cfg.res / 2)

            pc[:, 0] = -pc[:, 0]
            pc[:, 1] = -pc[:, 1]
            pc = pc_downsample(pc, 0.004)
            pc = pc.astype('float32')
            # high_res_indices = ME.utils.sparse_quantize(np.ascontiguousarray(pc), return_index=True, quantization_size=cfg.res)[1]
            # pc = pc[high_res_indices].astype(np.float32)
            pc_normal = estimate_normals(pc, 60).astype(np.float32)


            pcs = torch.from_numpy(pc[None]).cuda()
            pc_normals = torch.from_numpy(pc_normal[None]).cuda()
            ppf_preds = ppf_model.forward_test(pcs, pc_normals)
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
