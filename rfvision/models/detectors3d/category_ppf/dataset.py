import glob
import os
import numpy as np
from tqdm import tqdm
import pickle
import cv2
from utils import *
import torch
class NOCSForPPF:
    def __init__(self,
                 data_root,
                 category=2,
                 ):

        intrinsics = np.array([[591.0125, 0, 322.525], [0, 590.16775, 244.11084], [0, 0, 1]])

        self.category = category
        category_cfg = category_cfgs[category]

        log_dir = os.path.join(data_root, 'real_test_20210511T2129')
        result_pkl_list = glob.glob(os.path.join(log_dir, 'results_*.pkl'))
        result_pkl_list = sorted(result_pkl_list)[:]

        self.final_results = []
        for pkl_path in tqdm(result_pkl_list):
            with open(pkl_path, 'rb') as f:
                result = pickle.load(f)
                # print(result)
                if not 'gt_handle_visibility' in result:
                    result['gt_handle_visibility'] = np.ones_like(result['gt_class_ids'])
                    print('can\'t find gt_handle_visibility in the pkl.')
                else:
                    assert len(result['gt_handle_visibility']) == len(result['gt_class_ids']), "{} {}".format(
                        result['gt_handle_visibility'], result['gt_class_ids'])

            if type(result) is list:
                self.final_results += result
            elif type(result) is dict:
                self.final_results.append(result)
            else:
                assert False

        self.pcs = []
        self.pcs_normal = []

        for res in tqdm(self.final_results):
            res['image_path'] = '/hdd0/data/ppf_dataset' + res['image_path'][4:]
            img = cv2.imread(res['image_path'] + '_color.png')[:, :, ::-1]
            depth = cv2.imread(res['image_path'] + '_depth.png', -1)

            draw_img = img.copy()
            draw_image_bbox = img.copy()
            bboxs = res['pred_bboxes']
            masks = res['pred_masks']
            RTs = res['pred_RTs']
            scales = res['pred_scales']
            cls_ids = res['pred_class_ids']
            for i, bbox in enumerate(bboxs):
                if cls_ids[i] != self.category:
                    continue
                else:
                    pc, idxs = backproject(depth, intrinsics, masks[:, :, i])
                    pc /= 1000
                    # augment
                    pc = pc + np.clip(category_cfg['res'] / 4 * np.random.randn(*pc.shape), -category_cfg['res'] / 2, category_cfg['res'] / 2)

                    pc[:, 0] = -pc[:, 0]
                    pc[:, 1] = -pc[:, 1]
                    pc = pc_downsample(pc, 0.004)
                    pc = pc.astype('float32')
                    # high_res_indices = ME.utils.sparse_quantize(np.ascontiguousarray(pc), return_index=True, quantization_size=cfg.res)[1]
                    # pc = pc[high_res_indices].astype(np.float32)
                    pc_normal = estimate_normals(pc, category_cfg['knn']).astype(np.float32)

                    self.pcs.append(torch.FloatTensor(pc[None]))
                    self.pcs_normal.append(torch.FloatTensor(pc_normal[None]))

    def __len__(self):
        return len(self.final_results)

    def __getitem__(self, index):
        return self.pcs[index], self.pcs_normal

    def evaluate(self,
                 results):
        for i in len(self):
            ppf_pred = results[0]


if __name__ == '__main__':
    pass
    # dataset = NOCSForPPF(data_root='/hdd0/data/ppf_dataset')
    # from rfvision.tools.debug_tools import debug_model
    # m = debug_model('/home/hanyang/rfvision/flows/detectors3d/category_ppf/cfg.py',
    #                 '/home/hanyang/rfvision/work_dirs/ppf/epoch_200.pth').to(0)
    # res = m.forward_test(dataset.pcs[1].to(0), dataset.pcs_normal[1].to(0))