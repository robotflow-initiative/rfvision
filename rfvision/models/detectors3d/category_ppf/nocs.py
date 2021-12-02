import torch
import os
import glob
import tqdm
import pickle
import numpy as np
class NOCSForPPF(torch.utils.data.Dataset):
    def __init__(self,
                 data_root,
                 scene_id=1,
                 category=2,
                 pipeline=None,
                 test_mode=False,
                 ):
        self.data_root = data_root

        log_dir = os.path.join(self.data_root, 'real_test_20210511T2129')
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


