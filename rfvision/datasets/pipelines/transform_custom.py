import pickle
import sklearn
import cv2
import numpy as np
from rfvision.datasets.builder import PIPELINES
from PIL import Image

def generate_heatmap_2d(uv, heatmap_shape, sigma=7):
    '''
    Generate one heatmap in range (0, 1).
    Args:
        uv: single pixel coordinate, shape (1, 2),
        heatmap_shape: output shape of heatmap, tuple or list, typically: (256, 256)
        sigma:Gaussian sigma

    Returns:heatmap

    '''
    hm = np.zeros(heatmap_shape)
    hm[uv[1], uv[0]] = 1
    hm = cv2.GaussianBlur(hm, (sigma, sigma), 0)
    hm /= hm.max()  # normalize hm to [0, 1]
    return hm # outshape



@PIPELINES.register_module()
class GenerateCoef(object):
    def __init__(self, base_root, use_mask_bbox=False, scale=64, method='None', num_bases=-1,
                 keep_resized_mask=False, preserve_gt_mask=False):
        if sklearn.__version__ != '0.21.3':
            raise RuntimeError('sklearn version 0.21.3 is required. However get %s' % sklearn.__version__)
        with open(base_root, 'rb') as dico_file:
            self.dico = pickle.load(dico_file)
        self.dico.set_params(n_jobs=1)

        self.use_mask_bbox = use_mask_bbox
        self.scale = scale
        if method not in ['cosine', 'cosine_r']:
            raise NotImplementedError('%s not supported.' % method)
        self.method = method
        self.num_bases = num_bases
        self.keep_resized_mask = keep_resized_mask
        self.preserve_gt_mask = preserve_gt_mask

    @staticmethod
    def get_bbox(mask):
        coords = np.transpose(np.nonzero(mask))
        y, x, h, w = cv2.boundingRect(coords)
        return x, y, w+1, h+1

    def __call__(self, results):
        scale = self.scale
        if 'gt_masks' not in results:
            raise RuntimeError('`gt_masks` is missing')
        new_gt_bboxes = []
        resized_gt_masks = []
        coef_gt = []
        for mask, bbox in zip(results['gt_masks'], results['gt_bboxes']):
            if self.use_mask_bbox:
                x1, y1, w, h = self.get_bbox(mask)
                x2 = x1 + w
                y2 = y1 + h
                new_bbox = [x1, y1, x2, y2]
                new_gt_bboxes.append(new_bbox)
            else:
                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2])+1, int(bbox[3])+1  # hot fix
                assert(x1 <= x2 and y1 <= y2)

            obj_mask = mask[y1:y2, x1:x2].astype(np.uint8)  # {0, 1}

            # resized_mask = cv.resize(resized_mask, (scale, scale), interpolation=cv.INTER_NEAREST)
            resized_mask = Image.fromarray(obj_mask).resize((scale, scale), Image.NEAREST)
            resized_mask = np.reshape(resized_mask, (1, scale*scale))

            if self.method == 'cosine' or self.method == 'cosine_r':
                resized_mask = resized_mask.astype(np.int) * 2 - 1  # {-1, 1}

            coef = self.dico.transform(resized_mask)[0]  # TODO: Catch these warning
            if self.method == 'cosine_r':
                coef[0] /= 30.0
                coef[1] /= 10.0

            resized_gt_masks.append(resized_mask[0])
            assert coef.shape[0] == self.num_bases
            coef_gt.append(coef)

        if self.use_mask_bbox:
            results['gt_bboxes'] = np.stack(new_gt_bboxes)
        if self.keep_resized_mask:
            results['gt_resized_masks'] = np.stack(resized_gt_masks)  # should be {-1, 1} int Mask
        if not self.preserve_gt_mask:
            results.pop('gt_masks')  # No longer needed

        if len(coef_gt) == 0:
            results['gt_coefs'] = []
        else:
            results['gt_coefs'] = np.stack(coef_gt)

        return results


class JointsDepthProcessing:
    def __init__(self,
                 depth_scale,
                 depth_bound,
                 root_joint_id=0,
                 ):
        self.max_bound = depth_bound[1]
        self.min_bound = depth_bound[0]
        assert self.max_bound > self.min_bound
        self.depth_scale = depth_scale
        self.root_joint_id = root_joint_id
        self.depth_bound = depth_bound
    def __call__(self,
                 results):
        joints_depth = results['joints_cam'][:, 2]
        joints_depth = (joints_depth - self.min_bound) / (self.max_bound - self.min_bound)
        joints_depth /= self.depth_scale
        joints_depth -= joints_depth[self.root_joint_id]
        results['joints_depth'] = joints_depth
        return joints_depth
