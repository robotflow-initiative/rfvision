from PIL import Image
import numpy as np
import numpy.ma as ma
import scipy.io as scio
import random
import torch
import torchvision.transforms as transforms

from rflib.parallel import DataContainer as DC

from ..builder import PIPELINES

border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
img_width = 480
img_length = 640


def get_bbox(label):
    rows = np.any(label, axis=1)
    cols = np.any(label, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    rmax += 1
    cmax += 1
    r_b = rmax - rmin
    for tt in range(len(border_list)):
        if r_b > border_list[tt] and r_b < border_list[tt + 1]:
            r_b = border_list[tt + 1]
            break
    c_b = cmax - cmin
    for tt in range(len(border_list)):
        if c_b > border_list[tt] and c_b < border_list[tt + 1]:
            c_b = border_list[tt + 1]
            break
    center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
    rmin = center[0] - int(r_b / 2)
    rmax = center[0] + int(r_b / 2)
    cmin = center[1] - int(c_b / 2)
    cmax = center[1] + int(c_b / 2)
    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > img_width:
        delt = rmax - img_width
        rmax = img_width
        rmin -= delt
    if cmax > img_length:
        delt = cmax - img_length
        cmax = img_length
        cmin -= delt
    return rmin, rmax, cmin, cmax


@PIPELINES.register_module()
class LoadPoseData(object):
    def __call__(self, results):
        img = Image.open(results['color_path'])
        depth = np.array(Image.open(results['depth_path']))
        label = np.array(Image.open(results['label_path']))
        meta = scio.loadmat(results['meta_path'])

        mask_back = ma.getmaskarray(ma.masked_equal(label, 0))

        results['color_img'] = img
        results['depth_img'] = depth
        results['label_img'] = label
        results['meta_mat'] = meta
        results['mask_back'] = mask_back

        return results


@PIPELINES.register_module()
class PoseImgPreprocess(object):
    def __init__(self, mode='train', front_num=2, trancolor_params = (0.2, 0.2, 0.2, 0.05), minimum_num_pt=50):
        if mode == 'train':
            self.add_noise = True
        else:
            self.add_noise = False
        self.trancolor = transforms.ColorJitter(*trancolor_params)
        self.front_num = front_num
        self.minimum_num_pt = minimum_num_pt

    def __call__(self, results):
        add_front = False
        if self.add_noise:
            random_seeds = results['random_seeds']
            for seed in random_seeds:
                front = np.array(self.trancolor(Image.open(results['color_path']).convert("RGB")))
                front = np.transpose(front, (2, 0, 1))
                f_label = np.array(Image.open('{0}/{1}-label.png'.format(results['img_prefix'], seed)))
                front_label = np.unique(f_label).tolist()[1:]
                if len(front_label) < self.front_num:
                    continue
                front_label = random.sample(front_label, self.front_num)
                for f_i in front_label:
                    mk = ma.getmaskarray(ma.masked_not_equal(f_label, f_i))
                    if f_i == front_label[0]:
                        mask_front = mk
                    else:
                        mask_front = mask_front * mk
                t_label = results['label_img'] * mask_front
                if len(t_label.nonzero()[0]) > 1000:
                    label = t_label
                    add_front = True
                    break
        else:
            label = results['label_img']

        obj = results['meta_mat']['cls_indexes'].flatten().astype(np.int32)

        while 1:
            idx = np.random.randint(0, len(obj))
            mask_depth = ma.getmaskarray(ma.masked_not_equal(results['depth_img'], 0))
            mask_label = ma.getmaskarray(ma.masked_equal(label, obj[idx]))
            mask = mask_label * mask_depth
            if len(mask.nonzero()[0]) > self.minimum_num_pt:
                break

        if self.add_noise:
            img = self.trancolor(results['color_img'])
        else:
            img = results['color_img']

        rmin, rmax, cmin, cmax = get_bbox(mask_label)
        img = np.transpose(np.array(img)[:, :, :3], (2, 0, 1))[:, rmin:rmax, cmin:cmax]

        if results['sample_name'][:8] == 'data_syn':
            seed = results['real_seed']
            back = np.array(self.trancolor(Image.open('{0}/{1}-color.png'.format(results['img_prefix'], seed)).convert("RGB")))
            back = np.transpose(back, (2, 0, 1))[:, rmin:rmax, cmin:cmax]
            img_masked = back * results['mask_back'][rmin:rmax, cmin:cmax] + img
        else:
            img_masked = img

        if self.add_noise and add_front:
            img_masked = img_masked * mask_front[rmin:rmax, cmin:cmax] + front[:, rmin:rmax, cmin:cmax] * ~(
            mask_front[rmin:rmax, cmin:cmax])

        if results['sample_name'][:8] == 'data_syn':
            img_masked = img_masked + np.random.normal(loc=0.0, scale=7.0, size=img_masked.shape)

        results['img'] = img_masked
        results['selected_idx'] = idx
        results['bbox'] = (rmin, rmax, cmin, cmax)
        results['mask'] = mask

        return results


@PIPELINES.register_module()
class CreatePoseGT(object):
    def __init__(self, mode='train', num_pt=1000, noise_trans=0.03, refine=False,
                 num_pt_mesh_small=500,
                 num_pt_mesh_large=2600):
        if mode == 'train':
            self.add_gt_cloud = True
        else:
            self.add_gt_cloud = False
        self.num_pt = num_pt
        self.noise_trans = noise_trans
        self.xmap = np.array([[j for i in range(640)] for j in range(480)])
        self.ymap = np.array([[i for i in range(640)] for j in range(480)])
        self.refine = refine
        self.num_pt_mesh_small = num_pt_mesh_small
        self.num_pt_mesh_large = num_pt_mesh_large

    def __call__(self, results):
        meta = results['meta_mat']
        obj = results['meta_mat']['cls_indexes'].flatten().astype(np.int32)
        idx = results['selected_idx']
        rmin, rmax, cmin, cmax = results['bbox']
        cam_cx, cam_cy, cam_fx, cam_fy = results['cam_params']

        target_r = meta['poses'][:, :, idx][:, 0:3]
        target_t = np.array([meta['poses'][:, :, idx][:, 3:4].flatten()])
        add_t = np.array([random.uniform(-self.noise_trans, self.noise_trans) for i in range(3)])

        choose = results['mask'][rmin:rmax, cmin:cmax].flatten().nonzero()[0]
        if len(choose) > self.num_pt:
            c_mask = np.zeros(len(choose), dtype=int)
            c_mask[:self.num_pt] = 1
            np.random.shuffle(c_mask)
            choose = choose[c_mask.nonzero()]
        else:
            choose = np.pad(choose, (0, self.num_pt - len(choose)), 'wrap')

        depth_masked = results['depth_img'][rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        xmap_masked = self.xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        ymap_masked = self.ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        choose = np.array([choose])

        cam_scale = meta['factor_depth'][0][0]
        pt2 = depth_masked / cam_scale
        pt0 = (ymap_masked - cam_cx) * pt2 / cam_fx
        pt1 = (xmap_masked - cam_cy) * pt2 / cam_fy
        cloud = np.concatenate((pt0, pt1, pt2), axis=1)
        if self.add_gt_cloud:
            cloud = np.add(cloud, add_t)

        dellist = [j for j in range(0, len(results['cld'][obj[idx]]))]
        if self.refine:
            dellist = random.sample(dellist, len(results['cld'][obj[idx]]) - self.num_pt_mesh_large)
        else:
            dellist = random.sample(dellist, len(results['cld'][obj[idx]]) - self.num_pt_mesh_small)
        model_points = np.delete(results['cld'][obj[idx]], dellist, axis=0)

        target = np.dot(model_points, target_r.T)
        if self.add_gt_cloud:
            target = np.add(target, target_t + add_t)
        else:
            target = np.add(target, target_t)

        results['cloud'] = cloud
        results['choose'] = choose
        results['model_points'] = model_points
        results['target'] = target
        results['index'] = [int(obj[idx]) - 1]

        return results


@PIPELINES.register_module()
class DefaultPoseFormatBundle(object):
    def __init__(self):
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __call__(self, results):
        results['cloud'] = DC(torch.from_numpy(results['cloud'].astype(np.float32)), stack=True, pad_dims=1)
        results['choose'] = DC(torch.LongTensor(results['choose'].astype(np.int32)), stack=True, pad_dims=1)
        results['img'] = DC(self.norm(torch.from_numpy(results['img'].astype(np.float32))), stack=True)
        results['model_points'] = DC(torch.from_numpy(results['model_points'].astype(np.float32)), stack=True, pad_dims=1)
        results['index'] = DC(torch.LongTensor(results['index']).reshape(-1, 1), stack=True, pad_dims=1)
        results['target'] = DC(torch.from_numpy(results['target'].astype(np.float32)), stack=True, pad_dims=1)

        return results

    def __repr__(self):
        return self.__class__.__name__

