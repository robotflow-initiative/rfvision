import os
import numpy as np
import cv2
import torch
os.environ.update(PYOPENGL_PLATFORM='egl',)
import trimesh
import pyrender
import open3d as o3d
from tqdm import tqdm
import pickle
from rfvision.datasets import DATASETS
from rfvision.datasets.pipelines import Compose

def pc_downsample(pc: np.ndarray, voxel_size=0.05):
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pc))
    return np.array(pcd.voxel_down_sample(voxel_size).points)

def estimate_normals(pc, knn):
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pc))
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn))
    return np.array(pcd.normals)

def backproject(depth, intrinsics, instance_mask):
    intrinsics_inv = np.linalg.inv(intrinsics)
    image_shape = depth.shape
    width = image_shape[1]
    height = image_shape[0]

    x = np.arange(width)
    y = np.arange(height)

    # non_zero_mask = np.logical_and(depth > 0, depth < 5000)
    non_zero_mask = (depth > 0)
    final_instance_mask = np.logical_and(instance_mask, non_zero_mask)

    idxs = np.where(final_instance_mask)
    grid = np.array([idxs[1], idxs[0]])

    length = grid.shape[1]
    ones = np.ones([1, length])
    uv_grid = np.concatenate((grid, ones), axis=0)  # [3, num_pixel]

    xyz = intrinsics_inv @ uv_grid  # [3, num_pixel]
    xyz = np.transpose(xyz)  # [num_pixel, 3]

    z = depth[idxs[0], idxs[1]]

    pts = xyz * z[:, np.newaxis] / xyz[:, -1:]
    pts[:, 0] = -pts[:, 0]
    pts[:, 1] = -pts[:, 1]
    return pts, idxs


@DATASETS.register_module()
class ShapeNetDatasetForPPF(torch.utils.data.Dataset):
    def __init__(self,
                 data_root,
                 ann_file,
                 category=2,
                 test_mode=False):
        super().__init__()

        self.res = 5e-3
        self.tr_num_bins = 32
        self.rot_num_bins = 36
        self.cls_bins = True
        self.up_sym = False
        self.right_sym = False
        self.z_right = False
        self.npoint_max = 10000
        self.K = np.float32([[591.0125, 0, 320],
                             [0, 590.16775, 240],
                             [0, 0, 1]])

        self.knn=60
        self.category = category

        random_seed = 0

        if random_seed is not None:
            np.random.seed(random_seed)

        with open(ann_file, 'r') as f:
            lines = f.readlines()
        self.outputs = []
        self.mesh_path = []
        for line in lines:
            line = line[2:-1]
            category = line.split('/')[0]
            if category == category_list[self.category]:
                mesh_path = os.path.join(data_root, line, 'models', 'model_normalized.obj')
                self.mesh_path.append([mesh_path, category])

        print('Processing...')
        self.outputs = tuple(self.get_pc_and_normals(i) for i in tqdm(range((len(self)))))


    def get_pc_and_normals(self, idx):
        r = pyrender.OffscreenRenderer(viewport_width=640, viewport_height=480)
        mesh = trimesh.load(self.mesh_path[idx][0])
        obj_scale = shapenet_obj_scales[self.mesh_path[idx][1]]

        mesh_pose = np.eye(4)
        y_angle = np.random.uniform(0, 2 * np.pi)
        x_angle = np.random.uniform(25 / 180 * np.pi, 65 / 180 * np.pi)
        yy_angle = np.random.uniform(-15 / 180 * np.pi, 15 / 180 * np.pi)
        # rotate to nocs coord
        flip2nocs = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
        mesh_pose[:3, :3] = roty(yy_angle)[:3, :3] @ rotx(x_angle)[:3, :3] @ roty(y_angle)[:3, :3]
        tr = np.array([np.random.uniform(-0.3, 0.3), np.random.uniform(-0.3, 0.3), -np.random.uniform(0.6, 2.0)])
        mesh_pose[:3, -1] = tr

        bounds = mesh.bounds
        trans_mat = np.eye(4)
        trans_mat[:3, -1] = -(bounds[1] + bounds[0]) / 2

        scale_mat = np.eye(4)
        scale = np.random.uniform(obj_scale[0], obj_scale[1])
        scale_mat[:3, :3] *= scale
        mesh.apply_transform(mesh_pose @ scale_mat @ trans_mat)
        if isinstance(mesh, trimesh.Scene):
            scene = pyrender.Scene.from_trimesh_scene(mesh)
            scene.bg_color = np.random.rand(3)
        else:
            scene = pyrender.Scene(ambient_light=np.array([0.02, 0.02, 0.02, 1.0]), bg_color=np.random.rand(3))
            scene.add(pyrender.Mesh.from_trimesh(mesh), pose=np.eye(4))

        direc_l = pyrender.DirectionalLight(color=np.ones(3), intensity=np.random.uniform(5, 15))
        spot_l = pyrender.SpotLight(color=np.ones(3), intensity=np.random.uniform(0, 10),
                           innerConeAngle=np.pi / 16, outerConeAngle=np.pi / 6)

        cam_pose = np.eye(4)
        cam = pyrender.camera.IntrinsicsCamera(591.0125, 590.16775, 320, 240)

        scene.add(cam, pose=cam_pose)
        scene.add(direc_l, pose=cam_pose)
        scene.add(spot_l, pose=cam_pose)

        rgb, depth = r.render(scene)
        mask = (depth > 0).astype(bool)
        pc = backproject(depth.astype('float32'), self.K.astype('float32'), mask)[0]
        pc[:, 0] = -pc[:, 0]
        pc[:, 2] = -pc[:, 2]
        pc -= tr
        pc = (flip2nocs @ np.linalg.inv(mesh_pose[:3, :3]) @ pc.T).T

        # random jitter, all point together
        pc = pc + np.clip(self.res / 4 * np.random.randn(*pc.shape), -self.res / 2, self.res / 2)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc)
        pcd = pcd.voxel_down_sample(self.res)
        pc = np.array(pcd.points)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=self.knn))
        normals = np.array(pcd.normals).astype(np.float32)
        if pc.shape[0] < 100 or pc.shape[0] > self.npoint_max:
            return self.get_pc_and_normals(idx=np.random.randint(len(self)))
        return pc, normals, bounds, scale

    def get_item_impl(self, pc, normals, bounds, scale):
        # dummy normal
        targets_tr, targets_rot, targets_rot_aux, point_idxs = generate_target(pc, normals, self.up_sym, self.right_sym,
                                                                               self.z_right, 200000)

        if self.cls_bins:
            tr_range = tr_ranges[self.category]
            targets_tr = np.stack([
                real2prob(np.clip(targets_tr[:, 0] + tr_range[0], 0, 2 * tr_range[0]), 2 * tr_range[0], self.tr_num_bins,
                          circular=False),
                real2prob(np.clip(targets_tr[:, 1], 0, tr_range[1]), tr_range[1], self.tr_num_bins, circular=False),
            ], 1)

        if self.cls_bins:
            targets_rot = np.stack([
                real2prob(targets_rot[:, 0], np.pi, self.rot_num_bins, circular=False),
                real2prob(targets_rot[:, 1], np.pi, self.rot_num_bins, circular=False),
            ], 1)

        targets_scale = np.log(((bounds[1] - bounds[0]) / 2).astype(np.float32) * scale) - np.log(
            np.array(scale_ranges[self.category]))

        return {'pcs':pc.astype(np.float32), 'pc_normals':normals, 'targets_tr':targets_tr, 'targets_rot':targets_rot,
                'targets_rot_aux': targets_rot_aux, 'targets_scale': targets_scale.astype(np.float32),
                'point_idxs': point_idxs}

    def __getitem__(self, idx):
        pc, normals, bounds, scale = self.outputs[idx]
        return self.get_item_impl(pc, normals, bounds, scale)

    def __len__(self):
        return len(self.mesh_path)


@DATASETS.register_module()
class NOCSForPPF(torch.utils.data.Dataset):
    def __init__(self,
                 data_root,
                 scene_id=1,
                 category=2,
                 pipeline=None,
                 test_mode=False,
                 ):
        self.data_root = data_root
        self.category = category
        self.res = 5e-3
        self.knn = 60

        intrinsics = np.array([[591.0125, 0, 322.525], [0, 590.16775, 244.11084], [0, 0, 1]])

        scene_dir = os.path.join(self.data_root, 'real_test', 'scene_' + str(scene_id))
        log_dir = os.path.join(self.data_root, 'real_test_20210511T2129')
        gt_pc_dir = os.path.join(self.data_root, 'obj_models', 'real_test')

        id_list = []
        for i in os.listdir(scene_dir):
            if i[:4] not in id_list:
                id_list.append(i[:4])

        self.outputs = []
        print('Processing...')



        # for i in tqdm(id_list):
        for i in ['0000']:
            path = os.path.join(scene_dir, i)
            _, depth, masks, _, class_ids, _, infos, _, gt_RTs = self.read(path)
            pkl = pickle.load(open(os.path.join(self.data_root, log_dir, 'results_real_test_{}.pkl'.format('_'.join(path.split('/')[-2:]))), 'rb'))
            pred_cls_ids = pkl['pred_class_ids']
            pred_masks = pkl['pred_masks']
            for (info, cls_id, mask, RT) in zip(infos, class_ids, masks, gt_RTs):
                if cls_id != self.category:
                    continue

                scale = np.power(np.linalg.det(RT[:3, :3]), 1. / 3)
                RT[:3, :3] /= scale
                gt_pc = np.loadtxt(os.path.join(gt_pc_dir, f'{info[2]}_vertices.txt'))
                gt_pc = pc_downsample(gt_pc, self.res)
                # rough mask with background
                if cls_id not in list(pred_cls_ids):
                    break
                mask = pred_masks[:, :, list(pred_cls_ids).index(cls_id)].astype('int0')
                pc = backproject(depth, intrinsics, mask)[0]
                # augment
                pc = pc + np.clip(self.res / 4 * np.random.randn(*pc.shape), -self.res / 2, self.res / 2)
                pc /= 1000
                pc[:, 0] = -pc[:, 0]
                pc[:, 1] = -pc[:, 1]

                pc = pc_downsample(pc, voxel_size=self.res)
                pc_normal = estimate_normals(pc, self.knn).astype(np.float32)

                point_idxs = np.random.randint(0, pc.shape[0], (1000000, 2))

                self.outputs.append([pc.astype('float32'),
                                     pc_normal.astype('float32'),
                                     point_idxs.astype('int0'),
                                     gt_pc.astype('float32'), RT])
        if pipeline is not None:
            self.pipeline = Compose(pipeline)

    def __len__(self):
        return len(self.outputs)

    def __getitem__(self, idx):
        out = {'pcs':self.outputs[idx][0],
               'pc_normals':self.outputs[idx][1],
               'point_idxs':self.outputs[idx][2],
               'gt_pc': self.outputs[idx][3],
               'RT': self.outputs[idx][4]}
        if self.pipeline:
            return self.pipeline(out)
        else:
            return out


    def read(self, base_path):
        path = base_path
        mask_path = path + '_mask.png'
        coord_path = path + '_coord.png'
        meta_path = path + "_meta.txt"

        image = cv2.imread(path + '_color.png')[..., ::-1]
        depth = cv2.imread(path + '_depth.png', -1)

        assert os.path.exists(mask_path), "{} is missing".format(mask_path)
        assert os.path.exists(coord_path), "{} is missing".format(coord_path)

        inst_dict = {}
        for line in open(meta_path).read().splitlines():
            line_info = line.split(' ')
            inst_id = int(line_info[0])  ##one-indexed
            cls_id = int(line_info[1])  ##zero-indexed
            # symmetry_id = int(line_info[2])
            inst_dict[inst_id] = cls_id

        meta_path = path + '_meta.txt'

        mask_im = cv2.imread(mask_path)[:, :, 2]
        coord_map = cv2.imread(coord_path)[:, :, :3]
        coord_map = coord_map[:, :, (2, 1, 0)]

        masks, coords, class_ids, scales, infos = self.process_data(mask_im, coord_map, inst_dict, meta_path)
        scales_bbox = scales / np.linalg.norm(scales, axis=-1, keepdims=True)
        masks = np.transpose(masks, [2, 0, 1])

        image_path_parsing = path.split('/')
        gt_pkl_path = os.path.join(self.data_root, 'gts/real_test', f'results_real_test_{image_path_parsing[-2]}_{image_path_parsing[-1]}.pkl')
        if (os.path.exists(gt_pkl_path)):
            with open(gt_pkl_path, 'rb') as f:
                gt = pickle.load(f)
            gt_RTs = gt['gt_RTs']
        return image, depth, masks, coords, class_ids, scales, infos, scales_bbox, gt_RTs

    def process_data(self, mask_im, coord_map, inst_dict, meta_path):
        # parsing mask
        cdata = mask_im
        cdata = np.array(cdata, dtype=np.int32)

        # instance ids
        instance_ids = list(np.unique(cdata))
        instance_ids = sorted(instance_ids)
        # remove background
        assert instance_ids[-1] == 255
        del instance_ids[-1]

        cdata[cdata == 255] = -1
        assert (np.unique(cdata).shape[0] < 20)

        num_instance = len(instance_ids)
        h, w = cdata.shape

        # flip z axis of coord map
        coord_map = np.array(coord_map, dtype=np.float32) / 255
        coord_map[:, :, 2] = 1 - coord_map[:, :, 2]

        masks = np.zeros([h, w, num_instance], dtype=np.uint8)
        coords = np.zeros((h, w, num_instance, 3), dtype=np.float32)
        class_ids = np.zeros([num_instance], dtype=np.int_)
        scales = np.zeros([num_instance, 3], dtype=np.float32)

        with open(meta_path, 'r') as f:
            lines = f.readlines()

        scale_factor = np.zeros((len(lines), 3), dtype=np.float32)
        all_words = []
        for i, line in enumerate(lines):
            words = line[:-1].split(' ')
            bbox_file = os.path.join(self.data_root, 'obj_models', 'real_test', words[2] + '.txt')
            scale_factor[i, :] = np.loadtxt(bbox_file)
            scale_factor[i, :] /= np.linalg.norm(scale_factor[i, :])
            all_words.append(words)

        i = 0

        # delete ids of background objects and non-existing objects
        inst_id_to_be_deleted = []
        for inst_id in inst_dict.keys():
            if inst_dict[inst_id] == 0 or (not inst_id in instance_ids):
                inst_id_to_be_deleted.append(inst_id)
        for delete_id in inst_id_to_be_deleted:
            del inst_dict[delete_id]

        all_words_clean = all_words.copy()
        idx2inst = {}
        for inst_id in instance_ids:  # instance mask is one-indexed
            if not inst_id in inst_dict:
                continue
            inst_mask = np.equal(cdata, inst_id)
            assert np.sum(inst_mask) > 0
            assert inst_dict[inst_id]

            masks[:, :, i] = inst_mask
            coords[:, :, i, :] = np.multiply(coord_map, np.expand_dims(inst_mask, axis=-1))

            # class ids is also one-indexed
            class_ids[i] = inst_dict[inst_id]
            scales[i, :] = scale_factor[inst_id - 1, :]
            all_words_clean[i] = all_words[inst_id - 1]
            idx2inst[i] = inst_id
            i += 1
        masks = masks[:, :, :i]
        coords = coords[:, :, :i, :]
        coords = np.clip(coords, 0, 1)

        class_ids = class_ids[:i]
        scales = scales[:i]
        all_words_clean = all_words_clean[:i]

        return masks, coords, class_ids, scales, all_words_clean

    def aug_test(self):
        pass
    def extract_feat(self):
        pass
    def simple_test(self):
        pass

# bottle
# -0.25~0.25, 0~0.25, 0.05, 0.15, 0.05

# bowl
# -0.12~0.12, 0~0.12, 0.07, 0.03, 0.07

# camera
# -0.15~0.15, 0~0.15, 0.05, 0.05, 0.07

# can
# -0.1~0.1, 0~0.1, 0.037, 0.055, 0.037

# laptop:
# -0.3 ~ 0.3, 0~0.3, 0.13, 0.1, 0.15

# mug
# -0.12~0.12, 0~0.12, 0.06, 0.05, 0.045

tr_ranges = {
    1: [0.25, 0.25],
    2: [0.12, 0.12],
    3: [0.15, 0.15],
    4: [0.1, 0.1],
    5: [0.3, 0.3],
    6: [0.12, 0.12]
}

scale_ranges = {
    1: [0.05, 0.15, 0.05],
    2: [0.07, 0.03, 0.07],
    3: [0.05, 0.05, 0.07],
    4: [0.037, 0.055, 0.037],
    5: [0.13, 0.1, 0.15],
    6: [0.06, 0.05, 0.045]
}


def real2prob(val, max_val, num_bins, circular=False):
    is_torch = isinstance(val, torch.Tensor)
    if is_torch:
        res = torch.zeros((*val.shape, num_bins), dtype=val.dtype).to(val.device)
    else:
        res = np.zeros((*val.shape, num_bins), dtype=val.dtype)

    if not circular:
        interval = max_val / (num_bins - 1)
        if is_torch:
            low = torch.clamp(torch.floor(val / interval).long(), max=num_bins - 2)
        else:
            low = np.clip(np.floor(val / interval).astype(np.int64), a_min=None, a_max=num_bins - 2)
        high = low + 1

        # huge memory
        if is_torch:
            res.scatter_(-1, low[..., None], torch.unsqueeze(1. - (val / interval - low), -1))
            res.scatter_(-1, high[..., None], 1. - torch.gather(res, -1, low[..., None]))
        else:
            np.put_along_axis(res, low[..., None], np.expand_dims(1. - (val / interval - low), -1), -1)
            np.put_along_axis(res, high[..., None], 1. - np.take_along_axis(res, low[..., None], -1), -1)
        return res
    else:
        interval = max_val / num_bins
        if is_torch:
            val_new = torch.clone(val)
        else:
            val_new = val.copy()
        val_new[val < interval / 2] += max_val
        res = real2prob(val_new - interval / 2, max_val, num_bins + 1)
        res[..., 0] += res[..., -1]
        return res[..., :-1]


def generate_target(pc, pc_normal, up_sym=False, right_sym=False, z_right=False, subsample=200000):
    if subsample is None:
        xv, yv = np.meshgrid(np.arange(pc.shape[1]), np.arange(pc.shape[1]))
        point_idxs = np.stack([yv, xv], -1).reshape(-1, 2)
    else:
        point_idxs = np.random.randint(0, pc.shape[0], size=[subsample, 2])

    a = pc[point_idxs[:, 0]]
    b = pc[point_idxs[:, 1]]
    pdist = a - b
    pdist_unit = pdist / (np.linalg.norm(pdist, axis=-1, keepdims=True) + 1e-7)
    proj_len = np.sum(a * pdist_unit, -1)
    oc = a - proj_len[..., None] * pdist_unit
    dist2o = np.linalg.norm(oc, axis=-1)
    target_tr = np.stack([proj_len, dist2o], -1)

    up = np.array([0, 1, 0])
    down = np.array([0, -1, 0])
    if z_right:
        right = np.array([0, 0, 1])
        left = np.array([0, 0, -1])
    else:
        right = np.array([1, 0, 0])
        left = np.array([-1, 0, 0])
    up_cos = np.arccos(np.sum(pdist_unit * up, -1))
    if up_sym:
        up_cos = np.minimum(up_cos, np.arccos(np.sum(pdist_unit * down, -1)))
    right_cos = np.arccos(np.sum(pdist_unit * right, -1))
    if right_sym:
        right_cos = np.minimum(right_cos, np.arccos(np.sum(pdist_unit * left, -1)))
    target_rot = np.stack([up_cos, right_cos], -1)

    pairwise_normals = pc_normal[point_idxs[:, 0]]
    pairwise_normals[np.sum(pairwise_normals * pdist_unit, -1) < 0] *= -1
    target_rot_aux = np.stack([
        np.sum(pairwise_normals * up, -1) > 0,
        np.sum(pairwise_normals * right, -1) > 0
    ], -1).astype(np.float32)
    return target_tr.astype(np.float32).reshape(-1, 2), target_rot.astype(np.float32).reshape(-1,
                                                                                              2), target_rot_aux.reshape(
        -1, 2), point_idxs.astype(np.int64)


def rotz(a):
    return np.array([[np.cos(a), np.sin(a), 0, 0],
                     [-np.sin(a), np.cos(a), 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])


def roty(a):
    return np.array([[np.cos(a), 0, -np.sin(a), 0],
                     [0, 1, 0, 0],
                     [np.sin(a), 0, np.cos(a), 0],
                     [0, 0, 0, 1]])


def rotx(a):
    return np.array([[1, 0, 0, 0],
                     [0, np.cos(a), -np.sin(a), 0],
                     [0, np.sin(a), np.cos(a), 0],
                     [0, 0, 0, 1]])


shapenet_obj_scales = {
    '02946921': [0.128, 0.18],
    '02876657': [0.2300, 0.4594],
    '02880940': [0.1851, 0.2381],
    '02942699': [0.1430, 0.2567],
    '03642806': [0.3862, 0.5353],
    '03797390': [0.1501, 0.1995]
}

category_list = ['02946921', '02876657', '02880940', '02942699', '03642806', '03797390']


if __name__ == '__main__':
    # dataset = ShapeNetDatasetForPPF(data_root='/hdd0/data/shapenet_v2/ShapeNetCore.v2',
    #                     ann_file='/hdd0/data/ppf_dataset/shapenet_val.txt')
    # data = dataset[0]

    dataset = NOCSForPPF(data_root='/hdd0/data/ppf_dataset/', scene_id=1)
