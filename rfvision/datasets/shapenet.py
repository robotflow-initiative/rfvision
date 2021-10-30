import torch
import os
import numpy as np
from rfvision.datasets import CustomDataset, DATASETS


class ShapeNetDataset(CustomDataset):
    CLASSES = ['02954340', '03001627', '02958343', '04099429', '04090263', '04460130',
               '02933112', '04530566', '03467517', '04330267', '02942699', '03046257',
               '03636649', '02992529', '03593526', '02747177', '02876657', '03761084',
               '03624134', '03790512', '02801938', '03710193', '03938244', '04256520',
               '03211117', '03325088', '02871439', '03928116', '02946921', '03207941',
               '02808440', '03759954', '02924116', '03991062', '03337140', '02843684',
               '04401088', '03642806', '03513137', '04468005', '02773838', '04004475',
               '04379243', '02880940', '04554684', '02828884', '04074963', '03691459',
               '03797390', '02691156', '02818832', '03085013', '03261776', '03948459',
               '04225987']

    def __init__(self,
                 data_root,
                 ann_file,
                 ):
        with open(ann_file, 'r') as f:
            annos = f.readlines()


        pass
    def __getitem__(self, item):
        pass

    def __len__(self):
        pass



class ShapeNetDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, model_names):
        super().__init__()
        os.environ.update(
            OMP_NUM_THREADS='1',
            OPENBLAS_NUM_THREADS='1',
            NUMEXPR_NUM_THREADS='1',
            MKL_NUM_THREADS='1',
            PYOPENGL_PLATFORM='osmesa',
            PYOPENGL_FULL_LOGGING='0'
        )
        self.cfg = cfg
        self.intrinsics = np.array([[591.0125, 0, 320], [0, 590.16775, 240], [0, 0, 1]])

        self.model_names = []
        for name in model_names:
            self.model_names.append(name)

    def get_item_impl(self, model_name, cfg, intrinsics):
        import OpenGL
        OpenGL.FULL_LOGGING = False
        OpenGL.ERROR_LOGGING = False
        from pyrender import PerspectiveCamera, \
            DirectionalLight, SpotLight, PointLight, \
            MetallicRoughnessMaterial, \
            Primitive, Mesh, Node, Scene, \
            OffscreenRenderer, PinholeCamera, RenderFlags
        r = OffscreenRenderer(viewport_width=640, viewport_height=480)
        shapenet_cls, mesh_name = model_name.split('/')
        path = f'/home/neil/disk/ShapeNetCore.v2/{shapenet_cls}/{mesh_name}/models/model_normalized.obj'
        mesh = trimesh.load(path)
        obj_scale = shapenet_obj_scales[f'{shapenet_cls}']

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
            scene = Scene.from_trimesh_scene(mesh)
            scene.bg_color = np.random.rand(3)
        else:
            scene = Scene(ambient_light=np.array([0.02, 0.02, 0.02, 1.0]), bg_color=np.random.rand(3))
            scene.add(Mesh.from_trimesh(mesh), pose=np.eye(4))

        direc_l = DirectionalLight(color=np.ones(3), intensity=np.random.uniform(5, 15))
        spot_l = SpotLight(color=np.ones(3), intensity=np.random.uniform(0, 10),
                           innerConeAngle=np.pi / 16, outerConeAngle=np.pi / 6)

        cam_pose = np.eye(4)
        cam = PinholeCamera(591.0125, 590.16775, 640, 480)

        scene.add(cam, pose=cam_pose)
        scene.add(direc_l, pose=cam_pose)
        scene.add(spot_l, pose=cam_pose)

        rgb, depth = r.render(scene)
        # rgb = rgb2gray(rgb)
        # rgb = np.stack([rgb] * 3, -1)
        # vis.image(np.moveaxis(rgb[::-1, ::-1], [0, 1, 2], [1, 2, 0]), win=2, env='debug')
        # vis.image(np.moveaxis(rgb, [0, 1, 2], [1, 2, 0]), win=3, env='debug')
        r.delete()

        mask = (depth > 0).astype(bool)
        idxs = np.where(mask)

        # background ikea
        table_names = os.listdir(hydra.utils.to_absolute_path('ikea_data'))
        table_names = [table_name for table_name in table_names if
                       os.path.isdir(hydra.utils.to_absolute_path(f'ikea_data/{table_name}'))]
        table_name = np.random.choice(table_names)
        color_fns = glob(hydra.utils.to_absolute_path(f'ikea_data/{table_name}/*_color.png'))
        color_fn = np.random.choice(color_fns)
        bg = cv2.imread(color_fn)[:, :, ::-1]
        rgb = rgb * mask[..., None] + bg * (1. - mask[..., None])

        pc, _ = backproject(depth, intrinsics, mask)
        pc[:, 0] = -pc[:, 0]
        pc[:, 2] = -pc[:, 2]
        pc -= tr
        pc = (flip2nocs @ np.linalg.inv(mesh_pose[:3, :3]) @ pc.T).T

        # random jitter, all point together
        pc = pc + np.clip(cfg.res / 4 * np.random.randn(*pc.shape), -cfg.res / 2, cfg.res / 2)

        discrete_coords, indices = ME.utils.sparse_quantize(np.ascontiguousarray(pc), return_index=True,
                                                            quantization_size=cfg.res)
        pc = pc[indices]
        # pc = (discrete_coords * self.cfg.res).numpy()
        if pc.shape[0] < 100 or pc.shape[0] > cfg.npoint_max:
            return self.get_item_impl(self.model_names[np.random.randint(len(self))], self.cfg, self.intrinsics)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=cfg.knn))
        normals = np.array(pcd.normals).astype(np.float32)

        # dummy normal
        targets_tr, targets_rot, targets_rot_aux, point_idxs = generate_target(pc, normals, cfg.up_sym, cfg.right_sym,
                                                                               cfg.z_right, 200000)

        if self.cfg.cls_bins:
            tr_range = tr_ranges[cfg.category]
            targets_tr = np.stack([
                real2prob(np.clip(targets_tr[:, 0] + tr_range[0], 0, 2 * tr_range[0]), 2 * tr_range[0], cfg.tr_num_bins,
                          circular=False),
                real2prob(np.clip(targets_tr[:, 1], 0, tr_range[1]), tr_range[1], cfg.tr_num_bins, circular=False),
            ], 1)

        # pc_colors = (rgb / 255).astype(np.float32)
        # pc_colors *= (1 + 0.4 * np.random.random(3) - 0.2) # brightness change for each channel
        # pc_colors += np.expand_dims((0.05 * np.random.random((pc_colors.shape[0], pc_colors.shape[1])) - 0.025), -1) # jittering on each pixel
        # pc_colors = np.clip(pc_colors, 0, 1)

        # pc_colors = rgb2gray(pc_colors).astype(np.float32)
        # pc_colors = np.stack([pc_colors] * 3, -1)

        if self.cfg.cls_bins:
            targets_rot = np.stack([
                real2prob(targets_rot[:, 0], np.pi, cfg.rot_num_bins, circular=False),
                real2prob(targets_rot[:, 1], np.pi, cfg.rot_num_bins, circular=False),
            ], 1)

        targets_scale = np.log(((bounds[1] - bounds[0]) / 2).astype(np.float32) * scale) - np.log(
            np.array(scale_ranges[cfg.category]))
        # print(targets_scale)
        # return pc_colors, (depth / 1000).astype(np.float32), np.stack(idxs, -1).astype(np.int64), \
        return pc.astype(np.float32), normals, targets_tr, targets_rot, targets_rot_aux, targets_scale.astype(
            np.float32), point_idxs

    def __getitem__(self, idx):
        model_name = self.model_names[idx]
        return self.get_item_impl(model_name, self.cfg, self.intrinsics)

    def __len__(self):
        return len(self.model_names)



if __name__ == '__main__':
    pass