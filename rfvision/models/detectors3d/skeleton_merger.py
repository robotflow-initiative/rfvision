from rfvision.models.builder import DETECTORS, build_backbone, build_head
from rfvision.models import BaseDetector
from rflib.runner import load_checkpoint
import torch
import open3d as o3d
import numpy as np
import os



@DETECTORS.register_module()
class SkeletonMerger(BaseDetector):
    def __init__(self,
                 backbone, 
                 head,
                 init_cfg=None,
                 **kwargs):
        super().__init__()
        self.backbone = build_backbone(backbone)
        self.head = build_head(head)
        self.init_weights(init_cfg)

    def forward_train(self, points):
        APP_PT = torch.cat([points, points, points], -1)
        KP, GF = self.backbone(APP_PT)
        pred_dict = self.head(points, KP, GF)
        """
        'reconstructed_parts':RP,
        'keypoint_cloud':KPCD,
        'keypoint_activation':activation,
        'learned_offset':LF,
        'mask':MA
        """
        losses = self.head.loss(points, pred_dict)
        return losses
    
    def forward_test(self, points):
        APP_PT = torch.cat([points, points, points], -1)
        KP, GF = self.backbone(APP_PT)
        pred_dict = self.head(points, KP, GF)
        return pred_dict,
        
        
    def forward(self, return_loss=True, **kwargs):
        points = kwargs['points']
        if return_loss == True:
            losses = self.forward_train(points)
            return losses
        else:
            pred_dict = self.forward_test(points)
            return pred_dict

    def show_results(self, data, results, out_dir=None):
        points = np.array(data['points']).reshape(-1, 3)
        colors = np.array(data['colors']).reshape(-1, 3)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        metas = data['img_metas'].data[0][0]

        # faces = np.array(data['faces']).reshape(-1, 3)
        # vertices = np.array(data['vertices']).reshape(-1, 3)
        # mesh = o3d.geometry.TriangleMesh()
        # mesh.vertices = o3d.utility.Vector3dVector(vertices)
        # mesh.faces = o3d.utility.Vector3dVector(faces)
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(pcd)
        for kp_xyz in results[0]['keypoints_xyz'][0]:
            mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
            mesh_sphere.translate(np.array(kp_xyz.cpu()))
            mesh_sphere.paint_uniform_color([1,0,0]) # red
            vis.add_geometry(mesh_sphere)
            # mesh += mesh_sphere
        vis.run()

        if out_dir is not None:
            img_name = os.path.join(out_dir, metas['model_id'] + '.png')
            vis.capture_screen_image(img_name, do_render=True)

    def init_weights(self, init_cfg=None):
        if isinstance(init_cfg, str):
            from rfvision.utils import get_root_logger
            logger = get_root_logger()
            load_checkpoint(self, init_cfg, strict=False, logger=logger)

    def simple_test(self, img, img_metas, **kwargs):
        pass
    def aug_test(self, imgs, img_metas, **kwargs):
        pass
    def extract_feat(self, imgs):
        pass