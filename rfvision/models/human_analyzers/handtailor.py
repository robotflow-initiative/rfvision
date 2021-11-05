from rfvision.models.builder import build_detector
import numpy as np
from rfvision.models.human_analyzers.utils.mano_layers import ManoLayer
import torch
from rfvision.models.detectors3d import Base3DDetector
from rfvision.core.visualizer_pose import imshow_mesh_3d
import cv2


class HandTailor(Base3DDetector):
    def __init__(self,
                 hand_net,
                 ik_net,
                 triangles_file,
                 mano_cfg=None,):

        self.hand_net = build_detector(hand_net)
        self.ik_net = build_detector(ik_net)

        self.mano_layers = ManoLayer(**mano_cfg) if mano_cfg is not None else ManoLayer()
        self.triangles = np.loadtxt(mano_cfg['triangles_file'])

    def __call__(self, img, img_metas):

        self.hand_net.eval()
        self.ik_net.eval()

        with torch.no_grad():
            res = self.hand_net.forward_test(img)

        # res['preds'][:, 2] is the joints root-relative d,
        # therefore '+ res['rel_root_depth']' to restore the original joints d,
        # then '/1000' transfer d from unit mm to m
        joints_uvd = (res['preds'][:, 2] + res['rel_root_depth']) / 1000
        joints_xyz = self._pixel2cam(pixel_coord=joints_uvd, f=img_metas['focal'], c=img_metas['princpt'])
        with torch.no_grad():
            theta = self.ik_net.forward_test(joints_xyz)

        # to numpy
        theta = theta.cpu().numpy()
        beta = np.zeros(10)
        vertices = self.mano_layers(theta, beta)

        img_draw = imshow_mesh_3d(img=img,
                                  vertices=[vertices],
                                  faces=self.triangle,
                                  camera_center=img_metas['princpt'],
                                  focal_length=img_metas['focal'])

        return img_draw

    def _pixel2cam(self, pixel_coord, f, c):
        x = (pixel_coord[:, 0] - c[0]) / f[0] * pixel_coord[:, 2]
        y = (pixel_coord[:, 1] - c[1]) / f[1] * pixel_coord[:, 2]
        z = pixel_coord[:, 2]
        cam_coord = np.concatenate((x[:, None], y[:, None], z[:, None]), 1)
        return cam_coord


if __name__ == '__main__':
    ik_net = dict(type='IKNet')
    hand_net = dict(
        type='Interhand3D',
        backbone=dict(type='ResNet', depth=50, init_cfg='torchvision://resnet50', ),
        keypoint_head=dict(
            type='Topdown3DHeatmapSimpleHead',
            keypoint_head_cfg=dict(
                in_channels=2048,
                out_channels=21 * 64,
                depth_size=64,
                num_deconv_layers=3,
                num_deconv_filters=(256, 256, 256),
                num_deconv_kernels=(4, 4, 4),
            ),
            root_head_cfg=dict(
                in_channels=2048,
                heatmap_size=64,
                hidden_dims=(512,),
            ),
            hand_type_head_cfg=dict(
                in_channels=2048,
                num_labels=2,
                hidden_dims=(512,),
            ),
            loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True),
            loss_root_depth=dict(type='L1LossPose', use_target_weight=True),
            loss_hand_type=dict(type='BCELoss', use_target_weight=True)
        ),
        train_cfg=dict(),
        test_cfg=dict(
            flip_test=True,
            post_process='default',
            shift_heatmap=True,
            modulate_kernel=11))
    # use your own img
    img = cv2.imread('YOUR_IMAGE', flags=-1)

    # use your own princpt and focal of your camera
    img_metas = dict(princpt=np.array([300, 300]),
                     focal=np.array([1250, 1250]))
    model = HandTailor(ik_net=ik_net, hand_net=hand_net, triangles_file='hand')
    img_draw = model(img, img_metas)