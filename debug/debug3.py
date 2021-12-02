from rfvision.models.builder import build_detector
from rfvision.datasets.builder import build_dataset, build_dataloader
import rflib
from rflib.runner import load_checkpoint
from rflib.parallel import RFDataParallel
from copy import deepcopy
import torch
from torchvision import transforms, datasets as ds
from rfvision.tools import count_paras, show_tensor_img, format_single_data
import matplotlib.pyplot as plt
import pickle
from rfvision.models.detectors.solov2 import SOLOv2
from rfvision.models.detectors3d.category_ppf.category_ppf_dataset import NOCSForPPF
import cv2
import numpy as np
# data_root = '/hdd0/data/ppf_dataset/'
# dataset = NOCSForPPF(data_root)

hand_net_cfg ='/home/hanyang/rfvision/flows/human_analyzers/hand/interhand3d/res50_interhand3d_all_256x256.py'
cfg=rflib.Config.fromfile(hand_net_cfg)
hand_net = build_detector(cfg.model)
rflib.runner.load_checkpoint(hand_net, '/home/hanyang/weights/res50_intehand3dv1.0_all_256x256-42b7f2ac_20210702.pth')
# hand_net.init_weights()
hand_net.eval()


def draw_kpts(img: np.ndarray, kpts: np.ndarray):
    img_draw = img.copy()
    for i in kpts:
        cv2.drawMarker(img_draw, np.int0(i[:2]), (0, 255, 0), markerSize=5)
    return img_draw


from torchvision.transforms import functional as F

for i in range(10):
    img_ori = cv2.imread(f'/home/hanyang/handtailor/demo/0{i}.jpg', flags=-1)
    img = F.to_tensor(img_ori)
    img = F.normalize(img, mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225])
    img = img.unsqueeze(0)

    img_metas = [{'bbox_id': [],
                  'heatmap3d_depth_bound':400,
                  'root_depth_bound':400,
                  'center': (256 / 2, 256 / 2),
                  'scale': 1,
                  'image_file': '/home/hanyang/handtailor/demo/09.jpg',
                  'flip_pairs':[]}]

    with torch.no_grad():
        res = hand_net.forward_test(img, img_metas)

    joints_uv_pred = res['preds'][0][:, :2]
    img_draw = draw_kpts(img_ori, joints_uv_pred)
    import matplotlib.pyplot as plt

    plt.imshow(img_draw)
    plt.show()