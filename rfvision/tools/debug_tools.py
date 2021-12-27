import numpy as np
import matplotlib.pyplot as plt
import torch

from rfvision.datasets.builder import build_dataset
import rflib
from rfvision.models.builder import build_detector
import cv2

def count_paras(model: torch.nn.Module):
    total_para = 0
    for i in model.state_dict().values():
        total_para += i.numel()
    return total_para


def show_tensor_img(img_tensor: torch.Tensor):
    assert img_tensor.shape[0] == 3
    img = img_tensor.cpu().numpy().transpose(1, 2, 0)[:, :, ::-1]
    img = img[:,:, ::-1] #bgr to rgb
    plt.imshow(img)
    plt.show()


def format_single_data(data: dict, devices='cpu'):
    data = {k: v.data.unsqueeze(0).to(devices) if isinstance(v.data, torch.Tensor) else [v.data] for k, v in data.items()}
    return data

def init_constant_for_all(model: torch.nn.Module, val=0.01):
    for v in model.state_dict().values():
        torch.nn.init.constant_(v, val)

def debug_model(cfg_path, checkpoints_path=None):
    cfg = rflib.Config.fromfile(cfg_path)
    model = build_detector(cfg.model,
                           train_cfg=cfg.model.get('train_cfg', None),
                           test_cfg=cfg.model.get('test_cfg', None))
    if checkpoints_path is not None:
        rflib.runner.load_checkpoint(model, checkpoints_path)
    else:
        init_constant_for_all(model, 0.01)
    return model

def debug_children(model: torch.nn.Module, input: torch.Tensor):
    model.eval()
    res = []
    with torch.no_grad():
        for i in model.children():
            if len(res) is 0:
                res.append(i(input))
            else:
                res.append(i(res[-1]))
        return res

def debug_dataset(cfg_path, set='train'):
    cfg = rflib.Config.fromfile(cfg_path)
    if set == 'train':
        dataset = build_dataset(cfg.data.train)
    elif set == 'test':
        dataset = build_dataset(cfg.data.test)
    elif set == 'val':
        dataset = build_dataset(cfg.data.val)
    return dataset


def draw_bbox_xyxy(img, bboxes):
    bboxes = np.int0(bboxes)
    img_draw = img.copy()
    for i in bboxes:
        cv2.rectangle(img_draw, i[:2], i[2:], (0, 255, 0))
    plt.imshow(img_draw)
    plt.show()
    return img_draw