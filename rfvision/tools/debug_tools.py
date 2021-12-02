import numpy as np
import matplotlib.pyplot as plt
import torch


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

def init_constant_for_all(model: torch.nn.Module, val=1):
    {k: torch.nn.init.constant_(v, val) for k, v in model.state_dict().items()}