import torch
# from rfvision.models.human_analyzers.utils.mano_layers import ManoLayer
# from rfvision.models import build_detector
# from rflib.runner import load_checkpoint
# import rflib
# import pickle
# import matplotlib.pyplot as plt
# import numpy as np
# import cv2
# import trimesh
# import open3d as o3d
# from rfvision.datasets import build_dataset
# from rfvision.models.human_analyzers.handtailor.utils import MeshRenderer

a = torch.load('/home/hanyang/handtailor/checkpoints/model.pt')
ik_state_dict = {}
for k, v in a['model'].items():
    if k.startswith('ik'):
        ik_state_dict.update({k[11:]: v})
a['state_dict'] = ik_state_dict
torch.save(a, '/home/hanyang/ik_lvjun.pt')
