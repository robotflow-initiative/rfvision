import torch
from rfvision.models.human_analyzers.utils.mano_layers import ManoLayer
from rfvision.models import build_detector
from rflib.runner import load_checkpoint
import rflib
import pickle
import matplotlib.pyplot as plt
import numpy as np
import cv2
import trimesh
from rfvision.components.utils.top_down_utils import xyz2uv
import open3d as o3d

def save_pc(xyz):
    xyz = o3d.utility.Vector3dVector(xyz)
    pc = o3d.geometry.PointCloud(xyz)
    o3d.io.write_point_cloud('/home/hanyang/test.ply', pc)

def normalize_point_cloud(pc):
    centroid = pc[9]
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc_normalized = pc / m
    return pc_normalized, centroid, m

triangles = np.loadtxt('/home/hanyang/weights/handtailor/hand.npy')
pts_ik = '/home/hanyang/work_dir/iknet/epoch_50.pth'
pts_ik = '/home/hanyang/ik_lvjun.pt'
cfg_ik = '/home/hanyang/rfvision/flows/human_analyzers/hand/others/iknet.py'
cfg_ik = rflib.Config.fromfile(cfg_ik)
m_ik = build_detector(cfg_ik.model)
load_checkpoint(m_ik, pts_ik)
m_ik.eval()


# id = 7
dataset = pickle.load(open('/home/hanyang/show_dir/rhd/test_0_to_10.pkl', 'rb'))

# dataset = pickle.load(open('/home/hanyang/show_dir/interhand3d/test_0_to_10.pkl', 'rb'))
for id in range(10):
    data = dataset[id]
    img_metas = data['img_metas'].data
    img_metas['joints_cam'] = img_metas['joints_xyz']
    img_ori = cv2.imread(img_metas['image_file'])
    princpt = img_metas['princpt']
    focal = img_metas['focal']
    K = np.array([[focal[0], 0, princpt[0]],
                     [0, focal[1], princpt[1]],
                     [0, 0, 1]])
    joints_uv = xyz2uv(np.array(img_metas['joints_cam']).copy(), K)
    for i in joints_uv[:21]:
        cv2.drawMarker(img_ori, np.int0(i[:2]), (0, 255, 0), markerSize=5)
    plt.imshow(img_ori[:,:,::-1])
    plt.show()

    # joints_xyz = np.array(img_metas['joints_xyz']).copy()
    joints_xyz = np.array(img_metas['joints_cam']).copy()[:21]
    joints_xyz /= 1000
    reorder = np.array([20, 3,2,1,0, 7,6,5,4, 11,10,9,8, 15,14,13,12, 19,18,17,16])

    joint_bone = np.linalg.norm(joints_xyz[9] - joints_xyz[0])
    joints_xyz / joint_bone

    # joints_xyz = joints_xyz[reorder]
    joints_xyz = normalize_point_cloud(joints_xyz)[0]
    save_pc(joints_xyz)
    joints_xyz = torch.FloatTensor(joints_xyz).unsqueeze(0)

    with torch.no_grad():
        full_poses_res = m_ik(joints_xyz=joints_xyz, return_loss=False)
        full_poses_res = full_poses_res.cpu().numpy()

    mano = ManoLayer(mano_root='/home/hanyang/weights/handtailor', flat_hand_mean=True, use_pca=False)
    vertices, jtr, full_pose = mano(full_poses_res, betas =np.ones((1, 10)))
    # vertices, jtr, full_pose = mano(full_poses_res, betas =np.ones((1, 10)))

    mesh = trimesh.Trimesh(faces=triangles, vertices=vertices[0])
    mesh = mesh.as_open3d
    H = np.eye(4)
    H[2,2], H[1,1] = -1, -1
    mesh.transform(H)
    o3d.io.write_triangle_mesh(f'/home/hanyang/handtailor_test/test' + str(id) + '.obj', mesh)

def test(id = 20000):
    triangles = np.loadtxt('/home/hanyang/weights/handtailor/hand.npy')
    pts_ik = '/home/hanyang/work_dir/iknet/epoch_50.pth'
    cfg_ik = '/home/hanyang/rfvision/flows/human_analyzers/hand/iknet.py'
    cfg_ik = rflib.Config.fromfile(cfg_ik)
    m_ik = build_detector(cfg_ik.model)
    load_checkpoint(m_ik, pts_ik)

    m_ik.eval()

    joints_xyz = np.load('/home/hanyang/ikdata/hanyang/joints_xyz.npy')
    full_poses = np.load('/home/hanyang/ikdata/hanyang/full_poses.npy')

    joints_xyz_test = torch.FloatTensor(joints_xyz[id] - joints_xyz[id][0]).unsqueeze(0)
    full_poses_test = torch.FloatTensor(full_poses[id:id + 1])

    with torch.no_grad():
        print(m_ik(joints_xyz = joints_xyz_test, full_poses=full_poses_test))
        full_poses_res = m_ik(joints_xyz = joints_xyz_test, return_loss=False)
        full_poses_res = full_poses_res.cpu().numpy()


    mano = ManoLayer(mano_root='/home/hanyang/weights/handtailor', flat_hand_mean=True, use_pca=False)
    vertices, jtr, full_pose = mano(full_poses[id: id+1], betas =np.ones((1, 10)))
    # vertices, jtr, full_pose = mano(full_poses_res, betas =np.ones((1, 10)))

    mesh = trimesh.Trimesh(faces=triangles, vertices=vertices[0])
    mesh = mesh.as_open3d
    H = np.eye(4)
    # H[2,2], H[1,1] = -1, -1
    mesh.transform(H)
    o3d.io.write_triangle_mesh('/home/hanyang/test.obj', mesh)
    save_pc(joints_xyz[id])

# test()