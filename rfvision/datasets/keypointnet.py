import open3d as o3d
import os 
import numpy as np
import torch
import collections
from . import DATASETS
from .custom3d import Custom3DDataset
from .pipelines import Compose
import rflib
from rfvision.core.evaluation_pose import keypoint_epe, keypoint_pck_accuracy,  keypoint_auc



ID_CLASSES = {"02691156": "airplane",
              "02808440": "bathtub",
              "02818832": "bed",
              "02876657": "bottle",
              "02954340": "cap",
              "02958343": "car",
              "03001627": "chair",
              "03467517": "guitar",
              "03513137": "helmet",
              "03624134": "knife",
              "03642806": "laptop",
              "03790512": "motorcycle",
              "03797390": "mug",
              "04225987": "skateboard",
              "04379243": "table",
              "04530566": "vessel",}

CLASSES_ID = {value:key for key, value in ID_CLASSES.items()}
CLASSES_ALL = list(CLASSES_ID.keys())
ID_ALL = list(CLASSES_ID.values())
ID2LABEL = {ID:i for i,ID in enumerate(ID_ALL)}


@DATASETS.register_module()
class KeypointNetDataset(Custom3DDataset):
    def __init__(self,
                 data_root,
                 split_file_root='./data/keypointnet',
                 split='test',
                 with_ply=False,
                 classes=['chair'], 
                 pipeline=None,
                 test_mode=False,
                 ):
        assert split in ('train', 'val', 'test', 'all')
        self.split = split
        self.data_root = data_root
        self.anno_root = os.path.join(self.data_root, 'annotations')
        self.pcd_root = os.path.join(self.data_root, 'pcds')
        self.ply_root = os.path.join(self.data_root, 'ShapeNetCore.v2.ply')
        self.split_file_root = split_file_root
        self.with_ply = with_ply
        self.test_mode = test_mode
        
        self.CLASSES = classes
        self.annos = self.load_annotations()

        if pipeline is not None:
            self.pipeline = Compose(pipeline)

        if not self.test_mode:
            self._set_group_flag()

    def load_annotations(self):
        # ID
        if self.CLASSES[0] == 'all':
            ID = ID_ALL
        else:
            ID = [CLASSES_ID[CLASS] for CLASS in self.CLASSES]
            
        # split
        annos_split = []
        if self.split == 'all':
            annos = rflib.load(os.path.join(self.anno_root,'all.json'))
            for anno in annos:
                if anno['class_id'] in ID:
                    annos_split.append(anno)
        else:
            if self.split == 'train':
                with open(os.path.join(self.split_file_root, 'train.txt'), 'r') as f:
                    split_set = f.readlines()
            elif self.split == 'test':
                with open(os.path.join(self.split_file_root, 'test.txt'), 'r') as f:
                    split_set = f.readlines()
            elif self.split == 'val':
                with open(os.path.join(self.split_file_root, 'val.txt'), 'r') as f:
                    split_set = f.readlines()
            ID_split = tuple(i[9:].rstrip('\n') for i in split_set if i[:8] in ID)
            for CLASS in self.CLASSES:
                annos = rflib.load(os.path.join(self.anno_root, f'{CLASS}.json'))
                for anno in annos:
                    if anno['model_id'] in ID_split:
                        annos_split.append(anno)
        return annos_split
    
    def __len__(self,):
        return len(self.annos)
    
    def __getitem__(self, index):
        results = {}
        # read points
        pcd_filename = os.path.join(self.pcd_root, self.annos[index]['class_id'], self.annos[index]['model_id'] + '.pcd')
        pc = o3d.io.read_point_cloud(pcd_filename)
        points = np.array(pc.points)
        colors = np.array(pc.colors)

        # read annos
        keypoints_xyz = np.array([kp['xyz'] for kp in self.annos[index]['keypoints']]) 
        semantic_id = np.array([kp['semantic_id'] for kp in self.annos[index]['keypoints']])
        keypoints_index = np.array([(kp['pcd_info']['point_index']) for kp in self.annos[index]['keypoints']])

        # pcd infos
        results['pcd_filename'] = pcd_filename
        results['points'] = points.astype('float32')                       # numpy array, shape: (2048, 3)
        results['colors'] = colors.astype('float32')
        # keypoints infos
        results['keypoints_xyz'] = keypoints_xyz.astype('float32')         # numpy array, shape: (n_keypoint, 3)
        results['keypoints_semantic_id'] = semantic_id.astype('int32')     # numpy array, shape: (n_keypoint,)
        results['keypoints_index'] = keypoints_index.astype('int32')       # numpy array, shape: (n_keypoint,)
        # other infos
        results['model_id'] = self.annos[index]['model_id']
        results['class_id'] = self.annos[index]['class_id']

        # add ply infos
        if self.with_ply == True:
            # read mesh
            ply_filename = os.path.join(self.ply_root, self.annos[index]['class_id'],
                                        self.annos[index]['model_id'] + '.ply')
            mesh = o3d.io.read_triangle_mesh(ply_filename)
            vertices = np.array(mesh.vertices)
            faces = np.array(mesh.triangles)
            vertex_colors = np.array(mesh.vertex_colors)

            results['ply_filename'] = ply_filename
            results['vertices'] = vertices.astype('float32')
            results['faces'] = faces.astype('float32')
            results['vertex_colors'] = vertex_colors.astype('float32')
        results = self.pipeline(results)
        return results
    
    def fwd_alignment_scores(self, gt_results , pred_results):
        pred_semantic_id = []
        for gt_result, pred_result in zip(gt_results, pred_results):
            pred_kps_xyz = pred_result['keypoints_xyz'].transpose(1,0) # k1 x 1 x 3
            gt_kps_xyz = gt_result['keypoints_xyz'].unsqueeze(0) # 1 x k2 x 3
            dist = torch.sum(torch.square(pred_kps_xyz.cuda() - gt_kps_xyz.cuda()), -1)  # k1 x k2
            argminfwd = torch.argmin(dist, -1)  # k1
            pred_semantic_id.append([gt_result['keypoints_semantic_id'][argm] for argm in argminfwd])
        pred_semantic_id = torch.tensor(pred_semantic_id, dtype=torch.int32)  # n x k1
        acc = []
        for pa in pred_semantic_id:
            for pb in pred_semantic_id:
                acc.append(torch.mean((pa == pb).float()))
        return torch.mean(torch.tensor(acc))

    def bwd_alignment_scores(self, gt_results, pred_results):
        preds = collections.defaultdict(list)

        for gt_result, pred_result in zip(gt_results, pred_results):
            pred_kps_xyz = pred_result['keypoints_xyz'].transpose(1,0) # k1 x 1 x 3
            gt_kps_xyz = gt_result['keypoints_xyz'].unsqueeze(0) # 1 x k2 x 3
            dist = torch.sum(torch.square(pred_kps_xyz.cuda() - gt_kps_xyz.cuda()), -1)  # k1 x k2
            argminbwd = torch.argmin(dist, -2)  # k2
            for i, sem_id in enumerate(gt_result['keypoints_semantic_id']):
                preds[int(sem_id)].append(int(argminbwd[i])) # 'int' : convert 'tensor' to 'python int'
        q = []
        for semarr in preds.values():
            semarr = torch.tensor(semarr)
            q.append(torch.mean((semarr.unsqueeze(-1) == semarr.unsqueeze(0)).float()))
        return torch.mean(torch.tensor(q))

    def mIoU(self, gt_results, pred_results, thresholds):

        pred_kps_list = []
        gt_kps_list = []
        for gt_result, pred_result in zip(gt_results, pred_results):
            # release
            gt_kps_index = gt_result['keypoints_index']
            gt_points = gt_result['points']
            pred_kps_xyz = pred_result['keypoints_xyz']

            pred_kps_xyz = pred_kps_xyz.transpose(1,0)
            gt_points = torch.unsqueeze(gt_points, 0)
            dist = torch.sqrt(torch.sum(torch.square(pred_kps_xyz.cuda() - gt_points.cuda()), -1))  # k1 x k2
            argminfwd = torch.argmin(dist, -1)  # k1
            pred_kps_list.append(gt_points[argminfwd])
            gt_kps_list.append(gt_points[gt_kps_index.long()])  # tensor indice 'long' type is needed
            
        for threshold in thresholds:
            npos = 0
            fp_sum = 0
            fn_sum = 0
            for ground_truths, kpcd in zip(gt_kps_list, pred_kps_list):
                kpcd_e = np.expand_dims(kpcd, 1)  # k1 x 1 x 3
                gt_e = np.expand_dims(ground_truths, 0)  # 1 x k2 x 3
                dist = np.sqrt(np.sum(np.square(kpcd_e - gt_e), -1))  # k1 x k2
                npos += len(np.min(dist, -2))
                fp_sum += np.count_nonzero(np.min(dist, -1) > threshold)
                fn_sum += np.count_nonzero(np.min(dist, -2) > threshold)            
        return (npos - fn_sum) / (npos + fp_sum)
    
    def show(self,results,out_dir,filename,view_x_rotation=0, view_y_rotation=0,show=True):
        assert out_dir is not None, 'Expect out_dir, got none.'
        pcd_filename = results['pcd_filename']
        pc = o3d.io.read_point_cloud(pcd_filename)
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(pc)
        for kp_idx in results['keypoints_index']:
            mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
            mesh_sphere.translate(pc[kp_idx])
            mesh_sphere.paint_uniform_color([1,0,0]) # red
            vis.add_geometry(mesh_sphere)
        if show == True:
            vis.run()
        else:
            vis.capture_screen_image(os.path.join(out_dir, filename), do_render = True)
        
    def evaluate(self,
                 results,
                 metric='DAS',
                 logger=None,
                 by_epoch=None,
                 ):

        assert metric in ['DAS', 'mIoU']
        # get gt_results
        gt_results = []
        for index in range(len(self.annos)):
            gt_result = self.__getitem__(index)
            gt_results.append(gt_result)
        # get pred_results
        pred_results = results
        # calculate score
        ret_dict = {}
        if metric == 'DAS':
            forward_alignment_score = self.fwd_alignment_scores(gt_results , pred_results)
            backward_alignment_score = self.bwd_alignment_scores(gt_results, pred_results)
            ret_dict = {'forward_alignment_score':float(forward_alignment_score),
                        'backward_alignment_score':float(backward_alignment_score)}
        elif metric == 'mIoU':
            mIoU = self.mIoU(gt_results, pred_results, thresholds = np.linspace(0., 0.1))
            ret_dict = {'mIoU': float(mIoU)}
        return ret_dict


    # def evaluate(self, results, metric, **kwargs):
    #     # notes:
    #     # kps: keypoints
    #     metrics = metric if isinstance(metric, list) else [metric]
    #     allowed_metrics = ['PCK', 'AUC', 'EPE']
    #     for metric in metrics:
    #         if metric not in allowed_metrics:
    #             raise KeyError(f'metric {metric} is not supported')
    #
    #     gt_kps, pred_kps = (), ()
    #     gt_kps_visible = ()
    #     for i in range(len(results)):
    #         gt_result = self[i]
    #         pred_result = results[i]
    #
    #         gt_kps_visible += (gt_result['kps_visible'],)
    #         gt_kps += (gt_result['kps'],)
    #         pred_kps += (pred_result['kps'],)
    #
    #     gt_kps_visible = np.array(gt_kps_visible)
    #     gt_kps = np.array(gt_kps)
    #     pred_kps = np.array(torch.cat(pred_kps).cpu())
    #
    #     normalize = np.ones((len(results), 2))
    #
    #     score = {}
    #     for metric in metrics:
    #         if metric == 'EPE':
    #             score['EPE'] = keypoint_epe(pred_kps, gt_kps, gt_kps_visible)
    #         elif metric == 'AUC':
    #             score['AUC'] = keypoint_auc(pred_kps, gt_kps, gt_kps_visible,
    #                                         normalize=normalize)
    #         elif metric == 'PCK':
    #             score['PCK'] = keypoint_pck_accuracy(pred_kps, gt_kps, gt_kps_visible,
    #                                                  thr=0.2,
    #                                                  normalize=normalize)[1]
    #
    #     # rflib.dump(score, os.path.join(res_folder, 'result_keypoints.json'))
    #
    #     return score
if __name__ == '__main__':
    dataset = KeypointNetDataset(data_root = '')