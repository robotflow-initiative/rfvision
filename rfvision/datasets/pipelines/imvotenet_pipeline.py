import numpy as np
import robotflow
import os
from robotflow.rflearner.datasets.builder import PIPELINES
import torch 

@PIPELINES.register_module()
class LoadImVote:
    def __init__(self,
                 data_root, 
                 class_names,
                 max_imvote_per_pixel=3,
                 ):
        
        self.data_root = data_root
        self.max_imvote_per_pixel = max_imvote_per_pixel
        self.vote_dims = 1+self.max_imvote_per_pixel*4
        self.cat2label = {cat: class_names.index(cat) for cat in class_names}
        self.MAX_NUM_2D_DET = 100
        self.MAX_NUM_PIXEL = 530 * 730
        self.NUM_CLS = len(class_names)
        
        self.cls_id_map = {}
        self.cls_score_map = {}
        self.bbox_2d_map = {}
                
        bbox2d_train_dir = os.path.join(data_root, 'sunrgbd_2d_bbox_50k_v1_train')
        bbox2d_test_dir = os.path.join(data_root, 'sunrgbd_2d_bbox_50k_v1_val')
        cls_id_map_train , cls_score_map_train, bbox_2d_map_train = self._pre_load_bbox2d(bbox2d_train_dir)
        cls_id_map_test , cls_score_map_test, bbox_2d_map_test = self._pre_load_bbox2d(bbox2d_test_dir)
        
        self.cls_id_map.update(cls_id_map_train)
        self.cls_id_map.update(cls_id_map_test)
        self.cls_score_map.update(cls_score_map_train)
        self.cls_score_map.update(cls_score_map_test)
        self.bbox_2d_map.update(bbox_2d_map_train)
        self.bbox_2d_map.update(bbox_2d_map_test)
        
        
    def _pre_load_bbox2d(self,bbox2d_dir):
        cls_id_map = {}
        cls_score_map = {}
        bbox_2d_map = {}
        for filename in os.listdir(bbox2d_dir):
            # Read 2D object detection boxes and scores
            cls_id_list = []
            cls_score_list = []
            bbox_2d_list = []
            idx = int(filename[:6])
            for line in open(os.path.join(bbox2d_dir , filename), 'r'):
                det_info = line.rstrip().split(" ")
                prob = float(det_info[-1])
                # Filter out low-confidence 2D detections
                if prob < 0.1:
                    continue
                cls_id_list.append(self.cat2label[det_info[0]])
                cls_score_list.append(prob)
                bbox_2d_list.append(np.float32(det_info[4:8]).astype('int32'))
            cls_id_map[idx] = cls_id_list
            cls_score_map[idx] = cls_score_list
            bbox_2d_map[idx] = bbox_2d_list
        return cls_id_map, cls_score_map, bbox_2d_map
        
    def get_imvote(self, idx):
        # Read image
        full_img = robotflow.rflib.imread(os.path.join(self.data_root, 'sunrgbd_trainval/image/', f'{idx:06d}.jpg'))
        full_img_height = full_img.shape[0]
        full_img_width = full_img.shape[1]
        
        # Read camera parameters
        # ------------------------------- 2D IMAGE VOTES ------------------------------
        cls_id_list = self.cls_id_map[idx]
        cls_score_list = self.cls_score_map[idx]
        bbox_2d_list =self. bbox_2d_map[idx]
        obj_img_list = []
        for i2d, (cls2d, box2d) in enumerate(zip(cls_id_list, bbox_2d_list)):
            xmin, ymin, xmax, ymax = box2d
            # During training we randomly drop 2D boxes to reduce over-fitting
            if np.random.random()>0.5:
                continue
            obj_img = full_img[ymin:ymax, xmin:xmax, :]
            obj_h = obj_img.shape[0]
            obj_w = obj_img.shape[1]
            # Bounding box coordinates (4 values), class id, index to the semantic cues
            meta_data = (xmin, ymin, obj_h, obj_w, cls2d, i2d)
            if obj_h == 0 or obj_w == 0:
                continue

            # Use 2D box center as approximation
            uv_centroid = np.array([int(obj_w/2), int(obj_h/2)])
            uv_centroid = np.expand_dims(uv_centroid, 0)

            v_coords, u_coords = np.meshgrid(range(obj_h), range(obj_w), indexing='ij')
            img_vote = np.transpose(np.array([u_coords, v_coords]), (1,2,0))
            img_vote = np.expand_dims(uv_centroid, 0) - img_vote 

            obj_img_list.append((meta_data, img_vote))

        full_img_votes = np.zeros((full_img_height,full_img_width,self.vote_dims), dtype=np.float32)
        # Empty votes: 2d box index is set to -1
        full_img_votes[:,:,3::4] = -1.

        for obj_img_data in obj_img_list:
            meta_data, img_vote = obj_img_data
            u0, v0, h, w, cls2d, i2d = meta_data
            for u in range(u0, u0+w):
                for v in range(v0, v0+h):
                    iidx = int(full_img_votes[v,u,0])
                    if iidx >= self.max_imvote_per_pixel: 
                        continue
                    full_img_votes[v,u,(1+iidx*4):(1+iidx*4+2)] = img_vote[v-v0,u-u0,:]
                    full_img_votes[v,u,(1+iidx*4+2)] = cls2d
                    full_img_votes[v,u,(1+iidx*4+3)] = i2d + 1 # add +1 here as we need a dummy feature for pixels outside all boxes
            full_img_votes[v0:(v0+h), u0:(u0+w), 0] += 1

        full_img_votes_1d = np.zeros((self.MAX_NUM_PIXEL*self.vote_dims), dtype=np.float32)
        full_img_votes_1d[0:full_img_height*full_img_width*self.vote_dims] = full_img_votes.flatten()
        full_img_1d = ((full_img - 128) / 255.)
        full_img_1d = np.zeros((self.MAX_NUM_PIXEL*3), dtype=np.float32)
        full_img_1d[:full_img_height*full_img_width*3] = full_img.flatten()

        # Semantic cues: one-hot vector for class scores
        cls_score_feats = np.zeros((1+self.MAX_NUM_2D_DET,self.NUM_CLS), dtype=np.float32)
        # First row is dumpy feature
        len_obj = len(cls_id_list)
        if len_obj:
            ind_obj = np.arange(1,len_obj+1)
            ind_cls = np.array(cls_id_list)
            cls_score_feats[ind_obj, ind_cls] = np.array(cls_score_list)
        
        imvote_dict = {}
        imvote_dict['cls_score_feats'] = cls_score_feats.astype(np.float32)
        imvote_dict['full_img_votes_1d'] = full_img_votes_1d.astype(np.float32)
        imvote_dict['full_img_1d'] = full_img_1d.astype(np.float32)
        imvote_dict['full_img_width'] = full_img_width
        return imvote_dict
    
    def __call__(self, results):
        info = results['ann_info']['info']
        imvote_dict = self.get_imvote(results['sample_idx'])
        
        # update Rtilt according to aug
        Rtilt = info['calib']['Rt']
        rot_mat_T = np.eye(3).T
        # rotation 
        if 'pcd_rotation' in results:
            rot_mat_T = results['pcd_rotation']
        # filp 
        if results['pcd_horizontal_flip'] == True:
            rot_mat_T[0,:] *= -1    
        Rtilt = np.dot(rot_mat_T.T, Rtilt)
        # scale
        pcd_scale_factor = np.float32([results['pcd_scale_factor']])
        Rtilt = np.dot(np.eye(3) * pcd_scale_factor, Rtilt)
        # add additional info to imvote_dict
        imvote_dict['scale'] = pcd_scale_factor.astype('float32')
        imvote_dict['calib_Rtilt'] = Rtilt.astype('float32')
        imvote_dict['calib_K'] = info['calib']['K'].reshape(3, 3, order = 'F').astype('float32')
        results['imvote_dict'] = imvote_dict
        return results
    
