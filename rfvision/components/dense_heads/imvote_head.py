import torch
from torch import nn as nn
from rfvision.models.builder import HEADS
from rfvision.components.utils import ImageFeatureModule, ImageMLPModule, append_img_feat, sample_valid_seeds
from .vote_head import VoteHead

class _VoteHead(VoteHead):
    def _extract_input(self, feat_tuple):
        return feat_tuple
       
class ImVoteModule(nn.Module):
    def __init__(self, 
                 max_imvote_per_pixel,
                 image_feature_dim, 
                 image_hidden_dim, 
                 ):
        super().__init__()
        self.max_imvote_per_pixel = max_imvote_per_pixel
        self.image_feature_dim = image_feature_dim
        self.image_feature_extractor = ImageFeatureModule(max_imvote_per_pixel=self.max_imvote_per_pixel)
        self.image_mlp = ImageMLPModule(image_feature_dim, image_hidden_dim=image_hidden_dim)
        
    def forward(self, feat_dict, imvote_dict):
        img_feat_list = self.image_feature_extractor(feat_dict, imvote_dict)
        assert len(img_feat_list) == self.max_imvote_per_pixel
        xyz, features, seed_inds = append_img_feat(img_feat_list, feat_dict, imvote_dict)
        seed_sample_inds = sample_valid_seeds(features[:,-1,:], 1024).cuda()
        features = torch.gather(features, -1, seed_sample_inds.unsqueeze(1).repeat(1,features.shape[1],1))
        xyz = torch.gather(xyz, 1, seed_sample_inds.unsqueeze(-1).repeat(1,1,3))
        seed_inds = torch.gather(seed_inds, 1, seed_sample_inds)
        
        pc_features = features[:,:256,:]
        img_features = features[:,256:,:]
        img_features = self.image_mlp(img_features)
        joint_features = torch.cat((pc_features, img_features), 1)
        results = {'seed_indices':seed_inds,
                   'seed_points':xyz,
                   'seed_features':features,
                   'img_features':img_features,
                   'pc_features':pc_features, 
                   'joint_features':joint_features,
                   } 
        
        return results
        
@HEADS.register_module()
class ImVoteHead(nn.Module):
    def __init__(self,
                 num_classes,
                 bbox_coder,
                 joint_only = False, 
                 train_cfg=None,
                 test_cfg=None,
                 imvote_module_cfg = None, 
                 vote_module_cfg_img_only=None,
                 vote_module_cfg_pc_only=None, 
                 vote_module_cfg_pc_img=None, 
                 vote_aggregation_cfg_img_only=None,
                 vote_aggregation_cfg_pc_only=None,
                 vote_aggregation_cfg_pc_img=None,
                 pred_layer_cfg=None,
                 conv_cfg=dict(type='Conv1d'),
                 norm_cfg=dict(type='BN1d'),
                 objectness_loss=None,
                 center_loss=None,
                 dir_class_loss=None,
                 dir_res_loss=None,
                 size_class_loss=None,
                 size_res_loss=None,
                 semantic_loss=None,
                 iou_loss=None,):
        super().__init__()
        self.joint_only = joint_only
        
        self.imvote_module = ImVoteModule(**imvote_module_cfg)
    
        self.img_only_head = _VoteHead(num_classes, bbox_coder, train_cfg, test_cfg, vote_module_cfg_img_only, vote_aggregation_cfg_img_only, 
                                       pred_layer_cfg, conv_cfg, norm_cfg, objectness_loss, center_loss, dir_class_loss, 
                                       dir_res_loss, size_class_loss, size_res_loss, semantic_loss, iou_loss)
                                       
        self.pc_only_head = _VoteHead(num_classes, bbox_coder, train_cfg, test_cfg, vote_module_cfg_pc_only, vote_aggregation_cfg_pc_only, 
                                      pred_layer_cfg, conv_cfg, norm_cfg, objectness_loss, center_loss, dir_class_loss, 
                                      dir_res_loss, size_class_loss, size_res_loss, semantic_loss, iou_loss)
        
        self.pc_img_head = _VoteHead(num_classes, bbox_coder, train_cfg, test_cfg, vote_module_cfg_pc_img, vote_aggregation_cfg_pc_img, 
                                     pred_layer_cfg, conv_cfg, norm_cfg, objectness_loss, center_loss, dir_class_loss, 
                                     dir_res_loss, size_class_loss, size_res_loss, semantic_loss, iou_loss)
        
    def forward(self, feat_dict, imvote_dict, sample_mod):
        feat_dict = self._extract_input(feat_dict)
        imvote_module_results = self.imvote_module(feat_dict, imvote_dict)
        results = {}
        if self.joint_only == False:
            # --------- IMAGE-ONLY TOWER ---------
            feat_tuple = (imvote_module_results['seed_points'],
                          imvote_module_results['img_features'],
                          imvote_module_results['seed_indices'])
                          
            img_only_results = self.img_only_head(feat_dict = feat_tuple, 
                                                  sample_mod = sample_mod)
            results['img_only_results'] = img_only_results
            # --------- POINTS-ONLY TOWER ---------
            feat_tuple = (imvote_module_results['seed_points'],
                          imvote_module_results['pc_features'],
                          imvote_module_results['seed_indices'])
            pc_only_results = self.pc_only_head(feat_dict = feat_tuple, 
                                                 sample_mod = sample_mod)
            results['pc_only_results'] = pc_only_results

        # --------- JOINT TOWER ---------
        feat_tuple = (imvote_module_results['seed_points'],
                      imvote_module_results['joint_features'],
                      imvote_module_results['seed_indices'])
        pc_img_results = self.pc_img_head(feat_dict = feat_tuple, 
                                          sample_mod = sample_mod)
        results['pc_img_results'] = pc_img_results

        return results
        
    def _extract_input(self,feat_dict):
        feat_dict['fp_xyz'] = feat_dict['fp_xyz'][-1]
        feat_dict['fp_features'] = feat_dict['fp_features'][-1]
        feat_dict['fp_indices'] = feat_dict['fp_indices'][-1]
        return feat_dict
        
        
    def loss(self,
             bbox_preds,
             points,
             gt_bboxes_3d,
             gt_labels_3d,
             pts_semantic_mask=None,
             pts_instance_mask=None,
             img_metas=None,
             gt_bboxes_ignore=None,
             ret_target=False):
             
        losses = {}
        if self.joint_only == False:
            img_only_loss_weight = 0.3
            pc_only_loss_weight = 0.3
            pc_img_loss_weight = 0.4
                   
            img_only_losses = self.img_only_head.loss(bbox_preds['img_only_results'], 
                                                      points,
                                                      gt_bboxes_3d,
                                                      gt_labels_3d,
                                                      pts_semantic_mask=None,
                                                      pts_instance_mask=None,
                                                      img_metas=None,
                                                      gt_bboxes_ignore=None,
                                                      ret_target=False)
            img_only_loss = sum([value for value in img_only_losses.values()])
            losses['img_only_loss'] = img_only_loss * img_only_loss_weight
            
            
            pc_only_losses = self.pc_only_head.loss(bbox_preds['pc_only_results'], 
                                                    points,
                                                    gt_bboxes_3d,
                                                    gt_labels_3d,
                                                    pts_semantic_mask=None,
                                                    pts_instance_mask=None,
                                                    img_metas=None,
                                                    gt_bboxes_ignore=None,
                                                    ret_target=False)
            pc_only_loss = sum([value for value in pc_only_losses.values()])
            losses['pc_only_loss'] = pc_only_loss * pc_only_loss_weight
        else:
            pc_img_loss_weight = 1
        pc_img_losses = self.pc_img_head.loss(bbox_preds['pc_img_results'], 
                                              points,
                                              gt_bboxes_3d,
                                              gt_labels_3d,
                                              pts_semantic_mask=None,
                                              pts_instance_mask=None,
                                              img_metas=None,
                                              gt_bboxes_ignore=None,
                                              ret_target=False)
        pc_img_loss = sum([value for value in pc_img_losses.values()])
        losses['pc_img_loss'] = pc_img_loss * pc_img_loss_weight
        return losses
        
        
    def init_weights(self):
        """Initialize weights of VoteHead."""
        pass
