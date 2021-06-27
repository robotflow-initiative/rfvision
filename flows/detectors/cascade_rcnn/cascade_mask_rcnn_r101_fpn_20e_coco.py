_base_ = './cascade_mask_rcnn_r50_fpn_20e_coco.py'
model = dict(init_cfg='torchvision://resnet101', backbone=dict(depth=101))
