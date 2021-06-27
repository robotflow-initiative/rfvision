_base_ = './cascade_mask_rcnn_r50_fpn_1x_coco.py'
model = dict(init_cfg='torchvision://resnet101', backbone=dict(depth=101))
