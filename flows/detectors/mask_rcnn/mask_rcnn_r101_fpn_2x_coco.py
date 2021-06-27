_base_ = './mask_rcnn_r50_fpn_2x_coco.py'
model = dict(init_cfg='torchvision://resnet101', backbone=dict(depth=101))
