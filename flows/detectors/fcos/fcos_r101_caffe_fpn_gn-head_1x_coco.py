_base_ = './fcos_r50_caffe_fpn_gn-head_1x_coco.py'
model = dict(
    init_cfg='open-mmlab://detectron/resnet101_caffe',
    backbone=dict(depth=101))
