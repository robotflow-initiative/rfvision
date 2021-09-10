_base_ = [
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

model = dict(type='SOLOV2',
             init_cfg=None,
             backbone=dict(type='ResNet', depth=18,
                           num_stages=4,
                           frozen_stages=1,
                           out_indices=(0, 1, 2, 3)),
             neck=dict(type='FPN',
                       in_channels=[64, 128, 256, 512],
                       out_channels=256,
                       start_level=0,
                       num_outs=5,
                       upsample_cfg=dict(mode='nearest')),
             mask_feat_head=dict(type='MaskFeatHead',
                                 in_channels=256,
                                 out_channels=128,
                                 start_level=0,
                                 end_level=3),
             bbox_head=dict(type='SOLOv2Head',
                            num_classes=80,
                            in_channels=256,
                            seg_feat_channels=256,
                            stacked_convs=2,
                            strides=[8, 8, 16, 32, 32],
                            scale_ranges=((1, 56), (28, 112), (56, 224), (112, 448), (224, 896)),
                            num_grids=[40, 36, 24, 16, 12],
                            ins_out_channels=128
                            ),
             test_cfg=dict(nms_pre=500,
                           score_thr=0.1,
                           mask_thr=0.5,
                           update_thr=0.05,
                           kernel='gaussian',  # gaussian/linear
                           sigma=2.0,
                           max_per_img=30),
        )

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize',
         img_scale=[(768, 512), (768, 480), (768, 448),
                    (768, 416), (768, 384), (768, 352)],
         keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'],
         meta_keys=['filename', 'ori_shape', 'img_shape', 'pad_shape',
                    'scale_factor', 'flip', 'img_norm_cfg'],)
]

test_pipeline=[
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(768, 448),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

