# model settings
model = dict(
    type='ANCSH',
    backbone=dict(
        type='PointNet2ForArticulation',
        in_channels=3),
    nocs_head=dict(
        type='ANCSHHead',
        n_parts=3,
        mixed_pred=True,
        loss_miou=dict(type='ArtiMIoULoss'),
        loss_nocs=dict(type='ArtiNOCSLoss', TYPE_L='L2', MULTI_HEAD=True, SELF_SU=False),
        loss_vect=dict(type='ArtiVECTLoss', TYPE_L='L2', MULTI_HEAD=False, SELF_SU=False),
        loss_weights=[10.0, 1.0, 1.0, 5.0, 5.0, 0.2, 1.0, 1.0]))
# model training and testing settings
train_cfg = dict(
    rpn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.7,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=256,
            pos_fraction=0.5,
            neg_pos_ub=-1,
            add_gt_as_proposals=False),
        allowed_border=0,
        pos_weight=-1,
        debug=False),
    rpn_proposal=dict(
        nms_across_levels=False,
        nms_pre=2000,
        nms_post=2000,
        max_num=2000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0.5,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=64,
            pos_fraction=0.25,
            neg_pos_ub=-1,
            add_gt_as_proposals=True),
        mask_size=28,
        pos_weight=-1,
        debug=False))
test_cfg = dict(
    rpn=dict(
        nms_across_levels=False,
        nms_pre=1000,
        nms_post=1000,
        max_num=1000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        score_thr=0.05,
        nms=dict(type='nms', iou_thr=0.5),
        max_per_img=100,
        mask_thr_binary=0.5))
# dataset settings
category = 'box'
data_track = 'real'
if data_track == 'synthetic':
    dataset_type = 'ArtiImgDataset'
    data_root = '/disk1/data/arti_data/synthetic_data/' + category + '/'
else:
    dataset_type = 'ArtiImgDataset'
    data_root = '/disk1/data/arti_data/real_data/' + category + '/'

test_dataset_type = 'ArtiImgDataset'
test_data_root = '/disk1/data/arti_data/real_data/' + category + '/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='CreatePointData', downsample_voxel=0.005),
    dict(type='LoadArtiNOCSData'),
    dict(type='LoadArtiJointData'),
    dict(type='CreateArtiJointGT'),
    dict(type='DownSampleArti', num_points=1024),
    dict(type='DefaultFormatBundleArti'),
    dict(type='Collect', keys=['parts_pts', 'parts_pts_feature', 'parts_cls', 'mask_array',
                                'nocs_p', 'nocs_g', 'offset_heatmap',
                                'offset_unitvec', 'joint_orient', 'joint_cls',
                                'joint_cls_mask', 'joint_params'],
                         meta_keys=['img_prefix', 'sample_name', 'norm_factors', 'corner_pts',
                                    'joint_ins']),
]
test_pipeline = [
    dict(type='CreatePointData', downsample_voxel=0.005),
    dict(type='LoadArtiNOCSData'),
    dict(type='LoadArtiJointData'),
    dict(type='CreateArtiJointGT'),
    dict(type='DownSampleArti', num_points=2048),
    dict(type='DefaultFormatBundleArti'),
    dict(type='Collect', keys=['parts_pts', 'parts_pts_feature', 'parts_cls', 'mask_array',
                                'nocs_p', 'nocs_g', 'offset_heatmap',
                                'offset_unitvec', 'joint_orient', 'joint_cls',
                                'joint_cls_mask', 'joint_params'],
                         meta_keys=['img_prefix', 'sample_name', 'norm_factors', 'corner_pts',
                                    'joint_ins']),
]
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=16,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'train_meta.txt',
        img_prefix=data_root,
        intrinsics_path=data_root + 'camera_intrinsic.json',
        pipeline=train_pipeline,
        domain='real',
        n_max_parts=3),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'test.txt',
        img_prefix=data_root,
        intrinsics_path=data_root + 'camera_intrinsic.json',
        pipeline=test_pipeline,
        domain='real',
        n_max_parts=3
    ),
    test=dict(
        type=test_dataset_type,
        ann_file=test_data_root + 'test.txt',
        img_prefix=test_data_root,
        intrinsics_path=test_data_root + 'camera_intrinsic.json',
        pipeline=test_pipeline,
        domain='real',
        n_max_parts=3
    ))
# optimizer
optimizer = dict(type='Adam', lr=0.001)
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=50,
    warmup_ratio=1.0 / 3,
    gamma=0.7,
    step=[4, 8, 12, 16])
checkpoint_config = dict(interval=10)
# yapf:disable
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
    ])
# yapf:enable
evaluation = dict(interval=100)
# runtime settings
total_epochs = 20
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/ancsh_pointnet_1x_{}_{}2'.format(data_track, category)
load_from = None
resume_from = None
workflow = [('train', 1)]
