"""
Re-implementation of [E.J. Smith, et al.: 3D Shape Reconstruction from Vision and Touch](https://arxiv.org/abs/2007.03778). 
Based on its official implementation.
"""

# model settings
model = dict(
    type="3DVisionAndTouch",
    backbone=dict(),
    head=dict(),
    train_cfg=dict(),
    test_cfg=dict(),
)

# dataset settings
category = 'box'
dataset_type = 'ArtiImgDataset'
data_root = '/disk1/data/arti_data/synthetic_data/' + category + '/'
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
    samples_per_gpu=128,
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
optimizer = dict(type='Adam', lr=0.001, weight_decay=0)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[260, 280])
runner = dict(type='EpochBasedRunner', max_epochs=300)

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
