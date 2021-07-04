# model settings
model = dict(
    type='DenseFusion',
    estimator=dict(type='DenseFusionEstimator',
                   backbone=dict(
                       type='DenseFusionResNet'),
                   neck=dict(
                       type='DenseFusionEstimatorNeck',
                       num_points=1000),
                   pose_head=dict(
                       type='DenseFusionEstimatorHead',
                       num_points=1000,
                       num_objects=21,
                       loss_dis=dict(type='DenseFusionEstimationLoss', num_points_mesh=500, sym_list=[12, 15, 18, 19, 20]))
                   ),
    refiner=dict(type='DenseFusionRefiner',
                 neck=dict(
                     type='DenseFusionRefinerNeck',
                     num_points=1000),
                 pose_head=dict(
                     type='DenseFusionRefinerHead',
                     num_points=1000,
                     num_objects=21,
                     loss_dis=dict(type='DenseFusionRefinementLoss', num_points_mesh=500, sym_list=[12, 15, 18, 19, 20])))
    )
# model training and testing settings
train_cfg = dict(densefusion_head=dict(type=None))
test_cfg = dict(densefusion_head=dict(type=None))
# dataset settings
dataset_type = 'YCBVideoDataset'
data_root = '/disk6/YCB_Video_Dataset/'
data_config_path = '/home/hanyang/rfvision/datasets/ycb_video'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadPoseData'),
    dict(type='PoseImgPreprocess', mode='train'),
    dict(type='CreatePoseGT', mode='train'),
    dict(type='DefaultPoseFormatBundle'),
    dict(type='Collect', keys=['cloud', 'choose', 'img', 'model_points', 'index', 'target'],
                         meta_keys=['img_prefix', 'sample_name']),
]
test_pipeline = [
    dict(type='LoadPoseData'),
    dict(type='PoseImgPreprocess', mode='test'),
    dict(type='CreatePoseGT', mode='test'),
    dict(type='DefaultPoseFormatBundle'),
    dict(type='Collect', keys=['cloud', 'choose', 'img', 'model_points', 'index', 'target'],
                         meta_keys=['img_prefix', 'sample_name']),
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        ann_file=data_config_path + 'train_data_list.txt',
        img_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_config_path + 'test_data_list.txt',
        img_prefix=data_root,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_config_path + 'test_data_list.txt',
        img_prefix=data_root,
        pipeline=test_pipeline))
# optimizer
optimizer = dict(type='Adam', lr=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=50,
    warmup_ratio=1.0 / 3,
    gamma=0.3,
    step=[500])
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
total_epochs = 500
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/densefusion_ycb_1x'
load_from = None
resume_from = None
workflow = [('train', 1)]
