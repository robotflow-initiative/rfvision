model = dict(type='ArticulationEstimator')
data_root = '/disk4/data/arti_data/real_data/box/'
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type='ArticulationDataset',
        ann_file=data_root + 'train_meta.txt',
        img_prefix=data_root,
        intrinsics_path=data_root + 'camera_intrinsic.json',
        n_max_parts=13),
    val=dict(
        type='ArticulationDataset',
        ann_file=data_root + 'test_meta.txt',
        img_prefix=data_root,
        intrinsics_path=data_root + 'camera_intrinsic.json',
        n_max_parts=13),
    test=dict(
        type='ArticulationDataset',
        ann_file=data_root + 'test.txt',
        img_prefix=data_root,
        intrinsics_path=data_root + 'camera_intrinsic.json',
        n_max_parts=13),
)


checkpoint_config = dict(interval=20)
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = None
load_from = None
resume_from = None
workflow = [('train', 1)]

lr = 0.001  # max learning rate
optimizer = dict(type='Adam', lr=lr, weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))
lr_config = dict(policy='step', warmup='linear', warmup_iters=500, step=[80, 90])
# runtime settings
total_epochs = 100
find_unused_parameters = True