dataset_type = 'ABCVisionDataset'
data_root = '/hdd0/data/abc/'
anno_root = '/home/hanyang/rfvision/rfvision/models/detectors3d/touch_and_vision/data'
num_samples = 10000
mode = 'no'
num_grasps=1

model = dict(type='VisionEncoder',
             anno_root=anno_root,
             mode='no',
             GEOmetrics=False,
             )


train_pipeline = [dict(type='Collect',
                       keys=['gt_points', 'img_occ', 'img_unocc', 'sheets', 'successful'],
                       meta_keys=[])]

data = dict(
    # batch_size: 16
    # random seed: 0
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        anno_root=anno_root,
        pipeline=train_pipeline,
        classes=['0001', '0002'],
        num_samples=num_samples,
        set_type='train',
        num_grasps=num_grasps,
        test_mode=False),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        anno_root=anno_root,
        pipeline=train_pipeline,
        classes=['0001', '0002'],
        num_samples=num_samples,
        set_type='test',
        num_grasps=num_grasps,
        test_mode=True),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        anno_root=anno_root,
        pipeline=train_pipeline,
        classes=['0001', '0002'],
        num_samples=num_samples,
        set_type='valid',
        num_grasps=num_grasps,
        test_mode=False))

evaluation = dict(interval=1, metric='bbox')

optimizer = dict(type='Adam', lr=0.001, weight_decay=0)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='fixed',
    warmup=None)
# lr_config = dict(
#     policy='step',
#     warmup=None,
#     step=[250, 275])
runner = dict(type='EpochBasedRunner', max_epochs=3000)

checkpoint_config = dict(interval=30)
log_config = dict(
    interval=50,
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
