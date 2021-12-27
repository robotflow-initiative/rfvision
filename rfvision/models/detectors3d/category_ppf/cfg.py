checkpoint_config = dict(interval=100)
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = None
load_from = None
resume_from = None
workflow = [('train', 1)]

lr = 1e-3
optimizer = dict(type='Adam', lr=lr, weight_decay=0)
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))
lr_config = dict(policy='step', warmup=None, step=[24, 32])
# runtime settings
total_epochs = 200

model = dict(type='CategoryPPF',
             category=2)
train_dataset = 'ShapeNetDatasetForPPF'

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=0,
    train=dict(type=train_dataset,
               category=2,
               data_root='/hdd0/data/shapenet_v2/ShapeNetCore.v2',
               ann_file='/hdd0/data/ppf_dataset/shapenet_train.txt'),
)
gpu_ids = [0]
seed = 0
