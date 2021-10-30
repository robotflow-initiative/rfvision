_base_ = ['../_base_/default_runtime.py',]

optimizer = dict(type='SGD', lr=0.00001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup=None,
    step=[35, 45])
total_epochs = 50
checkpoint_config = dict(interval=10)
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
# model settings
model = dict(type='INKVNet')
work_dir = '/home/hanyang/work_dir/iknet'
data_root = '/home/hanyang/ikdata/'
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=0,
    train=dict(
        type='IKDataset',
        data_root=data_root,
        split='all'),
    val=dict(
        type='IKDataset',
        data_root=data_root,
        split='test'),
    test=dict(
        type='IKDataset',
        data_root=data_root,
        split='test'))