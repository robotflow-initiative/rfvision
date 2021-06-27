_base_ = ['../_base_/default_runtime.py',]

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup=None,
    step=[50, 75])
total_epochs = 1
checkpoint_config = dict(interval=50)
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
# model settings
model = dict(
    type='IKNet',
    backbone=dict(type='IKNetBackbone'),
    )


data_root = '/home/hanyang/ik_dataset'
data = dict(
    samples_per_gpu=256,
    workers_per_gpu=0,
    train=dict(
        type='INVKDataset',
        data_root=data_root,
        split='train'),
    val=dict(
        type='INVKDataset',
        data_root=data_root,
        split='test'),
    test=dict(
        type='INVKDataset',
        data_root=data_root,
        split='test'),
)
