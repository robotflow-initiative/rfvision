_base_ = ['../_base_/default_runtime.py',]
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup=None,
    step=[25, 40, 75, 90, 125, 140])
total_epochs = 150
checkpoint_config = dict(interval=49)

# model settings
model = dict(
    type='HandTailor',
    manonet=dict(type='ManoNetBackbone'),
    loss=dict(type='HandTailorLoss', ),
    iknet=dict(type='IKNet',
               backbone=dict(type='IKNetBackbone'),
               init_cfg='/home/hanyang/robotflow/work_dirs/iknet/new_100.pth'),
    backbone2d=dict(type='HandTailor2DBackbone'),
    backbone3d=dict(type='HandTailor3DBackbone'),
    epoch_2d=0,
    epoch_3d=50,
    epoch_mano=50,
    )

train_pipeline = [
    dict(type='HandTailorPipeline',
         num_joints=21,
         img_shape=(256, 256),
         heatmap_shape=(64, 64),
         heatmap_sigma=2,
         test_mode=False),
    dict(type='ToTensor', keys=['img', 'heatmap', 'heatmap_weight', 'K', 'joints_uvd']),
    dict(
        type='Collect',
        keys=['img', 'heatmap', 'heatmap_weight', 'K', 'joints_uvd'],
        meta_keys=[])
]

val_pipeline = [
    dict(type='HandTailorPipeline',
         num_joints=21,
         img_shape=(256, 256),
         heatmap_shape=(64, 64),
         heatmap_sigma=2,
         test_mode=True),
    dict(type='ToTensor',
         keys=['img', 'heatmap', 'heatmap_weight', 'K', 'joint_bone',
                'joints_uvd', 'joints_uv', 'joint_root', 'joints_xyz',]),
    dict(
        type='Collect',
        keys=['img', 'heatmap', 'heatmap_weight', 'K', 'joints_uvd',
              'joints_uv', 'joints_xyz', 'joint_root', 'joint_bone'],
        meta_keys=[])
]
test_pipeline = val_pipeline

data_root = '/home/hanyang/handdataset/FreiHAND_pub_v1'
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=0,
    train=dict(
        type='FreiHandDataset',
        pipeline=train_pipeline,
        data_root=data_root,
        split='train'),
    val=dict(
        type='FreiHandDataset',
        pipeline=test_pipeline,
        data_root=data_root,
        split='test'),
    test=dict(
        type='FreiHandDataset',
        pipeline=test_pipeline,
        data_root=data_root,
        split='test'),
)

runner = dict(type='HandTailorRunner', max_epochs=total_epochs)
find_unused_parameters=True