_base_ = ['../_base_/default_runtime.py', ]

model = dict(
    type='HandTailor',
    manonet=dict(type='ManoNetBackbone'),
    loss=dict(type='HandTailorLoss'),
    iknet=dict(
        type='IKNet', backbone=dict(type='IKNetBackbone'), init_cfg=None),
    backbone2d=dict(type='HandTailor2DBackbone'),
    backbone3d=dict(type='HandTailor3DBackbone'),
    epoch_2d=50,
    epoch_3d=50,
    epoch_mano=50)

img_norm_cfg = dict(mean=[0, 0, 0], std=[255., 255., 255.], to_rgb=True)

train_pipeline = [
    dict(type='GetJointsUV'),
    dict(type='AffineCorp', centralize=True, img_outsize=(256, 256),
         rot_angle_range=(-180, 180)),
    dict(type='GenerateHeatmap2D', heatmap_shape=(64, 64), sigma=11),
    dict(type='JointsNormalize'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageFormatBundle'),
    dict(type='ToTensor', keys=['img', 'heatmap', 'heatmap_weight', 'K', 'joints_uvd']),
    dict(type='Collect3D', keys=['img', 'heatmap', 'heatmap_weight', 'K', 'joints_uvd'])
]

val_pipeline = [
    dict(type='GetJointsUV'),
    dict(type='Pad', size=(256, 256)),
    dict(type='GenerateHeatmap2D', heatmap_shape=(64, 64), sigma=11),
    dict(type='JointsNormalize'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageFormatBundle'),
    dict(type='ToTensor', keys=['img', 'heatmap', 'heatmap_weight', 'K', 'joint_bone',
                                'joints_uvd', 'joints_uv', 'joint_root', 'joints_xyz', ]),
    dict(type='Collect3D', keys=['img', 'heatmap', 'heatmap_weight', 'K', 'joint_bone',
                                 'joints_uvd', 'joints_uv', 'joint_root', 'joints_xyz', ]),
]

data_root = '/hdd0/data/FreiHAND_pub_v1'
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=8,
    train=dict(
        type='FreiHandDataset',
        pipeline=train_pipeline,
        data_root='/hdd0/data/FreiHAND_pub_v1',
        split='train'),
    val=dict(
        type='FreiHandDataset',
        pipeline=val_pipeline,
        data_root='/hdd0/data/FreiHAND_pub_v1',
        split='test'),
    test=dict(
        type='FreiHandDataset',
        pipeline=val_pipeline,
        data_root='/hdd0/data/FreiHAND_pub_v1',
        split='test'))

optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='step', warmup=None, step=[25, 40, 75, 90, 125, 140])
total_epochs = 150

runner = dict(type='EpochControlledRunner', max_epochs=150)
find_unused_parameters = True
gpu_ids = range(0, 1)
