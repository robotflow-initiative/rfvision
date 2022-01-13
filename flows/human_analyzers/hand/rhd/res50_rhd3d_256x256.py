_base_ = ['../../_base_/datasets/rhd2d.py']
log_level = 'INFO'
load_from = None
resume_from = None
dist_params = dict(backend='nccl')
workflow = [('train', 1)]
checkpoint_config = dict(interval=50)
evaluation = dict(
    interval=50,
    metric=['MRRPE', 'MPJPE', 'Handedness_acc'],
    save_best='MPJPE_all')
optimizer = dict(
    type='Adam',
    lr=1e-4,
)
gpu_ids = range(1, 3)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[180, 200])
total_epochs = 210

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])

channel_cfg = dict(
    num_output_channels=21,
    dataset_joints=21,
    dataset_channel=[
        [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
            19, 20
        ],
    ],
    inference_channel=[
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
        20
    ])
# model settings
model = dict(
    type='Interhand3D',
    backbone=dict(type='ResNet', depth=50, init_cfg='torchvision://resnet50',),
    keypoint_head=dict(
        type='Topdown3DHeatmapSimpleHead',
        keypoint_head_cfg=dict(
            in_channels=2048,
            out_channels=21 * 64,
            depth_size=64,
            num_deconv_layers=3,
            num_deconv_filters=(256, 256, 256),
            num_deconv_kernels=(4, 4, 4),
        ),
        root_head_cfg=dict(
            in_channels=2048,
            heatmap_size=64,
            hidden_dims=(512, ),
        ),
        hand_type_head_cfg=dict(
            in_channels=2048,
            num_labels=2,
            hidden_dims=(512, ),
        ),
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True),
        loss_root_depth=dict(type='L1LossPose', use_target_weight=True),
        loss_hand_type=dict(type='BCELoss', use_target_weight=True)
        ),
)
train_cfg = dict()
test_cfg = dict(
    flip_test=False,
    post_process='default',
    shift_heatmap=True,
    modulate_kernel=11)
# In RHD:
# For train set, joints_z in range (-52.540000915527344, 1182.0)
# For test set, joints_z in range (-48.76000213623047, 994.0)
# For train set, joints_z (root_relative, root_joint_id: 0) in range (-326.9000244140625, 294.6999816894531)
# For test set, joints_z (root_relative, root_joint_id: 0) in range (-199.5999755859375, 189.99996948242188)

data_cfg = dict(
    image_size=[256, 256],
    heatmap_size=[64, 64, 64],
    heatmap3d_depth_bound=[-326.9000244140625, 294.6999816894531],
    heatmap_size_root=64,
    root_depth_bound=[-52.540000915527344, 1182.0],
    num_output_channels=channel_cfg['num_output_channels'],
    num_joints=channel_cfg['dataset_joints'],
    dataset_channel=channel_cfg['dataset_channel'],
    inference_channel=channel_cfg['inference_channel'])


train_pipeline = [
    dict(type='LoadImageFromFileSimple'),
    # dict(type='HandRandomFlip', flip_prob=0.5),
    dict(type='TopDownGetRandomScaleRotation', rot_factor=90, scale_factor=0.3),
    dict(type='TopDownAffine'),
    dict(type='ToTensorPose'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(
        type='MultitaskGatherTarget',
        pipeline_list=[
            [dict(
                type='Generate3DHeatmapTarget',
                sigma=2.5,
                max_bound=255,
            )], [dict(type='HandGenerateRelDepthTarget')],
            [
                dict(
                    type='RenameKeys',
                    key_pairs=[('hand_type', 'target'),
                               ('hand_type_valid', 'target_weight')])
            ]
        ],
        pipeline_indices=[0, 1, 2],
    ),

    dict(
        type='Collect',
        keys=['img', 'target', 'target_weight'],
        meta_keys=[
            'image_file', 'joints_3d', 'joints_3d_visible', 'center', 'scale',
            'rotation', 'flip_pairs'
        ]),
]

val_pipeline = [
    dict(type='LoadImageFromFileSimple'),
    dict(type='ToTensorPose'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=['image_file', 'center', 'scale', 'rotation', 'flip_pairs',
                   'heatmap3d_depth_bound', 'root_depth_bound', 'bbox_id', 'focal', 'princpt',
                   'joints_xyz']),
]

test_pipeline = val_pipeline

data_root = '/hdd0/data/rhd/RHD_published_v2'
data = dict(
    samples_per_gpu=32,
    workers_per_gpu=2,
    val_dataloader=dict(samples_per_gpu=16),
    test_dataloader=dict(samples_per_gpu=16),
    train=dict(
        type='Rhd3DDataset',
        ann_file=f'{data_root}/annotations/rhd_train.json',
        img_prefix=f'{data_root}/',
        data_cfg=data_cfg,
        pipeline=train_pipeline,
        dataset_info={{_base_.dataset_info}}),
    val=dict(
        type='Rhd3DDataset',
        ann_file=f'{data_root}/annotations/rhd_test.json',
        img_prefix=f'{data_root}/',
        data_cfg=data_cfg,
        pipeline=val_pipeline,
        dataset_info={{_base_.dataset_info}}),
    test=dict(
        type='Rhd3DDataset',
        ann_file=f'{data_root}/annotations/rhd_test.json',
        img_prefix=f'{data_root}/',
        data_cfg=data_cfg,
        pipeline=val_pipeline,
        dataset_info={{_base_.dataset_info}}),
)

