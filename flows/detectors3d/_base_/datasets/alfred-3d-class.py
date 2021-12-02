# dataset settings
dataset_type = 'AlfredDataset'
data_root = './json_2.1.0/'
class_names = (
    'Apple',
    'Bowl',
    'Bread',
    'ButterKnife',
    'Cup',
    'DishSponge',
    'Egg',
    'Fork',
    'Knife',
    'Ladle',
    'Lettuce',
    'Mug',
    'Pan',
    'PepperShaker',
    'Plate',
    'Potato',
    'Pot',
    'SaltShaker',
    'SoapBottle',
    'Spatula',
    'Spoon',
    'Tomato',
    'Box',
    'CreditCard',
    'KeyChain',
    'Laptop',
    'Pillow',
    'RemoteControl',
    'Statue',
    'Vase',
    'Candle',
    'Cloth',
    'HandTowel',
    'Plunger',
    'ScrubBrush',
    'SoapBar',
    'SprayBottle',
    'ToiletPaper',
    'Towel',
    'Newspaper',
    'Watch',
    'Book',
    'CellPhone',
    'WateringCan',
    'Glassbottle',
    'PaperTowelRoll',
    'WineBottle',
    'Pencil',
    'Kettle',
    'Boots',
    'TissueBox',
    'Pen',
    'AlarmClock',
    'BasketBall',
    'CD',
    'TeddyBear',
    'TennisRacket',
    'BaseballBat',
    'Footstool'
)

train_pipeline = [
    dict(
        type='LoadPointsFromFilePointFormer',
        coord_type='DEPTH',
        shift_height=True,
        use_color=True,
        load_dim=6,
        use_dim=6),
    dict(type='LoadAnnotations3D'),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
    ),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.523599, 0.523599],
        scale_ratio_range=[0.85, 1.15],
        shift_height=True),
    dict(type='PointSample', num_points=20000),
    dict(type='DefaultFormatBundle3D', class_names=class_names,
         with_label=False),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]

test_pipeline = [
    dict(
        type='LoadPointsFromFilePointFormer',
        coord_type='DEPTH',
        shift_height=True,
        use_color=True,
        load_dim=6,
        use_dim=6),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(
                type='RandomFlip3D',
                sync_2d=False,
                flip_ratio_bev_horizontal=0.5,
            ),
            dict(type='PointSample', num_points=20000),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ])
]

# construct a pipeline for data and gt loading in show function
# please keep its loading function consistent with test_pipeline (e.g. client)
eval_pipeline = [
    dict(
        type='LoadPointsFromFilePointFormer',
        coord_type='DEPTH',
        shift_height=True,
        load_dim=6,
        use_dim=6),
    dict(
        type='DefaultFormatBundle3D',
        class_names=class_names,
        with_label=False),
    dict(type='Collect3D', keys=['points'])
]


data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=5,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root + 'alfred_infos_train.pkl',
            pipeline=train_pipeline,
            filter_empty_gt=False,
            classes=class_names,
            # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
            # and box_type_3d='Depth' in sunrgbd and alfred dataset.
            box_type_3d='Depth')
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'alfred_infos_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        test_mode=True,
        box_type_3d='Depth'
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'alfred_infos_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        test_mode=True,
        box_type_3d='Depth')
)

evaluation = dict(pipeline=eval_pipeline)
