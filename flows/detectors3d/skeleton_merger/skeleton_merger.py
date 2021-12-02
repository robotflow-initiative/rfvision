_base_ = '../_base_/default_runtime.py'

n_keypoint = 10
# model settings
model = dict(
    type='SkeletonMerger',
    backbone=dict(type='PointNet2ForSkeletonMerger',
                  n_keypoint=n_keypoint,
                  ),
    head=dict(type='SkeletonMergerHead',
              n_keypoint=n_keypoint))

# dataset settings
train_dataset_type = 'ShapeNetCoreV2ForSkeletonMerger'
test_dataset_type ='KeypointNetDataset'
train_pipeline = [
    dict(type='IndoorPointSample', num_points=256),
    dict(type='NormalizePoints'),
    dict(type='ToTensor', keys=['points']),
    dict(type='Collect3D', keys=['points']),
]
test_pipeline = [
    dict(type='NormalizePoints'),
    dict(type='ToTensor', keys=['points', 'colors', 'keypoints_xyz', 'keypoints_semantic_id',
                                'keypoints_index']),
    dict(type='Collect3D', keys=['points', 'colors', 'keypoints_xyz', 'keypoints_semantic_id',
                                 'keypoints_index'],
                           meta_keys=['class_id', 'model_id']),
]

data = dict(
    samples_per_gpu=64,
    workers_per_gpu=2,
    train=dict(
        type=train_dataset_type,
        data_root='/disk1/data/skeleton_merger/shapenetcorev2_hdf5_2048',
        pipeline=train_pipeline,
        split='train',
        ),

    val=dict(
        type=test_dataset_type,
        data_root='/disk1/data/skeleton_merger/keypointnet/ShapeNetCore.v2',
        split='val',
        split_file_root = '/home/hanyang/robotflow/robotflow/rflearner/data/keypointnet',
        classes=['chair'],
        pipeline=test_pipeline,
        test_mode=True,
        ),
    test=dict(
        type=test_dataset_type,
        data_root='/disk1/data/skeleton_merger/keypointnet/ShapeNetCore.v2',
        split='val',
        split_file_root = '/home/hanyang/robotflow/robotflow/rflearner/data/keypointnet',
        classes=['chair'],
        pipeline=test_pipeline,
        test_mode=True,
    ))

# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup=None, 
    step=[50,65])
# runtime settings
total_epochs = 80
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
checkpoint_config = dict(interval=40)
find_unused_parameters = True