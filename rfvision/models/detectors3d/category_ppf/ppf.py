checkpoint_config = dict(interval=1)
log_config = dict(
    interval=50,
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


tr_ranges = {
    1: [0.25, 0.25],
    2: [0.12, 0.12],
    3: [0.15, 0.15],
    4: [0.1, 0.1],
    5: [0.3, 0.3],
    6: [0.12, 0.12]
}

scale_ranges = {
    1: [0.05, 0.15, 0.05],
    2: [0.07, 0.03, 0.07],
    3: [0.05, 0.05, 0.07],
    4: [0.037, 0.055, 0.037],
    5: [0.13, 0.1, 0.15],
    6: [0.06, 0.05, 0.045]
}



model = dict(type='CategoryPPF',
             scale_ranges=scale_ranges,
             tr_ranges=tr_ranges)
train_dataset = 'ShapeNetDatasetForPPF'
test_dataset = 'NOCSForPPF'

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=0,
    train=dict(type=train_dataset,
               data_root='/disk1/data/ShapeNetCore.v2/',
               ann_file='/disk1/data/ppf_dataset/shapenet_train.txt'),
    val=dict(type=test_dataset,
             data_root='/disk1/data/ppf_dataset/',
             pipeline=[dict(type='Collect',
                            keys=['pcs', 'pc_normals'],
                            meta_keys=['RT', 'gt_pc'])]),
    test=dict(type=test_dataset,
              data_root='/disk1/data/ppf_dataset/',
              pipeline=[dict(type='Collect',
                             keys=['pcs', 'pc_normals'],
                             meta_keys=['RT', 'gt_pc'])]))
