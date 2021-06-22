import rflib

def coco_classes():
    return [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
        'truck', 'boat', 'traffic_light', 'fire_hydrant', 'stop_sign',
        'parking_meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
        'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
        'sports_ball', 'kite', 'baseball_bat', 'baseball_glove', 'skateboard',
        'surfboard', 'tennis_racket', 'bottle', 'wine_glass', 'cup', 'fork',
        'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot_dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted_plant', 'bed', 'dining_table', 'toilet', 'tv',
        'laptop', 'mouse', 'remote', 'keyboard', 'cell_phone', 'microwave',
        'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
        'scissors', 'teddy_bear', 'hair_drier', 'toothbrush'
    ]


dataset_aliases = {
    'coco': ['coco', 'mscoco', 'ms_coco'],
}


def get_classes(dataset):
    """Get class names of a dataset."""
    alias2name = {}
    for name, aliases in dataset_aliases.items():
        for alias in aliases:
            alias2name[alias] = name

    if rflib.is_str(dataset):
        if dataset in alias2name:
            labels = eval(alias2name[dataset] + '_classes()')
        else:
            raise ValueError(f'Unrecognized dataset: {dataset}')
    else:
        raise TypeError(f'dataset must a str, but got {type(dataset)}')
    return labels
