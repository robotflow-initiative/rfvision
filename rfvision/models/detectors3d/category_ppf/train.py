from rfvision.apis import train_detector
import rflib
from rfvision.models.builder import build_detector
from rfvision.datasets.builder import build_dataset

if __name__ == '__main__':
    for category in range(6):
        print(f'Training category{str(category)}')
        cfg = './ppf.py'
        # cfg.gpu_ids = [2]
        cfg = rflib.Config.fromfile(cfg)
        cfg.model.category = category
        cfg.work_dir = f'/home/hanyang/rfvision/work_dir/category_ppf/category{str(category)}'
        cfg.data.train.category = category

        model = build_detector(cfg.model)
        model.init_weights()
        dataset = build_dataset(cfg.data.train)
        train_detector(model=model,
                       dataset=dataset,
                       cfg=cfg,
                       distributed=False,
                       validate=False)