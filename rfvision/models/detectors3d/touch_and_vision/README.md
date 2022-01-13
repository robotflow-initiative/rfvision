# 3D-Vision-and-Touch

## Introduction
Refer to: Skeleton Merger, an Unsupervised Aligned Keypoint Detector    
Code: https://github.com/facebookresearch/3D-Vision-and-Touch  
Paper: https://arxiv.org/abs/2007.03778

## Dataset
Download from https://dl.fbaipublicfiles.com/ABC/data.tar.gz  
then 
```
tar xf data.tar.gz -C data --strip-components=1
```

## Train
Touch
```
cd rfvision
python flows/train_pipeline/train.py rfvision/models/detectors3d/touch_and_vision/configs/touch_cfg.py
```

Vision
```
cd rfvision
rfvision/models/detectors3d/touch_and_vision/configs/vision_cfg.py
```
