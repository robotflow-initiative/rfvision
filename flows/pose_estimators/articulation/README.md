# Category-Level Articulated Object Pose Estimation
## Introduction
Refer to: Category-Level Articulated Object Pose Estimation  
Code: https://github.com/liuliu66/articulation_estimator_slim  
Paper: http://cn.arxiv.org/abs/1912.11913

## Dataset
Download from [TODO]().

## Training
There is no validation function in this model, therefore --no-validate is needed for training.  
Train with single gpu:
```
cd rfvision
python flows/train_pipeline/train.py flows/pose_estimators/articulation/articulation_estimator.py --no-validate
```
or

Train with multiple gpus:

```
cd rfvision
bash flows/train_pipeline/dist_train.sh flows/pose_estimators/articulation/articulation_estimator.py 4 --no-validate
```
## Visualization

Download demo files from https://github.com/liuliu66/articulation_estimator_slim/tree/main/demo to 
rfvision/models/pose_estimators/articulation/demo

Download the checkpoints file from [TODO]() to rfvision/models/pose_estimators/articulation/checkpoints/checkpoints.pth

Then
```
python visualization.py checkpoints/checkpoints.pth demo demo/det_bboxes.json
```




## Results and models

|    Backbone     |  Batch Size | GPU num| Mem (GB) | Loss | Download |
| :-------------: | :-----: | :------: |:------:| :-----: |:--------: |
|    PointNet++     |  64    | 2 | 0.38 |  2.3549  | [TODO]() |