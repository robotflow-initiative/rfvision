# HandTailor

## Introduction
Refer to: HandTailor: Towards High-Precision Monocular 3D Hand Recovery  
Code: https://github.com/LyuJ1998/HandTailor  
Paper: https://arxiv.org/abs/2102.09244

## Dataset
### RHD Dataset
Download RHD Dataset from https://lmb.informatik.uni-freiburg.de/resources/datasets/RenderedHandposeDataset.en.html.  
Please download the annotation files from [rhd_annotations](https://download.openmmlab.com/mmpose/datasets/rhd_annotations.zip).
```
rhd
|── annotations
|   |── rhd_train.json
|   |── rhd_test.json
`── training
|   |── color
|   |   |── 00000.jpg
|   |   |── 00001.jpg
|   |── depth
|   |   |── 00000.jpg
|   |   |── 00001.jpg
|   |── mask
|   |   |── 00000.jpg
|   |   |── 00001.jpg
`── evaluation
|   |── color
|   |   |── 00000.jpg
|   |   |── 00001.jpg
|   |── depth
|   |   |── 00000.jpg
|   |   |── 00001.jpg
|   |── mask
|   |   |── 00000.jpg
|   |   |── 00001.jpg
```
###IK Dataset  
Download from [TODO]()

###MANO Model
1.Go to MANO website http://mano.is.tue.mpg.de/  
2.Create an account by clicking Sign Up and provide your information  
3.Download Models and Code (the downloaded file should have the format mano_v*_*.zip). Note that all code and data from this download falls under the MANO license.  
4.unzip and copy the MANO_RIGHT.pkl file into the folder 

## Train
Train hand-joints detector first:
```
cd rfvision
python flows/train_pipeline/train.py flows/human_analyzers/hand/rhd/res50_rhd3d_256x256.py
```

Then  
Train IKNet

```
cd rfvision
python flows/train_pipeline/train.py flows/human_analyzers/hand/others/iknet.py
```

## Reconstruction

```
cd rfvision
python rfvision/models/human_analyzers/handtailor/handtailor.py
```