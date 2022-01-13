# RFVision

## Introduction
RFVision is an open-source toolbox for robot vision. It is a part of the [RobotFlow](https://wenqiangx.github.io/robotflowproject/) project.

The project takes a lot of design ideas from the great project [mmcv](https://github.com/open-mmlab/mmcv), [MMDetection](https://github.com/open-mmlab/mmdetection), [MMDetection3D](https://github.com/open-mmlab/mmdetection3d), but with significant modifications to adapt to robot vision tasks. For pure 2D or 3D vision tasks, we recommend to use MMDetection/3D as they support more baselines. 

But for robot vision where real-time visual servo, multi-modal perception are of more interest, RFVision is the project for you.

## License
This project is released under the [Apache 2.0 license](./LICENSE).

## Model Zoo
1. 2D Object Detection
   > Most bag of freebies are supported
   + [x] [Mask RCNN](flows/detectors/mask_rcnn/README.md)
   + [x] [HTC](flows/detectors/htc/README.md)
   + [x] [DCT-Mask](flows/detectors/dct_mask/README.md)
   + [x] [FCOS](flows/detectors/fcos/README.md)
   + [x] [SOLOv2](flows/detectors/solov2/)
   + [x] [YOLOX](flows/detectors/yolox/README.md)
   + [x] [USD-Seg](flows/detectors/yolo/README.md)
2. 3D Object Detection
   + [x] [VoteNet](flows/detectors3d/votenet/README.md)
   + [x] [ImVoteNet](flows/detectors3d/imvotenet/README.md)
3. Object Pose Estimation
   + [x] [SkeletonMerger](flows/detectors3d/skeleton_merger/README.md)
   + [x] [ArticualtedPoseEstimation](flows/pose_estimators/articulation/README.md)
   + [x] [PointFormer](flows/detectors3d/pointformer/README.md) 
   + [ ] OMAD
4. Human Analyzers
   + [x] [HMR](flows/human_analyzers/body/3d_mesh_sview_rgb_img/README.md)
   + [x] [SimpleBaseline3D](flows/human_analyzers/body/3d_kpt_sview_rgb_img/README.md)
   + [x] [InterNet](flows/human_analyzers/hand/interhand3d/README.md)
   + [x] [HandTailor](rfvision/models/human_analyzers/handtailor/README.md)
   + [ ] CPF (inference only, for training material, you may contact to get rfvision_restricted project)
5. Multi-modality
   + [x] [3D vision and touch for shape reconstruction](rfvision/models/detectors3d/touch_and_vision/README.md)

## Installation
Please refer to [get_started.md](docs/get_started.md) for installation

## Contributing
We appreciate all contributions to improve RFVision. Please refer to [CONTRIBUTING.md](docs/contributing.md) for the contributing guideline.

## Ackowledgement
We borrow many design from projects in OpenMMLab, we appreciate their efforts on lowering the research bar of computer vision.

When the era of robot learning comes, we wish our toolbox and benchmark could serve the growing research community by providing a flexible toolkit to reimplement existing methods and develop their own new methods.

## Citation
If you use this toolbox or benchmark in your research, please cite this project.
```
```

## Projects in RobotFlow
### Software
+ [RFLib](https://github.com/mvig-robotflow/rflib): RobotFlow foundational library for Robot Vision, Planning and Control.
+ [RFVision](https://github.com/mvig-robotflow/rfvision): RobotFlow vision-related toolbox and benchmark.
+ [RFMove](https://github.com/mvig-robotflow/rfmove): RobotFlow planning toolbox.
+ [ReinForce](https://github.com/mvig-robotflow/ReinForce): RobotFlow reinforcement learning toolbox.
+ [RFController](https://github.com/mvig-robotflow/rfcontroller): RobotFlow controller toolbox.
+ [rFUniverse](https://github.com/mvig-robotflow/rfuniverse): A Unity-based Multi-purpose Simulation Environment.
+ [RFBulletT](https://github.com/mvig-robotflow/rfbullett): A Pybullet-based Multi-purpose Simulation Environment.
+ [RF_ROS](https://github.com/mvig-robotflow/rf_ros): ROS integration. Both ROS1 and ROS2 are supported.
+ [RobotFlow](https://github.com/mvig-robotflow/robotflow): The barebone of the whole system. It organizes all the functionalities.
### Hardware
+ [RFDigit](https://github.com/mvig-robotflow/rfdigit): A Customized Digit Tactile Sensor.
### Open Ecosystem
+ [N-D Pose Annotator](https://github.com/liuliu66/6DPoseAnnotator): support both rigid and articulated object pose annotation.
+ [model format converter](https://github.com/mvig-robotflow/model_format_converter): URDF and related model format converter.
