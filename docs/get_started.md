## Requirements
> lower version may also work, we just have not test it yet.
> Some ops are required by torchvision.ops, please make sure the pytorch version is compatible.
+ Python >= 3.6
+ Ubuntu >= 18.04
+ CUDA >= 10.2
+ Pytorch >= 1.6

## Installation
1. `install CUDA`. If you install CUDA 10.2 and meet the gcc version error, add `--override` at the end of `sudo sh xxx.run`.
2. `install pytorch`. Install pytorch with correct cuda version. Please follow the [official guidance](https://pytorch.org/get-started/previous-versions/). Since the RFLib is compatible to different CUDA versions, it largely relieves the burden of inner compatibility.
3. `install requirements`.
    + if you want 2D/tactile-related perception algorithm only
    ```
    pip install -r requirements/2d.txt
    ```
    + if you want 2.5D/3D-related perception only
    ```
    pip install -r requirements/3d.txt
    ```
    + if you want a full install
    ```
    pip install -r requirements/full.txt
    ```
