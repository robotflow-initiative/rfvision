import os
import os.path as osp
import warnings

import numpy as np
import torch
import torch.nn as nn


class WrapFunction(nn.Module):
    """Wrap the function to be tested for torch.onnx.export tracking."""

    def __init__(self, wrapped_function):
        super(WrapFunction, self).__init__()
        self.wrapped_function = wrapped_function

    def forward(self, *args, **kwargs):
        return self.wrapped_function(*args, **kwargs)


def convert_result_list(outputs):
    """Convert the torch forward outputs containing tuple or list to a list
    only containing torch.Tensor.

    Args:
        output (list(Tensor) | tuple(list(Tensor) | ...): the outputs
        in torch env, maybe containing nested structures such as list
        or tuple.

    Returns:
        list(Tensor): a list only containing torch.Tensor
    """
    # recursive end condition
    if isinstance(outputs, torch.Tensor):
        return [outputs]

    ret = []
    for sub in outputs:
        ret += convert_result_list(sub)
    return ret
