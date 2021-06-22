import torch

def get_k_for_topk(k, size):
    """Get k of TopK for onnx exporting.

    The K of TopK in TensorRT should not be a Tensor, while in ONNX Runtime
      it could be a Tensor.Due to dynamic shape feature, we have to decide
      whether to do TopK and what K it should be while exporting to ONNX.
    If returned K is less than zero, it means we do not have to do
      TopK operation.

    Args:
        k (int or Tensor): The set k value for nms from config file.
        size (Tensor or torch.Size): The number of elements of \
            TopK's input tensor
    Returns:
        tuple: (int or Tensor): The final K for TopK.
    """
    ret_k = -1
    if k <= 0 or size <= 0:
        return ret_k
    if k < size:
        ret_k = k
    else:
        # ret_k is -1
        pass
    return ret_k