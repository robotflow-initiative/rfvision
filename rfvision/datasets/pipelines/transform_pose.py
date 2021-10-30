from torchvision.transforms import functional as F
from rfvision.datasets.pipelines import Compose
from rfvision.datasets import PIPELINES


@PIPELINES.register_module()
class ToTensorPose():
    """Transform image to Tensor.

    Required key: 'img'. Modifies key: 'img'.

    Args:
        results (dict): contain all information about training.
    """

    def __call__(self, results):
        results['img'] = F.to_tensor(results['img'])
        return results


@PIPELINES.register_module()
class NormalizeTensor():
    """Normalize the Tensor image (CxHxW), with mean and std.

    Required key: 'img'. Modifies key: 'img'.

    Args:
        mean (list[float]): Mean values of 3 channels.
        std (list[float]): Std values of 3 channels.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, results):
        results['img'] = F.normalize(
            results['img'], mean=self.mean, std=self.std)
        return results

@PIPELINES.register_module()
class MultitaskGatherTarget:
    """Gather the targets for multitask heads.

    Args:
        pipeline_list (list[list]): List of pipelines for all heads.
        pipeline_indices (list[int]): Pipeline index of each head.
    """

    def __init__(self,
                 pipeline_list,
                 pipeline_indices=None,
                 keys=('target', 'target_weight')):
        self.keys = keys
        self.pipelines = []
        for pipeline in pipeline_list:
            self.pipelines.append(Compose(pipeline))
        if pipeline_indices is None:
            self.pipeline_indices = list(range(len(pipeline_list)))
        else:
            self.pipeline_indices = pipeline_indices

    def __call__(self, results):
        # generate target and target weights using all pipelines
        pipeline_outputs = []
        for pipeline in self.pipelines:
            pipeline_output = pipeline(results)
            pipeline_outputs.append(pipeline_output.copy())

        for key in self.keys:
            result_key = []
            for ind in self.pipeline_indices:
                result_key.append(pipeline_outputs[ind].get(key, None))
            results[key] = result_key
        return results