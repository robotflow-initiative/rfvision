from rfvision.datasets import PIPELINES
from rfvision.components.utils import normalize_point_cloud
@PIPELINES.register_module()
class NormalizePoints:
    def __call__(self, results):
        points = results['points']
        pc_normalized, centroid, m = normalize_point_cloud(points)

        if 'keypoints_xyz' in results:
            keypoints_xyz = results['keypoints_xyz']
            keypoints_xyz_normalized = (keypoints_xyz - centroid) / m
            results['keypoints_xyz'] = keypoints_xyz_normalized
        return results
