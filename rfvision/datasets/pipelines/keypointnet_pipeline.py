from robotflow.rflearner.datasets import PIPELINES

@PIPELINES.register_module()
class NormalizePoints:

    def __call__(self, results):
        points = results['points']
        pc_min = points.min()
        pc_max = points.max()
        points_normalized = (points - pc_min) / (pc_max - pc_min)
        results['points'] = points_normalized

        if 'keypoints_xyz' in results:
            keypoints_xyz = results['keypoints_xyz']
            keypoints_xyz_normalized = (keypoints_xyz - pc_min) / (pc_max - pc_min)
            results['keypoints_xyz'] = keypoints_xyz_normalized

        if 'vertices' in results:
            vertices = results['vertices']
            vertices_min = vertices.min()
            vertices_max = vertices.max()
            vertices_normalized = (vertices - vertices_min) / (vertices_max - vertices_min)
            results['vertices'] = vertices_normalized
        return results
