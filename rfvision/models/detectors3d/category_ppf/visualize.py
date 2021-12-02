def visualize(vis, *pcs, **opts):
    vis_pc = np.concatenate(pcs)
    vis_label = np.ones((sum([p.shape[0] for p in pcs])), np.int64)
    a = 0
    for i, pc in enumerate(pcs):
        vis_label[a:a + pc.shape[0]] = i + 1
        a += pc.shape[0]
    vis.scatter(vis_pc, vis_label, **opts)


def validation(self, vertices, outputs, probs, res, point_idxs, n_ppfs, num_rots=36, visualize=False):
    with cp.cuda.Device(0):
        block_size = (vertices.shape[0] ** 2 + 512 - 1) // 512

        corners = np.stack([np.min(vertices, 0), np.max(vertices, 0)])
        grid_res = ((corners[1] - corners[0]) / res).astype(np.int32) + 1
        grid_obj = cp.asarray(np.zeros(grid_res, dtype=np.float32))
        ppf_kernel(
            (block_size, 1, 1),
            (512, 1, 1),
            (
                cp.asarray(vertices).astype(cp.float32), cp.asarray(outputs).astype(cp.float32),
                cp.asarray(probs).astype(cp.float32), cp.asarray(point_idxs).astype(cp.int32), grid_obj,
                cp.asarray(corners[0]), cp.float32(res),
                n_ppfs, num_rots, grid_obj.shape[0], grid_obj.shape[1], grid_obj.shape[2]
            )
        )

        grid_obj = grid_obj.get()

        # cand = np.array(np.unravel_index([np.argmax(grid_obj, axis=None)], grid_obj.shape)).T[::-1]
        # grid_obj[cand[-1][0]-20:cand[-1][0]+20, cand[-1][1]-20:cand[-1][1]+20, cand[-1][2]-20:cand[-1][2]+20] = 0
        if visualize:
            self.vis.heatmap(cv2.rotate(grid_obj.max(0), cv2.ROTATE_90_COUNTERCLOCKWISE), win=3,
                             opts=dict(title='front'))
            self.vis.heatmap(cv2.rotate(grid_obj.max(1), cv2.ROTATE_90_COUNTERCLOCKWISE), win=4,
                             opts=dict(title='bird'))
            self.vis.heatmap(cv2.rotate(grid_obj.max(2), cv2.ROTATE_90_COUNTERCLOCKWISE), win=5,
                             opts=dict(title='side'))

        cand = np.array(np.unravel_index([np.argmax(grid_obj, axis=None)], grid_obj.shape)).T[::-1]
        cand_world = corners[0] + cand * res
        # print(cand_world[-1])
        return grid_obj, cand_world