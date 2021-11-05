import torch
import torch.nn as nn
import torch.nn.functional as F
from rfvision.models.builder import DETECTORS
from rfvision.models.human_analyzers import BasePose
import numpy as np
import cupy as cp
import visdom as vis
from .voting import backvote_kernel, rot_voting_kernel
from .utils import validation, visualize, fibonacci_sphere


@DETECTORS.register_module()
class CategoryPPF(BasePose):
    def __init__(self,
                 tr_ranges,
                 scale_ranges,
                 category=2,
                 knn=60,
                 tr_num_bins=32,
                 rot_num_bins=36,
                 regress_right=True,
                 init_cfg=None,
                 train_cfg=None,
                 test_cfg=None):
        super().__init__(init_cfg=init_cfg)

        ############## para init ####################
        self.knn = knn
        self.tr_num_bins = tr_num_bins
        self.rot_num_bins = rot_num_bins
        self.regress_right = regress_right
        self.tr_ranges = tr_ranges
        self.scale_ranges = scale_ranges
        self.num_rots = 72
        self.n_threads = 512
        self.res = 0.005
        self.category = category
        ############### model init ###################
        self.point_encoder = PointEncoderRaw(k=self.knn, spfcs=[32, 64, 32, 32], num_layers=1, out_dim=32)
        self.ppf_encoder = PPFEncoder(ppffcs=[84, 32, 32, 16],
                                      out_dim=2 * self.tr_num_bins + 2 * self.rot_num_bins + 2 + 3)

        ################ loss init ####################
        self.kldiv = nn.KLDivLoss(reduction='batchmean')
        self.bcelogits = nn.BCEWithLogitsLoss()
        # self.loss_right_meter = AverageMeter()
        # self.loss_right_aux_meter = AverageMeter()


    def forward_train(self,
                      pcs,
                      pc_normals,
                      targets_tr,
                      targets_rot,
                      targets_rot_aux,
                      targets_scale,
                      point_idxs):

        with torch.no_grad():
            dist = torch.cdist(pcs, pcs)

        sprin_feat = self.point_encoder(pcs, pc_normals, dist)

        preds = self.ppf_encoder(pcs, pc_normals, sprin_feat, idxs=point_idxs[0])

        preds_tr = preds[..., :2 * self.tr_num_bins].reshape(-1, 2, self.tr_num_bins)
        preds_up = preds[..., 2 * self.tr_num_bins:2 * self.tr_num_bins + self.rot_num_bins]
        preds_right = preds[..., 2 * self.tr_num_bins + self.rot_num_bins:2 * self.tr_num_bins + 2 * self.rot_num_bins]

        preds_up_aux = preds[..., -5]
        preds_right_aux = preds[..., -4]

        preds_scale = preds[..., -3:]

        losses = {}
        loss_tr = self.kldiv(F.log_softmax(preds_tr[:, 0], dim=-1), targets_tr[0, :, 0]) + self.kldiv(
            F.log_softmax(preds_tr[:, 1], dim=-1), targets_tr[0, :, 1])
        loss_up = self.kldiv(F.log_softmax(preds_up[0], dim=-1), targets_rot[0, :, 0])
        loss_up_aux = self.bcelogits(preds_up_aux[0], targets_rot_aux[0, :, 0])
        loss_scale = F.mse_loss(preds_scale, targets_scale[:, None])

        losses['loss_tr'] = loss_tr
        losses['loss_up'] = loss_up
        losses['loss_up_aux'] = loss_up_aux
        losses['loss_scale'] = loss_scale


        # loss = loss_up + loss_tr + loss_up_aux + loss_scale
        if self.regress_right:
            loss_right = self.kldiv(F.log_softmax(preds_right[0], dim=-1), targets_rot[0, :, 1])
            loss_right_aux = self.bcelogits(preds_right_aux[0], targets_rot_aux[0, :, 1])
            losses['loss_right'] = loss_right
            losses['loss_right_aux'] = loss_right_aux
        return losses

    def forward_test(self,
                     pcs,
                     pc_normals,
                     img_metas
                     ):
        point_idxs = torch.randint(0, pcs.shape[1], (1000000, 2))
        RT = img_metas[0]['RT']
        gt_pc = img_metas[0]['gt_pc']

        with torch.no_grad():
            dist = torch.cdist(pcs, pcs)
            sprin_feat = self.point_encoder(pcs, pc_normals, dist)
            preds = self.ppf_encoder(pcs, pc_normals, sprin_feat, idxs=point_idxs)

        preds_tr = torch.softmax(preds[..., :2 * self.tr_num_bins].reshape(-1, 2, self.tr_num_bins), -1)
        preds_tr = torch.cat([torch.multinomial(preds_tr[:, 0], 1), torch.multinomial(preds_tr[:, 1], 1)], -1).float()[None]
        preds_tr[0, :, 0] = preds_tr[0, :, 0] / (self.tr_num_bins - 1) * 2 * self.tr_ranges[self.category][0] - self.tr_ranges[self.category][0]
        preds_tr[0, :, 1] = preds_tr[0, :, 1] / (self.tr_num_bins - 1) * self.tr_ranges[self.category][1]

        pc = pcs[0].cpu().numpy()
        grid_obj, candidates = validation(pc, preds_tr[0].cpu().numpy(), np.ones((pc.shape[0],)), self.res, point_idxs,
                                          point_idxs.shape[0], self.num_rots)

        corners = np.stack([np.min(pc, 0), np.max(pc, 0)])
        T_est = candidates[-1]
        T_gt = img_metas[0][''][:3, -1]
        T_err_sp = np.linalg.norm(T_est - T_gt)
        print('pred translation error: ', T_err_sp)

        # back vote filtering
        block_size = (point_idxs.shape[0] + self.n_threads - 1) // self.n_threads

        pred_center = T_est
        with cp.cuda.Device(0):
            output_ocs = cp.zeros((point_idxs.shape[0], 3), cp.float32)
            backvote_kernel(
                (block_size, 1, 1),
                (self.n_threads, 1, 1),
                (
                    cp.asarray(pc), cp.asarray(preds_tr[0].cpu().numpy()), output_ocs,
                    cp.asarray(point_idxs).astype(cp.int32), cp.asarray(corners[0]), cp.float32(self.res),
                    point_idxs.shape[0], self.num_rots, grid_obj.shape[0], grid_obj.shape[1], grid_obj.shape[2],
                    cp.asarray(pred_center).astype(cp.float32), cp.float32(3 * self.res)
                )
            )
        oc = output_ocs.get()
        mask = np.any(oc != 0, -1)
        point_idxs = point_idxs[mask]

        # unsupervised segmentation
        pc_idxs = np.array(list(set(list(point_idxs.reshape(-1)))), np.int64)
        contrib_cnt = (point_idxs.reshape(-1, 1) == pc_idxs[None]).sum(0)
        pc_idxs = pc_idxs[contrib_cnt > 175]
        mask = np.zeros((pc.shape[0],), bool)
        mask[pc_idxs] = True
        visualize(vis, pc[~mask], pc[mask], win=10, opts=dict(markersize=3))

        # # rotation vote
        angle_tol = 2
        num_samples = int(4 * np.pi / (angle_tol / 180 * np.pi))
        sphere_pts = np.array(fibonacci_sphere(num_samples))
        with torch.no_grad():
            preds = self.ppf_encoder(pcs, pc_normals, sprin_feat, idxs=point_idxs)

            preds_up = preds[..., 2 * self.tr_num_bins:2 * self.tr_num_bins + self.rot_num_bins]
            preds_right = preds[..., 2 * self.tr_num_bins + self.rot_num_bins:2 * self.tr_num_bins + 2 * self.rot_num_bins]
            preds_up_aux = preds[..., -5]
            preds_right_aux = preds[..., -4]
            preds_scale = preds[..., -3:]

            preds_tr = torch.softmax(preds[..., :2 * self.tr_num_bins].reshape(-1, 2, self.tr_num_bins), -1)
            preds_tr = torch.cat([torch.multinomial(preds_tr[:, 0], 1), torch.multinomial(preds_tr[:, 1], 1)], -1).float()[None]
            preds_tr[0, :, 0] = preds_tr[0, :, 0] / (self.tr_num_bins - 1) * 2 * self.tr_ranges[self.category][0] - \
                                self.tr_ranges[self.category][0]
            preds_tr[0, :, 1] = preds_tr[0, :, 1] / (self.tr_num_bins - 1) * self.tr_ranges[self.category][1]

            preds_up = torch.softmax(preds_up[0], -1)
            preds_up = torch.multinomial(preds_up, 1).float()[None]
            preds_up[0] = preds_up[0] / (self.rot_num_bins - 1) * np.pi

            preds_right = torch.softmax(preds_right[0], -1)
            preds_right = torch.multinomial(preds_right, 1).float()[None]
            preds_right[0] = preds_right[0] / (self.rot_num_bins - 1) * np.pi

        final_directions = []
        for direction, aux in zip([preds_up, preds_right], [preds_up_aux, preds_right_aux]):
            with cp.cuda.Device(0):
                candidates = cp.zeros((point_idxs.shape[0], self.num_rots, 3), cp.float32)

                block_size = (point_idxs.shape[0] + 512 - 1) // 512
                rot_voting_kernel(
                    (block_size, 1, 1),
                    (512, 1, 1),
                    (
                        cp.asarray(pc), cp.asarray(preds_tr[0].cpu().numpy()), cp.asarray(direction[0].cpu().numpy()),
                        candidates, cp.asarray(point_idxs).astype(cp.int32), cp.asarray(corners[0]).astype(cp.float32),
                        cp.float32(self.res),
                        point_idxs.shape[0],self.num_rots, grid_obj.shape[0], grid_obj.shape[1], grid_obj.shape[2]
                    )
                )
            candidates = candidates.get().reshape(-1, 3)
            start = np.arange(0, point_idxs.shape[0] * self.num_rots, self.num_rots)
            np.random.shuffle(start)
            sub_sample_idx = (start[:10000, None] + np.arange(self.num_rots)[None]).reshape(-1)
            candidates = candidates[sub_sample_idx]

            cos = np.matmul(candidates, sphere_pts.T)
            counts = np.sum(cos > np.cos(angle_tol / 180 * np.pi), 0)
            best_dir = np.array(sphere_pts[np.argmax(counts)])

            # filter up
            ab = pc[point_idxs[:, 0]] - pc[point_idxs[:, 1]]
            distsq = np.sum(ab ** 2, -1)
            ab_normed = ab / (np.sqrt(distsq) + 1e-7)[..., None]
            pc_normal = pc_normals.cpu.numpy()
            pairwise_normals = pc_normal[point_idxs[:, 0]]
            pairwise_normals[np.sum(pairwise_normals * ab_normed, -1) < 0] *= -1

            with torch.no_grad():
                target = torch.from_numpy((np.sum(pairwise_normals * best_dir, -1) > 0).astype(np.float32)).cuda()
                up_loss = self.bcelogits(aux[0], target).item()
                down_loss = self.bcelogits(aux[0], 1. - target).item()

            # print(up_loss, down_loss)
            if down_loss < up_loss:
                final_dir = -best_dir
            else:
                final_dir = best_dir
            final_directions.append(final_dir)

        up = final_directions[0]
        right = final_directions[1]
        right -= np.dot(up, right) * up
        right /= np.linalg.norm(right)

        if self.z_right:
            R_est = np.stack([np.cross(up, right), up, right], -1)
        else:
            R_est = np.stack([right, up, np.cross(right, up)], -1)
        if self.regress_right:
            rot_err = np.arccos((np.trace(RT[:3, :3].T @ R_est) - 1.) / 2.) / np.pi * 180
            print('pred up error: ', np.arccos(np.dot(up, RT[:3, 1])) / np.pi * 180)
        else:
            rot_err = np.arccos(np.dot(up, RT[:3, 1])) / np.pi * 180
        print('pred rotation error: ', rot_err)

        retrieved_pc = gt_pc @ R_est.T + T_est
        visualize(vis, pc, retrieved_pc, win=8, opts=dict(markersize=3))

        print('pred scale: ', np.exp(preds_scale[0].mean(0).cpu().numpy()) * self.scale_ranges[self.category])
        print('gt scale', gt_pc.max(0))



    def forward(self, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)

    def aug_test(self):
        pass
    def extract_feat(self):
        pass
    def simple_test(self):
        pass
    def show_result(self, **kwargs):
        pass


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ResLayer(torch.nn.Module):
    def __init__(self, dim_in, dim_out, bn=False) -> None:
        super().__init__()
        assert (bn is False)
        self.fc1 = torch.nn.Linear(dim_in, dim_out)
        if bn:
            self.bn1 = torch.nn.BatchNorm1d(dim_out)
        else:
            self.bn1 = lambda x: x
        self.fc2 = torch.nn.Linear(dim_out, dim_out)
        if bn:
            self.bn2 = torch.nn.BatchNorm1d(dim_out)
        else:
            self.bn2 = lambda x: x
        if dim_in != dim_out:
            self.fc0 = torch.nn.Linear(dim_in, dim_out)
        else:
            self.fc0 = None

    def forward(self, x):
        x_res = x if self.fc0 is None else self.fc0(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.bn2(self.fc2(x))
        return x + x_res


def conv_kernel(iunit, ounit, *hunits):
    layers = []
    for unit in hunits:
        layers.append(nn.Linear(iunit, unit))
        layers.append(nn.LayerNorm(unit))
        layers.append(nn.ReLU())
        iunit = unit
    layers.append(nn.Linear(iunit, ounit))
    return nn.Sequential(*layers)


def rifeat(points_r, points_s):
    """generate rotation invariant features
    Args:
        points_r (B x N x K x 3):
        points_s (B x N x 1 x 3):
    """

    # [*, 3] -> [*, 8] with compatible intra-shapes
    if points_r.shape[1] != points_s.shape[1]:
        points_r = points_r.expand(-1, points_s.shape[1], -1, -1)

    r_mean = torch.mean(points_r, -2, keepdim=True)
    l1, l2, l3 = r_mean - points_r, points_r - points_s, points_s - r_mean
    l1_norm = torch.norm(l1, 'fro', -1, True)
    l2_norm = torch.norm(l2, 'fro', -1, True)
    l3_norm = torch.norm(l3, 'fro', -1, True).expand_as(l2_norm)
    theta1 = (l1 * l2).sum(-1, keepdim=True) / (l1_norm * l2_norm + 1e-7)
    theta2 = (l2 * l3).sum(-1, keepdim=True) / (l2_norm * l3_norm + 1e-7)
    theta3 = (l3 * l1).sum(-1, keepdim=True) / (l3_norm * l1_norm + 1e-7)

    return torch.cat([l1_norm, l2_norm, l3_norm, theta1, theta2, theta3], dim=-1)


class SparseSO3Conv(nn.Module):
    def __init__(self, rank, n_in, n_out, *kernel_interns, layer_norm=True):
        super().__init__()
        self.kernel = conv_kernel(6, rank, *kernel_interns)
        self.outnet = nn.Linear(rank * n_in, n_out)
        self.rank = rank
        self.layer_norm = nn.LayerNorm(n_out) if layer_norm else None

    def do_conv_ranked(self, r_inv_s, feat):
        # [b, n, k, rank], [b, n, k, cin] -> [b, n, cout]
        kern = self.kernel(r_inv_s).reshape(*feat.shape[:-1], self.rank)
        # PointConv-like optimization
        contracted = torch.einsum("bnkr,bnki->bnri", kern, feat).flatten(-2)
        return self.outnet(contracted)

    def forward(self, feat_points, feat, eval_points):
        eval_points_e = torch.unsqueeze(eval_points, -2)
        r_inv_s = rifeat(feat_points, eval_points_e)
        conv = self.do_conv_ranked(r_inv_s, feat)
        if self.layer_norm is not None:
            return self.layer_norm(conv)
        return conv


class GlobalInfoProp(nn.Module):
    def __init__(self, n_in, n_global):
        super().__init__()
        self.linear = nn.Linear(n_in, n_global)

    def forward(self, feat):
        # [b, k, n_in] -> [b, k, n_in + n_global]
        tran = self.linear(feat)
        glob = tran.max(-2, keepdim=True)[0].expand(*feat.shape[:-1], tran.shape[-1])
        return torch.cat([feat, glob], -1)


class PointEncoderRaw(nn.Module):
    def __init__(self, k, spfcs, out_dim, num_layers=2, num_nbr_feats=2) -> None:
        super().__init__()
        self.k = k
        self.spconvs = nn.ModuleList()
        self.spconvs.append(SparseSO3Conv(32, num_nbr_feats, out_dim, *spfcs))
        self.aggrs = nn.ModuleList()
        self.aggrs.append(GlobalInfoProp(out_dim, out_dim // 4))
        for _ in range(num_layers - 1):
            self.spconvs.append(SparseSO3Conv(32, out_dim + out_dim // 4, out_dim, *spfcs))
            self.aggrs.append(GlobalInfoProp(out_dim, out_dim // 4))

    def forward(self, pc, pc_normal, dist):
        nbrs_idx = torch.topk(dist, self.k, largest=False, sorted=False)[1]  # [..., N, K]
        pc_nbrs = torch.gather(pc.unsqueeze(-3).expand(*pc.shape[:-1], *pc.shape[-2:]), -2,
                               nbrs_idx[..., None].expand(*nbrs_idx.shape, pc.shape[-1]))  # [..., N, K, 3]
        pc_nbrs_centered = pc_nbrs - pc.unsqueeze(-2)  # [..., N, K, 3]
        pc_nbrs_norm = torch.norm(pc_nbrs_centered, dim=-1, keepdim=True)

        pc_normal_nbrs = torch.gather(pc_normal.unsqueeze(-3).expand(*pc_normal.shape[:-1], *pc_normal.shape[-2:]), -2,
                                      nbrs_idx[..., None].expand(*nbrs_idx.shape,
                                                                 pc_normal.shape[-1]))  # [..., N, K, 3]
        pc_normal_cos = torch.sum(pc_normal_nbrs * pc_normal.unsqueeze(-2), -1, keepdim=True)

        feat = self.aggrs[0](self.spconvs[0](pc_nbrs, torch.cat([pc_nbrs_norm, pc_normal_cos], -1), pc))
        for i in range(len(self.spconvs) - 1):
            spconv = self.spconvs[i + 1]
            aggr = self.aggrs[i + 1]
            feat_nbrs = torch.gather(feat.unsqueeze(-3).expand(*feat.shape[:-1], *feat.shape[-2:]), -2,
                                     nbrs_idx[..., None].expand(*nbrs_idx.shape, feat.shape[-1]))
            feat = aggr(spconv(pc_nbrs, feat_nbrs, pc))
        return feat

    def forward_nbrs(self, pc, pc_normal, nbrs_idx):
        pc_nbrs = torch.gather(pc.unsqueeze(-3).expand(*pc.shape[:-1], *pc.shape[-2:]), -2,
                               nbrs_idx[..., None].expand(*nbrs_idx.shape, pc.shape[-1]))  # [..., N, K, 3]
        pc_nbrs_centered = pc_nbrs - pc.unsqueeze(-2)  # [..., N, K, 3]
        pc_nbrs_norm = torch.norm(pc_nbrs_centered, dim=-1, keepdim=True)

        pc_normal_nbrs = torch.gather(pc_normal.unsqueeze(-3).expand(*pc_normal.shape[:-1], *pc_normal.shape[-2:]), -2,
                                      nbrs_idx[..., None].expand(*nbrs_idx.shape,
                                                                 pc_normal.shape[-1]))  # [..., N, K, 3]
        pc_normal_cos = torch.sum(pc_normal_nbrs * pc_normal.unsqueeze(-2), -1, keepdim=True)

        feat = self.aggrs[0](self.spconvs[0](pc_nbrs, torch.cat([pc_nbrs_norm, pc_normal_cos], -1), pc))
        for i in range(len(self.spconvs) - 1):
            spconv = self.spconvs[i + 1]
            aggr = self.aggrs[i + 1]
            feat_nbrs = torch.gather(feat.unsqueeze(-3).expand(*feat.shape[:-1], *feat.shape[-2:]), -2,
                                     nbrs_idx[..., None].expand(*nbrs_idx.shape, feat.shape[-1]))
            feat = aggr(spconv(pc_nbrs, feat_nbrs, pc))
        return feat


class PPFEncoder(nn.Module):
    def __init__(self, ppffcs, out_dim) -> None:
        super().__init__()
        self.res_layers = nn.ModuleList()
        for i in range(len(ppffcs) - 1):
            dim_in, dim_out = ppffcs[i], ppffcs[i + 1]
            self.res_layers.append(ResLayer(dim_in, dim_out, bn=False))
        self.final = nn.Linear(ppffcs[-1], out_dim)

    def forward(self, pc, pc_normal, feat, dist=None, idxs=None):
        if idxs is not None:
            return self.forward_with_idx(pc[0], pc_normal[0], feat[0], idxs)[None]
        xx = pc.unsqueeze(-2) - pc.unsqueeze(-3)
        xx_normed = xx / (dist[..., None] + 1e-7)

        outputs = []
        for idx in torch.chunk(torch.arange(pc.shape[1]), 5):
            feat_chunk = feat[..., idx, :]
            target_shape = [*feat_chunk.shape[:-2], feat_chunk.shape[-2], feat.shape[-2],
                            feat_chunk.shape[-1]]  # B x NC x N x F
            xx_normed_chunk = xx_normed[..., idx, :, :]
            ppf = torch.cat([
                torch.sum(pc_normal[..., idx, :].unsqueeze(-2) * xx_normed_chunk, -1, keepdim=True),
                torch.sum(pc_normal.unsqueeze(-3) * xx_normed_chunk, -1, keepdim=True),
                torch.sum(pc_normal[..., idx, :].unsqueeze(-2) * pc_normal.unsqueeze(-3), -1, keepdim=True),
                dist[..., idx, :, None],
            ], -1)
            # ppf.zero_()
            final_feat = torch.cat(
                [feat_chunk[..., None, :].expand(*target_shape), feat[..., None, :, :].expand(*target_shape), ppf], -1)

            output = final_feat
            for res_layer in self.res_layers:
                output = res_layer(output)
            outputs.append(output)

        output = torch.cat(outputs, dim=-3)
        return self.final(output)

    def forward_with_idx(self, pc, pc_normal, feat, idxs):
        a_idxs = idxs[:, 0]
        b_idxs = idxs[:, 1]
        xy = pc[a_idxs] - pc[b_idxs]
        xy_norm = torch.norm(xy, dim=-1)
        xy_normed = xy / (xy_norm[..., None] + 1e-7)
        pnormal_cos = pc_normal[a_idxs] * pc_normal[b_idxs]
        ppf = torch.cat([
            torch.sum(pc_normal[a_idxs] * xy_normed, -1, keepdim=True),
            torch.sum(pc_normal[b_idxs] * xy_normed, -1, keepdim=True),
            torch.sum(pnormal_cos, -1, keepdim=True),
            xy_norm[..., None],
        ], -1)
        # ppf.zero_()

        final_feat = torch.cat([feat[a_idxs], feat[b_idxs], ppf], -1)

        output = final_feat
        for res_layer in self.res_layers:
            output = res_layer(output)
        return self.final(output)



