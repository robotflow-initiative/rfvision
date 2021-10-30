import torch.nn as nn
from rfvision.components import ResLayer
import torch


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


def conv_kernel(iunit, ounit, *hunits):
    layers = []
    for unit in hunits:
        layers.append(nn.Linear(iunit, unit))
        layers.append(nn.LayerNorm(unit))
        layers.append(nn.ReLU())
        iunit = unit
    layers.append(nn.Linear(iunit, ounit))
    return nn.Sequential(*layers)


class GlobalInfoProp(nn.Module):
    def __init__(self, n_in, n_global):
        super().__init__()
        self.linear = nn.Linear(n_in, n_global)

    def forward(self, feat):
        # [b, k, n_in] -> [b, k, n_in + n_global]
        tran = self.linear(feat)
        glob = tran.max(-2, keepdim=True)[0].expand(*feat.shape[:-1], tran.shape[-1])
        return torch.cat([feat, glob], -1)


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