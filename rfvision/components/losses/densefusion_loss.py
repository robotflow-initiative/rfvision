from torch.nn.modules.loss import _Loss
import torch
from robotflow.rflib.ops import KNearestNeighbor
from robotflow.rflearner.builder import LOSSES
from robotflow.rflearner.bricks.utils.knn import knn_search


def loss_refinement(pred_r, pred_t, target, model_points, idx, points, num_point_mesh, sym_list):
    pred_r = pred_r.view(1, 1, -1)
    pred_t = pred_t.view(1, 1, -1)
    bs, num_p, _ = pred_r.size()
    num_input_points = len(points[0])

    pred_r = pred_r / (torch.norm(pred_r, dim=2).view(bs, num_p, 1))

    base = torch.cat(((1.0 - 2.0 * (pred_r[:, :, 2] ** 2 + pred_r[:, :, 3] ** 2)).view(bs, num_p, 1), \
                      (2.0 * pred_r[:, :, 1] * pred_r[:, :, 2] - 2.0 * pred_r[:, :, 0] * pred_r[:, :, 3]).view(bs,
                                                                                                               num_p,
                                                                                                               1), \
                      (2.0 * pred_r[:, :, 0] * pred_r[:, :, 2] + 2.0 * pred_r[:, :, 1] * pred_r[:, :, 3]).view(bs,
                                                                                                               num_p,
                                                                                                               1), \
                      (2.0 * pred_r[:, :, 1] * pred_r[:, :, 2] + 2.0 * pred_r[:, :, 3] * pred_r[:, :, 0]).view(bs,
                                                                                                               num_p,
                                                                                                               1), \
                      (1.0 - 2.0 * (pred_r[:, :, 1] ** 2 + pred_r[:, :, 3] ** 2)).view(bs, num_p, 1), \
                      (-2.0 * pred_r[:, :, 0] * pred_r[:, :, 1] + 2.0 * pred_r[:, :, 2] * pred_r[:, :, 3]).view(bs,
                                                                                                                num_p,
                                                                                                                1), \
                      (-2.0 * pred_r[:, :, 0] * pred_r[:, :, 2] + 2.0 * pred_r[:, :, 1] * pred_r[:, :, 3]).view(bs,
                                                                                                                num_p,
                                                                                                                1), \
                      (2.0 * pred_r[:, :, 0] * pred_r[:, :, 1] + 2.0 * pred_r[:, :, 2] * pred_r[:, :, 3]).view(bs,
                                                                                                               num_p,
                                                                                                               1), \
                      (1.0 - 2.0 * (pred_r[:, :, 1] ** 2 + pred_r[:, :, 2] ** 2)).view(bs, num_p, 1)),
                     dim=2).contiguous().view(bs * num_p, 3, 3)

    ori_base = base
    base = base.contiguous().transpose(2, 1).contiguous()
    model_points = model_points.view(bs, 1, num_point_mesh, 3).repeat(1, num_p, 1, 1).view(bs * num_p, num_point_mesh,
                                                                                           3)
    target = target.view(bs, 1, num_point_mesh, 3).repeat(1, num_p, 1, 1).view(bs * num_p, num_point_mesh, 3)
    ori_target = target
    pred_t = pred_t.contiguous().view(bs * num_p, 1, 3)
    ori_t = pred_t

    pred = torch.add(torch.bmm(model_points, base), pred_t)

    if idx[0].item() in sym_list:
        target = target[0].transpose(1, 0).contiguous().view(3, 1)
        pred = pred.permute(2, 0, 1).contiguous().view(3, 1)
        dist_mat, inds = knn_search(target.T, pred.T, k=1)
        target = torch.index_select(target, 1, inds.view(-1) - 1)
        target = target.view(3, bs * num_p, num_point_mesh).permute(1, 2, 0).contiguous()
        pred = pred.view(3, bs * num_p, num_point_mesh).permute(1, 2, 0).contiguous()

    dis = torch.mean(torch.norm((pred - target), dim=2), dim=1)

    t = ori_t[0]
    points = points.view(1, num_input_points, 3)

    ori_base = ori_base[0].view(1, 3, 3).contiguous()
    ori_t = t.repeat(bs * num_input_points, 1).contiguous().view(1, bs * num_input_points, 3)
    new_points = torch.bmm((points - ori_t), ori_base).contiguous()

    new_target = ori_target[0].view(1, num_point_mesh, 3).contiguous()
    ori_t = t.repeat(num_point_mesh, 1).contiguous().view(1, num_point_mesh, 3)
    new_target = torch.bmm((new_target - ori_t), ori_base).contiguous()

    return dis / bs, new_points.detach(), new_target.detach()


def loss_estimation(pred_r, pred_t, pred_c, target, model_points, idx, points, w, refine, num_point_mesh, sym_list):
    bs, num_p, _ = pred_c.size()

    pred_r = pred_r / (torch.norm(pred_r, dim=2).view(bs, num_p, 1))

    base = torch.cat(((1.0 - 2.0 * (pred_r[:, :, 2] ** 2 + pred_r[:, :, 3] ** 2)).view(bs, num_p, 1), \
                      (2.0 * pred_r[:, :, 1] * pred_r[:, :, 2] - 2.0 * pred_r[:, :, 0] * pred_r[:, :, 3]).view(bs,
                                                                                                               num_p,
                                                                                                               1), \
                      (2.0 * pred_r[:, :, 0] * pred_r[:, :, 2] + 2.0 * pred_r[:, :, 1] * pred_r[:, :, 3]).view(bs,
                                                                                                               num_p,
                                                                                                               1), \
                      (2.0 * pred_r[:, :, 1] * pred_r[:, :, 2] + 2.0 * pred_r[:, :, 3] * pred_r[:, :, 0]).view(bs,
                                                                                                               num_p,
                                                                                                               1), \
                      (1.0 - 2.0 * (pred_r[:, :, 1] ** 2 + pred_r[:, :, 3] ** 2)).view(bs, num_p, 1), \
                      (-2.0 * pred_r[:, :, 0] * pred_r[:, :, 1] + 2.0 * pred_r[:, :, 2] * pred_r[:, :, 3]).view(bs,
                                                                                                                num_p,
                                                                                                                1), \
                      (-2.0 * pred_r[:, :, 0] * pred_r[:, :, 2] + 2.0 * pred_r[:, :, 1] * pred_r[:, :, 3]).view(bs,
                                                                                                                num_p,
                                                                                                                1), \
                      (2.0 * pred_r[:, :, 0] * pred_r[:, :, 1] + 2.0 * pred_r[:, :, 2] * pred_r[:, :, 3]).view(bs,
                                                                                                               num_p,
                                                                                                               1), \
                      (1.0 - 2.0 * (pred_r[:, :, 1] ** 2 + pred_r[:, :, 2] ** 2)).view(bs, num_p, 1)),
                     dim=2).contiguous().view(bs * num_p, 3, 3)

    ori_base = base
    base = base.contiguous().transpose(2, 1).contiguous()
    model_points = model_points.view(bs, 1, num_point_mesh, 3).repeat(1, num_p, 1, 1).view(bs * num_p, num_point_mesh,
                                                                                           3)
    target = target.view(bs, 1, num_point_mesh, 3).repeat(1, num_p, 1, 1).view(bs * num_p, num_point_mesh, 3)
    ori_target = target
    pred_t = pred_t.contiguous().view(bs * num_p, 1, 3)
    ori_t = pred_t
    points = points.contiguous().view(bs * num_p, 1, 3)
    pred_c = pred_c.contiguous().view(bs * num_p)

    pred = torch.add(torch.bmm(model_points, base), points + pred_t)

    if not refine:
        if idx[0].item() in sym_list:
            target = target[0].transpose(1, 0).contiguous().view(3, -1)
            pred = pred.permute(2, 0, 1).contiguous().view(3, -1)
            dist_mat, inds = knn_search(target.T, pred.T, k=1)
            target = torch.index_select(target, 1, inds.view(-1).detach() - 1)
            target = target.view(3, bs * num_p, num_point_mesh).permute(1, 2, 0).contiguous()
            pred = pred.view(3, bs * num_p, num_point_mesh).permute(1, 2, 0).contiguous()

    dis = torch.mean(torch.norm((pred - target), dim=2), dim=1)
    loss = torch.mean((dis * pred_c - w * torch.log(pred_c)), dim=0)

    pred_c = pred_c.view(bs, num_p)
    how_max, which_max = torch.max(pred_c, 1)
    dis = dis.view(bs, num_p)

    t = ori_t[which_max[0]] + points[which_max[0]]
    points = points.view(1, bs * num_p, 3)

    ori_base = ori_base[which_max[0]].view(1, 3, 3).contiguous()
    ori_t = t.repeat(bs * num_p, 1).contiguous().view(1, bs * num_p, 3)
    new_points = torch.bmm((points - ori_t), ori_base).contiguous()

    new_target = ori_target[0].view(1, num_point_mesh, 3).contiguous()
    ori_t = t.repeat(num_point_mesh, 1).contiguous().view(1, num_point_mesh, 3)
    new_target = torch.bmm((new_target - ori_t), ori_base).contiguous()

    return loss, dis[0][which_max[0]] / bs, new_points.detach(), new_target.detach()


@LOSSES.register_module()
class DenseFusionEstimationLoss(_Loss):

    def __init__(self, num_points_mesh, sym_list):
        super(DenseFusionEstimationLoss, self).__init__(True)
        self.num_pt_mesh = num_points_mesh
        self.sym_list = sym_list

    def forward(self, pred_dict, target, model_points, idx, points, w=0.1, refine=False):
        pred_r, pred_t, pred_c = pred_dict['out_rx'], pred_dict['out_tx'], pred_dict['out_cx']
        loss, dis, new_points, new_target = loss_estimation(pred_r, pred_t, pred_c, target, model_points, idx, points,
                                                            w, refine, self.num_pt_mesh,
                                                            self.sym_list)
        loss_dict = dict(loss=loss, dis=dis)
        return loss_dict, new_points, new_target


@LOSSES.register_module()
class DenseFusionRefinementLoss(_Loss):

    def __init__(self, num_points_mesh, sym_list):
        super(DenseFusionRefinementLoss, self).__init__(True)
        self.num_pt_mesh = num_points_mesh
        self.sym_list = sym_list

    def forward(self, pred_dict, target, model_points, idx, points, w=0.1, refine=False):
        pred_r, pred_t = pred_dict['out_rx'], pred_dict['out_tx']
        dis, new_points, new_target = loss_refinement(pred_r, pred_t, target, model_points, idx, points,
                                                      self.num_pt_mesh,
                                                      self.sym_list)
        loss_dict = dict(dis=dis)
        return loss_dict, new_points, new_target
