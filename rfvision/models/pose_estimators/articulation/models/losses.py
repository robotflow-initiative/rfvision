import torch
import torch.nn as nn
import torch.nn.functional as F

DIVISION_EPS = 1e-10


class MIoULoss(nn.Module):
    def __init__(self):
        super(MIoULoss, self).__init__()

    def forward(self, W, I_gt, matching_indices=None):
        W_reordered = W

        depth = W_reordered.shape[2]
        W_gt = F.one_hot(I_gt.long(), num_classes=depth).float()
        dot = torch.sum(W_gt * W_reordered, 1)
        denominator = torch.sum(W_gt, 1) + torch.sum(W_reordered, 1) - dot
        mIoU = dot / (denominator + DIVISION_EPS)
        return 1.0 - mIoU


class NOCSLoss(nn.Module):
    def __init__(self, TYPE_L='L2',
                 MULTI_HEAD=False, SELF_SU=False, contain_bg=True):
        super(NOCSLoss, self).__init__()
        self.type = TYPE_L
        self.multi_head = MULTI_HEAD
        self.self_su = SELF_SU
        self.contain_bg = contain_bg

    def forward(self, nocs, nocs_gt, confidence, mask_array=None):
        if self.contain_bg:
            start_part = 1
        else:
            start_part = 0
        n_parts = int(nocs.shape[2] / 3)
        if self.multi_head:
            loss_nocs = 0
            nocs_splits = torch.split(nocs, split_size_or_sections=int(nocs.shape[2] / n_parts), dim=2)
            mask_splits = torch.split(mask_array, split_size_or_sections=int(mask_array.shape[2] / n_parts), dim=2)
            for i in range(start_part, n_parts):
                diff_l2 = torch.norm(nocs_splits[i] - nocs_gt, dim=2)
                diff_abs = torch.sum(torch.abs(nocs_splits[i] - nocs_gt), dim=2)
                if not self.self_su:
                    if self.type == 'L2':
                        loss_nocs += torch.mean(mask_splits[i][:, :, 0] * diff_l2, dim=1)
                    else:
                        loss_nocs += torch.mean(mask_splits[i][:, :, 0] * diff_abs, dim=1)
                else:
                    if self.type == 'L2':
                        loss_nocs += torch.mean(mask_splits[i][:, :, 0] * diff_l2 * confidence[:, :, 0], dim=1)
                    else:
                        loss_nocs += torch.mean(mask_splits[i][:, :, 0] * diff_abs * confidence[:, :, 0], dim=1)
                if self.self_su:
                    loss_nocs += - 0.1 * torch.mean(confidence[:, :, 0].log(), axis=1)

            return loss_nocs

        else:
            diff_l2 = torch.norm(nocs - nocs_gt, dim=2)  # BxN
            diff_abs = torch.sum(torch.abs(nocs - nocs_gt), dim=2)  # BxN
            if not self.self_su:
                if self.type == 'L2':
                    return torch.mean(diff_l2, dim=1)  # B
                else:
                    return torch.mean(diff_abs, dim=1)  # B
            else:
                if self.type == 'L2':
                    return torch.mean(diff_l2 * confidence[:, :, 0] - 0.1 * confidence[:, :, 0].log(), dim=1)  # B
                else:
                    return torch.mean(confidence[:, :, 0] * diff_abs - 0.1 * confidence[:, :, 0].log(), dim=1)  # B


class VECTLoss(nn.Module):
    def __init__(self, TYPE_L='L2',
                 MULTI_HEAD=False, SELF_SU=False, contain_bg=True):
        super(VECTLoss, self).__init__()
        self.type = TYPE_L
        self.multi_head = MULTI_HEAD
        self.self_su = SELF_SU
        self.contain_bg = contain_bg

    def forward(self, vect, vect_gt, confidence=None, mask_array=None):
        if self.multi_head:
            if vect_gt.shape[2] == 1:
                n_parts = int(vect.shape[2])
            else:
                n_parts = int(vect.shape[2] / 3)
            start_part = 1
            loss_vect = 0
            vect_splits = torch.split(vect, split_size_or_sections=int(vect.shape[2] / n_parts), dim=2)
            mask_splits = torch.split(mask_array, split_size_or_sections=int(mask_array.shape[2] / n_parts), dim=2)
            for i in range(start_part, n_parts):
                diff_l2 = torch.norm(vect_splits[i] - vect_gt, dim=2)  # BxN
                diff_abs = torch.sum(torch.abs(vect_splits[i] - vect_gt), dim=2)
                if not self.self_su:
                    if self.type == 'L2':
                        loss_vect += torch.mean(mask_splits[i][:, :, 0] * diff_l2, dim=1)
                    else:
                        loss_vect += torch.mean(mask_splits[i][:, :, 0] * diff_abs, dim=1)
                else:
                    if self.type == 'L2':
                        loss_vect += torch.mean(mask_splits[i][:, :, 0] * diff_l2 * confidence[:, :, 0], dim=1)
                    else:
                        loss_vect += torch.mean(mask_splits[i][:, :, 0] * diff_abs * confidence[:, :, 0], dim=1)
                if self.self_su:
                    loss_vect += - 0.01 * torch.mean(confidence[:, :, 0].log(), dim=1)

            return loss_vect

        else:
            if vect.shape[2] == 1:
                vect = torch.squeeze(vect, 2)
                if confidence is not None:
                    diff_l2 = torch.abs(vect - vect_gt) * confidence
                    diff_abs = torch.abs(vect - vect_gt) * confidence
                else:
                    diff_l2 = torch.abs(vect - vect_gt)
                    diff_abs = torch.abs(vect - vect_gt)
            else:
                if confidence is not None:
                    diff_l2 = torch.norm(vect - vect_gt, dim=2) * confidence
                    diff_abs = torch.sum(torch.abs(vect - vect_gt), dim=2) * confidence
                else:
                    diff_l2 = torch.norm(vect - vect_gt, dim=2)
                    diff_abs = torch.sum(torch.abs(vect - vect_gt), dim=2)
            if not self.self_su:
                if self.type == 'L2':
                    return torch.mean(diff_l2, dim=1)
                else:
                    return torch.mean(diff_abs, dim=1)
            else:
                if self.type == 'L2':
                    return torch.mean(diff_l2 * confidence[:, :, 0] - 0.01 * confidence[:, :, 0].log(), dim=1)
                else:
                    return torch.mean(confidence[:, :, 0] * diff_abs - 0.01 * confidence[:, :, 0].log(), dim=1)