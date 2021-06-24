import torch
import torch.nn as nn
import torch.nn.functional as F

from rfvision.models.builder import build_loss, HEADS


@HEADS.register_module()
class DenseFusionRefinerHead(nn.Module):
    def __init__(self,
                 num_points,
                 num_objects,
                 loss_dis=None):
        super(DenseFusionRefinerHead, self).__init__()
        self.num_points = num_points

        self.conv1_r = torch.nn.Linear(1024, 512)
        self.conv1_t = torch.nn.Linear(1024, 512)

        self.conv2_r = torch.nn.Linear(512, 128)
        self.conv2_t = torch.nn.Linear(512, 128)

        self.conv3_r = torch.nn.Linear(128, num_objects * 4)  # quaternion
        self.conv3_t = torch.nn.Linear(128, num_objects * 3)  # translation

        self.num_objects = num_objects

        self.loss_dis = build_loss(loss_dis)

    def forward(self, ap_x, emb, obj, bs):
        rx = F.relu(self.conv1_r(ap_x))
        tx = F.relu(self.conv1_t(ap_x))

        rx = F.relu(self.conv2_r(rx))
        tx = F.relu(self.conv2_t(tx))

        rx = self.conv3_r(rx).view(bs, self.num_objects, 4)
        tx = self.conv3_t(tx).view(bs, self.num_objects, 3)

        out_rx_list = []
        out_tx_list = []
        for b in range(bs):
            out_rx_list.append(torch.index_select(rx[b], 0, obj[b][0]))
            out_tx_list.append(torch.index_select(tx[b], 0, obj[b][0]))

        out_rx = torch.cat(out_rx_list)
        out_tx = torch.cat(out_tx_list)

        return out_rx, out_tx

    def loss(self, pred_dict, target, model_points, obj, x):
        return self.loss_dis(pred_dict, target, model_points, obj, x)
    
    
@HEADS.register_module()
class DenseFusionEstimatorHead(nn.Module):
    def __init__(self,
                 num_points,
                 num_objects,
                 loss_dis=None):
        super(DenseFusionEstimatorHead, self).__init__()
        self.num_points = num_points

        self.conv1_r = torch.nn.Conv1d(1408, 640, 1)
        self.conv1_t = torch.nn.Conv1d(1408, 640, 1)
        self.conv1_c = torch.nn.Conv1d(1408, 640, 1)

        self.conv2_r = torch.nn.Conv1d(640, 256, 1)
        self.conv2_t = torch.nn.Conv1d(640, 256, 1)
        self.conv2_c = torch.nn.Conv1d(640, 256, 1)

        self.conv3_r = torch.nn.Conv1d(256, 128, 1)
        self.conv3_t = torch.nn.Conv1d(256, 128, 1)
        self.conv3_c = torch.nn.Conv1d(256, 128, 1)

        self.conv4_r = torch.nn.Conv1d(128, num_objects * 4, 1)  # quaternion
        self.conv4_t = torch.nn.Conv1d(128, num_objects * 3, 1)  # translation
        self.conv4_c = torch.nn.Conv1d(128, num_objects * 1, 1)  # confidence

        self.num_objects = num_objects

        self.loss_dis = build_loss(loss_dis)

    def forward(self, ap_x, obj, bs):
        rx = F.relu(self.conv1_r(ap_x))
        tx = F.relu(self.conv1_t(ap_x))
        cx = F.relu(self.conv1_c(ap_x))

        rx = F.relu(self.conv2_r(rx))
        tx = F.relu(self.conv2_t(tx))
        cx = F.relu(self.conv2_c(cx))

        rx = F.relu(self.conv3_r(rx))
        tx = F.relu(self.conv3_t(tx))
        cx = F.relu(self.conv3_c(cx))

        rx = self.conv4_r(rx).view(bs, self.num_objects, 4, self.num_points)
        tx = self.conv4_t(tx).view(bs, self.num_objects, 3, self.num_points)
        cx = torch.sigmoid(self.conv4_c(cx)).view(bs, self.num_objects, 1, self.num_points)

        out_rx_list = []
        out_tx_list = []
        out_cx_list = []
        for b in range(bs):
            out_rx_list.append(torch.index_select(rx[b], 0, obj[b][0]))
            out_tx_list.append(torch.index_select(tx[b], 0, obj[b][0]))
            out_cx_list.append(torch.index_select(cx[b], 0, obj[b][0]))

        out_rx = torch.cat(out_rx_list)
        out_tx = torch.cat(out_tx_list)
        out_cx = torch.cat(out_cx_list)

        out_rx = out_rx.contiguous().transpose(2, 1).contiguous()
        out_tx = out_tx.contiguous().transpose(2, 1).contiguous()
        out_cx = out_cx.contiguous().transpose(2, 1).contiguous()

        return out_rx, out_tx, out_cx

    def loss(self, pred_dict, target, model_points, obj, x):
        return self.loss_dis(pred_dict, target, model_points, obj, x)
