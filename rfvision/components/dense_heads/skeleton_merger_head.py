import torch.nn as nn 
import torch.nn.functional as F
import torch 
from rfvision.models.builder import HEADS


def minimum(x, y):
    return torch.min(torch.stack((x, y)), dim=0)[0]

def L2(embed):
    return 0.01 * (torch.sum(embed ** 2))

def composed_sqrt_chamfer(y_true, y_preds, activations):
    L = 0.0
    # activations: N x P where P: # sub-clouds
    # y_true: N x ? x 3
    # y_pred: P x N x ? x 3 
    part_backs = []
    for i, y_pred in enumerate(y_preds):
        # y_true: k1 x 3
        # y_pred: k2 x 3
        y_true_rep = torch.unsqueeze(y_true, axis=-2)  # k1 x 1 x 3
        y_pred_rep = torch.unsqueeze(y_pred, axis=-3)  # 1 x k2 x 3
        # k1 x k2 x 3
        y_delta = torch.sqrt(1e-4 + torch.sum((y_pred_rep - y_true_rep)**2, -1))
        # k1 x k2
        y_nearest = torch.min(y_delta, -2)[0]
        # k2
        part_backs.append(torch.min(y_delta, -1)[0])
        L = L + torch.mean(torch.mean(y_nearest, -1) * activations[:, i]) / len(y_preds)
    part_back_stacked = torch.stack(part_backs)  # P x N x k1
    sorted_parts, indices = torch.sort(part_back_stacked, dim=0)
    weights = torch.ones_like(sorted_parts[0])  # N x k1
    for i in range(len(y_preds)):
        w = minimum(weights, torch.gather(activations, -1, indices[i]))
        L = L + torch.mean(sorted_parts[i] * w)
        weights = weights - w
    L = L + torch.mean(weights * 20.0)
    return L


class PBlock(nn.Module):  # MLP Block
    def __init__(self, iu, *units, should_perm):
        super().__init__()
        self.sublayers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.should_perm = should_perm
        ux = iu
        for uy in units:
            self.sublayers.append(nn.Linear(ux, uy))
            self.batch_norms.append(nn.BatchNorm1d(uy))
            ux = uy

    def forward(self, input_x):
        x = input_x
        for sublayer, batch_norm in zip(self.sublayers, self.batch_norms):
            x = sublayer(x)
            if self.should_perm:
                x = x.permute(0, 2, 1)
            x = batch_norm(x)
            if self.should_perm:
                x = x.permute(0, 2, 1)
            x = F.relu(x)
        return x


class DecoderUnit(nn.Module):  # Decoder unit, one per line
    def __init__(self):
        super().__init__()
        self.emb = nn.Parameter(torch.randn((200, 3)) * 0.002)

    def forward(self, KPA, KPB):
        dist = torch.mean(torch.sqrt(1e-3 + (torch.sum(torch.square(KPA - KPB), dim=-1))))
        count = min(200, max(15, int((dist / 0.01).item())))
        device = dist.device
        self.f_interp = torch.linspace(0.0, 1.0, count).unsqueeze(0).unsqueeze(-1).to(device)
        self.b_interp = 1.0 - self.f_interp
        # KPA: N x 3, KPB: N x 3
        # Interpolated: N x count x 3
        K = KPA.unsqueeze(-2) * self.f_interp + KPB.unsqueeze(-2) * self.b_interp
        R = self.emb[:count, :].unsqueeze(0) + K  # N x count x 3
        return R.reshape((-1, count, 3)), self.emb
    
    
@HEADS.register_module()
class SkeletonMergerHead(nn.Module):  # Skeleton Merger structure
    def __init__(self, n_keypoint):
        super().__init__()
        self.k = n_keypoint
        self.PT_L = nn.Linear(self.k, self.k)
        self.MA_EMB = nn.Parameter(torch.randn([self.k * (self.k - 1) // 2]))
        self.MA = PBlock(1024, 512, 256, should_perm=False)
        self.MA_L = nn.Linear(256, self.k * (self.k - 1) // 2)
        self.DEC = nn.ModuleList()
        for i in range(self.k):
            DECN = nn.ModuleList()
            for j in range(i):
                DECN.append(DecoderUnit())
            self.DEC.append(DECN)

    def forward(self,input_x, KP, GF):
        KPL = self.PT_L(KP)
        KPA = F.softmax(KPL.permute(0, 2, 1), -1)  # [n, k, npt]
        KPCD = KPA.bmm(input_x)  # [n, k, 3]
        RP = []
        L = []
        for i in range(self.k):
            for j in range(i):
                R, EM = self.DEC[i][j](KPCD[:, i, :], KPCD[:, j, :])
                RP.append(R)
                L.append(EM)
        GFP = F.max_pool1d(GF, 16).squeeze(-1)
        MA = torch.sigmoid(self.MA_L(self.MA(GFP)))
        # MA = torch.sigmoid(self.MA_EMB).expand(input_x.shape[0], -1)
        LF = torch.cat(L, dim=1)  # P x 72 x 3
        
        pred_dict = {'reconstructed_parts':RP,
                     'keypoints_xyz':KPCD,
                     'keypoint_activation':KPA,
                     'learned_offset':LF,
                     'mask':MA}
        
        return pred_dict
        
    def loss(self, input_x, pred_dict):
        RP = pred_dict['reconstructed_parts']
        LF = pred_dict['learned_offset']
        MA = pred_dict['mask']
        blrc = composed_sqrt_chamfer(input_x, RP, MA)
        bldiv = L2(LF)
        
        losses = {'loss_rc':blrc,
                  'loss_div':bldiv}
        return losses
