from rfvision.models.builder import HEADS, build_loss
from rfvision.components.utils.dct_utils import dct_2d, idct_2d
from rfvision.components.roi_heads.mask_heads import FCNMaskHead
from rflib.cnn import ConvModule
import torch.nn as nn
import numpy as np
import torch
from torch.nn import functional as F



class DctMaskEncoding(object):
    """
    Apply DCT to encode the binary mask, and use the encoded vector as mask representation in instance segmentation.
    """
    def __init__(self, vec_dim, mask_size=128):
        """
        vec_dim: the dimension of the encoded vector, int
        mask_size: the resolution of the initial binary mask representaiton.
        """
        self.vec_dim = vec_dim
        self.mask_size = mask_size
        assert vec_dim <= mask_size*mask_size
        self.dct_vector_coords = self.get_dct_vector_coords(r=mask_size)

    def encode(self, masks, dim=None):
        """
        Encode the mask to vector of vec_dim or specific dimention.
        """
        if dim is None:
            dct_vector_coords = self.dct_vector_coords[:self.vec_dim]
        else:
            dct_vector_coords = self.dct_vector_coords[:dim]
        masks = masks.view([-1, self.mask_size, self.mask_size]).to(dtype=float)  # [N, H, W]
        dct_all = dct_2d(masks, norm='ortho')
        xs, ys = dct_vector_coords[:, 0], dct_vector_coords[:, 1]
        dct_vectors = dct_all[:, xs, ys]  # reshape as vector
        return dct_vectors  # [N, D]

    def decode(self, dct_vectors, dim=None):
        """
        intput: dct_vector numpy [N,dct_dim]
        output: mask_rc mask reconstructed [N, mask_size, mask_size]
        """
        device = dct_vectors.device
        if dim is None:
            dct_vector_coords = self.dct_vector_coords[:self.vec_dim]
        else:
            dct_vector_coords = self.dct_vector_coords[:dim]
            dct_vectors = dct_vectors[:, :dim]

        N = dct_vectors.shape[0]
        dct_trans = torch.zeros([N, self.mask_size, self.mask_size], dtype=dct_vectors.dtype).to(device)
        xs, ys = dct_vector_coords[:, 0], dct_vector_coords[:, 1]
        dct_trans[:, xs, ys] = dct_vectors
        mask_rc = idct_2d(dct_trans, norm='ortho')  # [N, mask_size, mask_size]
        return mask_rc

    def get_dct_vector_coords(self, r=128):
        """
        Get the coordinates with zigzag order.
        """
        dct_index = []
        for i in range(r):
            if i % 2 == 0:  # start with even number
                index = [(i-j, j) for j in range(i+1)]
                dct_index.extend(index)
            else:
                index = [(j, i-j) for j in range(i+1)]
                dct_index.extend(index)
        for i in range(r, 2*r-1):
            if i % 2 == 0:
                index = [(i-j, j) for j in range(i-r+1,r)]
                dct_index.extend(index)
            else:
                index = [(j, i-j) for j in range(i-r+1,r)]
                dct_index.extend(index)
        dct_idxs = np.asarray(dct_index)
        return dct_idxs


@HEADS.register_module()
class MaskRCNNDCTHead(FCNMaskHead):
    """
    A mask head with several conv layers, plus an upsample layer (with `ConvTranspose2d`).
    Predictions are made with a final 1x1 conv layer.
    """

    def __init__(self,
                 num_convs=5,
                 in_channels=256,
                 dct_vector_dim=300,
                 mask_size=128,
                 loss=dict(type='L1Loss'),
                 class_agnostic=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU'),
                 init_cfg=None):

        super().__init__(init_cfg=init_cfg)

        cfg = dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.dct_vector_dim = dct_vector_dim
        self.mask_size = mask_size
        self.loss_mask = build_loss(loss)
        self.class_agnostic = class_agnostic
        self.coder = DctMaskEncoding(vec_dim=dct_vector_dim, mask_size=mask_size)

        self.convs = nn.ModuleList()
        for i in range(num_convs):
            self.convs.append(
                ConvModule(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    **cfg))

        self.predictor_fc1 = nn.Linear(256 * 14 * 14, 1024)
        self.predictor_fc2 = nn.Linear(1024, 1024)
        self.predictor_fc3 = nn.Linear(1024, dct_vector_dim)

    def init_weights(self, init_cfg=None):
        super().init_weights()
        if self.init_cfg is not None:
            nn.init.normal_(self.predictor_fc3.weight, std=0.001)
            nn.init.constant_(self.predictor_fc3.bias, 0)

    def forward(self, x):
        """
        Args:
            x: input region feature(s) provided by :class:`ROIHeads`.
        Returns:
            A dict of losses in training. The predicted "instances" in inference.
        """
        for layer in self.convs:
            x = layer(x)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.predictor_fc1(x))
        x = F.relu(self.predictor_fc2(x))
        x = self.predictor_fc3(x)
        return x

    def loss(self, mask_pred, mask_targets, labels):
        loss = dict()
        if mask_pred.size(0) == 0:
            loss_mask = mask_pred.sum()
        else:
            mask_targets = self.coder.encode(mask_targets)
            if self.class_agnostic:
                loss_mask = self.loss_mask(mask_pred, mask_targets)
            else:
                loss_mask = self.loss_mask(mask_pred, mask_targets)
        loss['loss_mask'] = loss_mask
        return loss

    def get_seg_masks(self, mask_pred, det_bboxes, det_labels, rcnn_test_cfg,
                      ori_shape, scale_factor, rescale):
        mask_pred = self.coder.decode(mask_pred)
        super().get_seg_masks(mask_pred=mask_pred,
                              det_bboxes= det_bboxes,
                              det_labels=det_labels,
                              rcnn_test_cfg=rcnn_test_cfg,
                              ori_shape=ori_shape,
                              scale_factor=scale_factor,
                              rescale=rescale)

if __name__ == '__main__':
    n = 13
    # test model
    m = MaskRCNNDCTHead()
    t = torch.rand(n, 256, 14, 14)
    res_head = m(t)  # shape (13, 300)

    # test loss

    mask_pred = res_head
    mask_target = torch.rand(n, 128, 128)
    labels = torch.randint(low=0, high=80, size=(n, 1))
    loss = m.loss(mask_pred, mask_target, labels)
