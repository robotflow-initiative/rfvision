from rfvision.models.builder import HEADS
from rfvision.components.utils.dct_utils import dct_2d, idct_2d
from rfvision.components.roi_heads.mask_heads import FCNMaskHead
from rflib.cnn import ConvModule
import torch.nn as nn
import numpy as np
import torch
from torch.nn import functional as F
from .fcn_mask_head import GPU_MEM_LIMIT, BYTES_PER_FLOAT, _do_paste_mask, warn


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
        return dct_vectors  # [N, vec_dim]

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
    Refers to https://github.com/aliyun/DCT-Mask
    A mask head with several conv layers, plus an upsample layer (with `ConvTranspose2d`).
    Predictions are made with a final 1x1 conv layer.
    """

    def __init__(self,
                 num_convs=4,
                 in_channels=256,
                 dct_vector_dim=300,
                 mask_size=128,
                 dct_loss_type='l1',
                 mask_loss_para=0.007,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        cfg = dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.dct_vector_dim = dct_vector_dim
        self.dct_loss_type = dct_loss_type
        self.mask_loss_para = mask_loss_para
        self.mask_size = mask_size
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
            mask_loss = mask_pred.sum() * 0
        else:
            mask_targets = self.coder.encode(mask_targets) # shape (instances_num, 300)
            if self.dct_loss_type == "l1":
                num_instance = mask_targets.size()[0]
                mask_loss = F.l1_loss(mask_pred, mask_targets, reduction="none")
                mask_loss = self.mask_loss_para * mask_loss / num_instance
                mask_loss = torch.sum(mask_loss)

            elif self.dct_loss_type == "sl1":
                num_instance = mask_targets.size()[0]
                mask_loss = F.smooth_l1_loss(mask_pred, mask_targets, reduction="none")
                mask_loss = self.mask_loss_para * mask_loss / num_instance
                mask_loss = torch.sum(mask_loss)
            elif self.dct_loss_type == "l2":
                num_instance = mask_targets.size()[0]
                mask_loss = F.mse_loss(mask_pred, mask_targets, reduction="none")
                mask_loss = self.mask_loss_para * mask_loss / num_instance
                mask_loss = torch.sum(mask_loss)
            else:
                raise ValueError("Loss Type Only Support : l1, l2; yours: {}".format(self.dct_loss_type))
        loss['loss_mask'] = mask_loss
        return loss


    def dct_style_to_fcn_style(self, mask_pred, det_labels):
        ##################### convert shape of mask pred to FCNMaskHead used shape #################
        mask_pred = self.coder.decode(mask_pred.detach())  # shape (instances_num, 128, 128)
        # mask_pred  shape(instances_num, 128, 128) to shape(instances_num, 80, 128, 128)
        device = mask_pred.device
        # mask_pred shape in FCNMaskHead.get_seg_mask is (instances_num, 80, self.mask_size, self.mask_size)
        # example mask_pred.shape (18, 80, 28, 28) dim = 4,
        # but the self.coder.decode(mask_pred.detach()) output shape is (instances_num, 128, 128) dim = 3

        mask_pred_temp = torch.zeros((mask_pred.shape[0], 80, self.mask_size, self.mask_size),
                                      dtype=torch.float,
                                      device=device)
        for i, label in enumerate(det_labels):
            mask_pred_temp[i, label, :, :] = mask_pred[i]
        return mask_pred_temp

    def get_seg_masks(self, mask_pred, det_bboxes, det_labels, rcnn_test_cfg,
                      ori_shape, scale_factor, rescale):

        mask_pred = self.dct_style_to_fcn_style(mask_pred, det_labels)
        # if isinstance(mask_pred, torch.Tensor):
        #     mask_pred = mask_pred.sigmoid()
        # else:
        #     # In AugTest, has been activated before
        #     mask_pred = det_bboxes.new_tensor(mask_pred)
        if not isinstance(mask_pred, torch.Tensor):
            # in dct mask sigmoid is not used!!!
            mask_pred = det_bboxes.new_tensor(mask_pred)

        device = mask_pred.device
        cls_segms = [[] for _ in range(self.num_classes)
                     ]  # BG is not included in num_classes
        bboxes = det_bboxes[:, :4]
        labels = det_labels

        # In most cases, scale_factor should have been
        # converted to Tensor when rescale the bbox
        if not isinstance(scale_factor, torch.Tensor):
            if isinstance(scale_factor, float):
                scale_factor = np.array([scale_factor] * 4)
                warn('Scale_factor should be a Tensor or ndarray '
                     'with shape (4,), float would be deprecated. ')
            assert isinstance(scale_factor, np.ndarray)
            scale_factor = torch.Tensor(scale_factor)

        if rescale:
            img_h, img_w = ori_shape[:2]
            bboxes = bboxes / scale_factor
        else:
            w_scale, h_scale = scale_factor[0], scale_factor[1]
            img_h = np.round(ori_shape[0] * h_scale.item()).astype(np.int32)
            img_w = np.round(ori_shape[1] * w_scale.item()).astype(np.int32)

        N = len(mask_pred)
        # The actual implementation split the input into chunks,
        # and paste them chunk by chunk.
        if device.type == 'cpu':
            # CPU is most efficient when they are pasted one by one with
            # skip_empty=True, so that it performs minimal number of
            # operations.
            num_chunks = N
        else:
            # GPU benefits from parallelism for larger chunks,
            # but may have memory issue
            # the types of img_w and img_h are np.int32,
            # when the image resolution is large,
            # the calculation of num_chunks will overflow.
            # so we neet to change the types of img_w and img_h to int.
            # See https://github.com/open-mmlab/mmdetection/pull/5191
            num_chunks = int(
                np.ceil(N * int(img_h) * int(img_w) * BYTES_PER_FLOAT /
                        GPU_MEM_LIMIT))
            assert (num_chunks <=
                    N), 'Default GPU_MEM_LIMIT is too small; try increasing it'
        chunks = torch.chunk(torch.arange(N, device=device), num_chunks)

        threshold = rcnn_test_cfg.mask_thr_binary
        im_mask = torch.zeros(
            N,
            img_h,
            img_w,
            device=device,
            dtype=torch.bool if threshold >= 0 else torch.uint8)

        if not self.class_agnostic:
            mask_pred = mask_pred[range(N), labels][:, None]

        for inds in chunks:
            masks_chunk, spatial_inds = _do_paste_mask(
                mask_pred[inds],
                bboxes[inds],
                img_h,
                img_w,
                skip_empty=device.type == 'cpu')

            if threshold >= 0:
                masks_chunk = (masks_chunk >= threshold).to(dtype=torch.bool)
            else:
                # for visualization and debugging
                masks_chunk = (masks_chunk * 255).to(dtype=torch.uint8)

            im_mask[(inds, ) + spatial_inds] = masks_chunk

        for i in range(N):
            cls_segms[labels[i]].append(im_mask[i].detach().cpu().numpy())
        return cls_segms

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
