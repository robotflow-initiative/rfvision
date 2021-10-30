import rflib
import torch

from rfvision.components.roi_heads.mask_heads import FCNMaskHead, MaskRCNNDCTHead
from utils import _dummy_bbox_sampling


def test_FCNMaskHead_loss():
    """Test mask head loss when mask target is empty."""
    self = FCNMaskHead(
        num_convs=1,
        roi_feat_size=6,
        in_channels=8,
        conv_out_channels=8,
        num_classes=8)

    # Dummy proposals
    proposal_list = [
        torch.Tensor([[23.6667, 23.8757, 228.6326, 153.8874]]),
    ]

    gt_bboxes = [
        torch.Tensor([[23.6667, 23.8757, 238.6326, 151.8874]]),
    ]
    gt_labels = [torch.LongTensor([2])]
    sampling_results = _dummy_bbox_sampling(proposal_list, gt_bboxes,
                                            gt_labels)

    # create dummy mask
    import numpy as np
    from rfvision.core import BitmapMasks
    dummy_mask = np.random.randint(0, 2, (1, 160, 240), dtype=np.uint8)
    gt_masks = [BitmapMasks(dummy_mask, 160, 240)]

    # create dummy train_cfg
    train_cfg = rflib.Config(dict(mask_size=12, mask_thr_binary=0.5))

    # Create dummy features "extracted" for each sampled bbox
    num_sampled = sum(len(res.bboxes) for res in sampling_results)
    dummy_feats = torch.rand(num_sampled, 8, 6, 6)

    mask_pred = self.forward(dummy_feats)
    mask_targets = self.get_targets(sampling_results, gt_masks, train_cfg)
    pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
    loss_mask = self.loss(mask_pred, mask_targets, pos_labels)

    onegt_mask_loss = sum(loss_mask['loss_mask'])
    assert onegt_mask_loss.item() > 0, 'mask loss should be non-zero'
    return onegt_mask_loss

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