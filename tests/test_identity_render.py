import numpy as np
import torch
from loss_utils.identity_losses import compute_prototypes, identity_2d_prototype_ce_loss, identity_3d_smoothness_loss


def synth_identity_render(anchors_xy, anchor_ids, H=32, W=32, sigma=3.0):
    """Compute a synthetic per-pixel identity render by weighted accumulation of anchor ids.
    anchors_xy: (N,2) pixel coordinates
    anchor_ids: (N,D) identity vectors
    Returns identity_render: (D,H,W)
    """
    N, D = anchor_ids.shape
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')  # H,W
    coords = np.stack([yy, xx], axis=-1)[None, ...]  # 1,H,W,2
    anchors = anchors_xy[:, None, None, :]  # N,1,1,2
    diff = coords - anchors  # N,H,W,2
    dist2 = np.sum(diff**2, axis=-1)  # N,H,W
    weights = np.exp(-dist2 / (2.0 * (sigma**2)))  # N,H,W
    # normalize per-pixel
    denom = np.sum(weights, axis=0, keepdims=True) + 1e-9  # 1,H,W
    weights_norm = weights / denom  # N,H,W
    # accumulate id vectors: (N,D) -> (N,1,1,D) broadcast -> (D,H,W)
    id_arr = anchor_ids[:, :, None, None]  # N,D,1,1 -> we'll transpose
    id_arr = anchor_ids[:, None, None, :]  # N,1,1,D
    # compute weighted sum
    accum = np.tensordot(weights_norm, id_arr, axes=([0], [0]))  # H,W,D
    accum = np.transpose(accum, (2,0,1))  # D,H,W
    return torch.from_numpy(accum.astype(np.float32))


def test_identity_accumulation_and_proto_ce():
    H, W = 32, 32
    # 2 objects, create anchors left and right
    anchors_xy = np.array([[8, 8], [8, 24], [24, 8], [24, 24]], dtype=np.float32)
    # assign first two anchors to object 0, last two to object 1
    N = anchors_xy.shape[0]
    D = 4
    id_vecs = np.zeros((N, D), dtype=np.float32)
    id_vecs[0:2, 0] = 1.0  # prototype along dim0
    id_vecs[2:4, 1] = 1.0  # prototype along dim1

    identity_render = synth_identity_render(anchors_xy, id_vecs, H=H, W=W, sigma=3.0)

    # Build masks by simple nearest-anchor argmax (per-pixel largest contributing anchor)
    # Recompute weights to find argmax anchor per pixel
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    coords = np.stack([yy, xx], axis=-1)[None, ...]
    anchors = anchors_xy[:, None, None, :]
    diff = coords - anchors
    dist2 = np.sum(diff**2, axis=-1)
    weights = np.exp(-dist2 / (2.0 * (3.0**2)))
    top_anchor = np.argmax(weights, axis=0)  # H,W -> anchor index

    # masks per object: object 0 if top_anchor in [0,1]
    mask0 = np.isin(top_anchor, [0,1])
    mask1 = np.isin(top_anchor, [2,3])
    masks = torch.from_numpy(np.stack([mask0, mask1], axis=0))

    # compute prototypes and CE loss
    loss, meta = identity_2d_prototype_ce_loss(identity_render, masks, temperature=0.07, ignore_background=-1)
    # Expect low loss since id vectors align well
    assert loss.item() < 0.5, f'Identity CE loss too high: {loss.item()}'


def test_identity_3d_smoothness():
    # create simple anchor ids with small differences -> low smoothness loss
    N = 16
    D = 8
    center = torch.randn(D)
    anchor_ids = center.unsqueeze(0).repeat(N,1) + 0.01 * torch.randn(N, D)
    xyz = torch.randn(N, 3)
    loss = identity_3d_smoothness_loss(anchor_ids, xyz, k=4, sigma=0.1)
    assert loss.item() >= 0.0


if __name__ == '__main__':
    test_identity_accumulation_and_proto_ce()
    test_identity_3d_smoothness()
    print('Tests passed')
