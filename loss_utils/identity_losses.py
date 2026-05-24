import torch
import torch.nn as nn
import torch.nn.functional as F


def masks_to_label_map(masks: torch.Tensor, ids: torch.Tensor = None, background: int = -1):
    """
    Convert a stack of boolean masks [K, H, W] to a per-pixel label map [H, W].
    Overlaps are resolved by assigning larger masks first (area-priority).

    Args:
        masks: bool tensor [K, H, W]
        ids: optional tensor [K] with class ids; defaults to 0..K-1
        background: int label for background pixels
    Returns:
        label_map: long tensor [H, W] with values in ids or background
    """
    assert masks.dim() == 3
    K, H, W = masks.shape
    if ids is None:
        ids = torch.arange(K, device=masks.device, dtype=torch.long)

    flat = masks.reshape(K, -1).to(torch.bool)
    areas = flat.sum(dim=1)
    order = torch.argsort(areas, descending=True)

    label = torch.full((H * W,), background, dtype=torch.long, device=masks.device)
    for idx in order:
        mask_inds = flat[idx]
        label[(mask_inds) & (label == background)] = ids[idx]

    return label.view(H, W)


def compute_prototypes(identity_render: torch.Tensor, masks: torch.Tensor, eps: float = 1e-6):
    """
    Compute prototype embeddings for each mask.

    Args:
        identity_render: [D, H, W] embedding map (float)
        masks: [K, H, W] boolean mask tensor
    Returns:
        prototypes: [K, D] float tensor
    """
    assert identity_render.dim() == 3 and masks.dim() == 3
    D, H, W = identity_render.shape
    K = masks.shape[0]

    id_flat = identity_render.reshape(D, -1)  # [D, HW]
    masks_flat = masks.reshape(K, -1).to(identity_render.dtype)  # [K, HW]

    # (K, HW) @ (HW, D) -> (K, D)
    prototypes = masks_flat @ id_flat.T
    counts = masks_flat.sum(dim=1, keepdim=True)
    prototypes = prototypes / (counts + eps)
    return prototypes


def identity_2d_prototype_ce_loss(identity_render: torch.Tensor,
                                   masks: torch.Tensor,
                                   labels: torch.Tensor = None,
                                   temperature: float = 0.07,
                                   ignore_background: int = -1):
    """
    Prototype-based 2D identity loss. Builds prototypes from `masks` and
    classifies each labeled pixel by cosine similarity to those prototypes.

    Args:
        identity_render: [D, H, W]
        masks: [K, H, W] boolean masks used to create prototypes
        labels: optional [H, W] long tensor with per-pixel class id (must match prototypes order).
                If None, `masks_to_label_map` will be used to derive labels (ids=0..K-1).
        temperature: scaling for logits
        ignore_background: label value to ignore in loss

    Returns:
        loss: scalar tensor (cross-entropy over labeled pixels)
        metrics: dict with num_labeled_pixels
    """
    D, H, W = identity_render.shape
    K = masks.shape[0]

    prototypes = compute_prototypes(identity_render, masks)  # [K, D]

    if labels is None:
        labels = masks_to_label_map(masks)

    emb = identity_render.reshape(D, -1).T  # [HW, D]
    prototypes_norm = F.normalize(prototypes, dim=1)  # [K, D]
    emb_norm = F.normalize(emb, dim=1)  # [HW, D]

    logits = emb_norm @ prototypes_norm.T  # [HW, K]
    logits = logits / temperature

    target = labels.reshape(-1)
    valid = target != ignore_background
    if valid.sum() == 0:
        return torch.tensor(0.0, device=identity_render.device, requires_grad=True), {"num_labeled": 0}

    loss = F.cross_entropy(logits[valid], target[valid], reduction='mean')
    return loss, {"num_labeled": int(valid.sum().item())}


def identity_3d_smoothness_loss(anchor_ids: torch.Tensor,
                                anchors_xyz: torch.Tensor,
                                k: int = 8,
                                sigma: float = 0.1,
                                reduction: str = 'mean'):
    """
    3D smoothness regularizer on anchor identity vectors.

    L_reg = sum_{i,j in N(i)} w_ij * ||id_i - id_j||^2,
    where w_ij = exp(-dist^2 / sigma^2)

    Args:
        anchor_ids: [N, D]
        anchors_xyz: [N, 3]
        k: number of neighbors
        sigma: bandwidth for weights
    Returns:
        loss: scalar
    """
    N, D = anchor_ids.shape
    assert anchors_xyz.shape[0] == N

    if N <= 1:
        return torch.tensor(0.0, device=anchor_ids.device, requires_grad=True)

    # pairwise distances
    dist = torch.cdist(anchors_xyz, anchors_xyz)  # [N, N]
    # ignore self by setting large distance
    diag = torch.arange(N, device=anchor_ids.device)
    dist[diag, diag] = 1e9

    knn_dist, knn_idx = torch.topk(dist, k=k, largest=False)

    # gather neighbor ids
    neighbors = anchor_ids[knn_idx]  # [N, k, D]
    center = anchor_ids.unsqueeze(1)  # [N, 1, D]
    diffs = center - neighbors  # [N, k, D]
    sqnorm = (diffs ** 2).sum(dim=2)  # [N, k]

    weights = torch.exp(- (knn_dist ** 2) / (sigma ** 2 + 1e-9))  # [N, k]

    loss_per_anchor = (weights * sqnorm).sum(dim=1) / (weights.sum(dim=1) + 1e-9)

    if reduction == 'mean':
        return loss_per_anchor.mean()
    elif reduction == 'sum':
        return loss_per_anchor.sum()
    else:
        return loss_per_anchor


if __name__ == '__main__':
    # quick smoke test
    D = 8
    H = 32
    W = 32
    K = 3
    emb = torch.randn(D, H, W)
    masks = torch.zeros(K, H, W, dtype=torch.bool)
    masks[0, 5:20, 5:20] = 1
    masks[1, 10:28, 10:28] = 1
    masks[2, 0:8, 0:8] = 1

    loss2d, m = identity_2d_prototype_ce_loss(emb, masks)
    print('2D loss', loss2d.item(), m)

    N = 16
    ids = torch.randn(N, D)
    xyz = torch.randn(N, 3)
    loss3d = identity_3d_smoothness_loss(ids, xyz, k=4)
    print('3D loss', loss3d.item())
