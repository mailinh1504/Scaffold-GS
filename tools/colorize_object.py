#!/usr/bin/env python3
"""
Colorize a 3D object by modifying anchor features in a saved checkpoint.

Usage examples:
  # add warm red tint to object id 2
  python tools/colorize_object.py --ckpt out/chkpnt1000.pth --object-id 2 --rgb 0.1 0.0 0.0 --scale 1.0 --out out/chkpnt1000_colorized.pth

Behavior:
- Loads checkpoint (model_params, iter).
- Finds anchors belonging to given object id (via prototypes or argmax on anchor_id).
- Adds a small delta to the anchors' feature vector (`_local` / anchor_feat) to bias color MLP outputs.
- Saves modified checkpoint.

Notes:
- This is a heuristic: exact color change depends on `mlp_color` weights. Use small `--scale` and preview.
- For per-anchor precise color control, consider extending the model to store per-anchor color offsets.
"""

import argparse
import os
import numpy as np
import torch


def load_checkpoint(path):
    data = torch.load(path, map_location='cpu')
    model_params = data[0]
    it = data[1] if len(data) > 1 else None
    return model_params, it


def save_checkpoint(model_params, it, outpath):
    torch.save((model_params, it), outpath)


def compute_assignments_by_prototypes(anchor_ids, prototypes):
    if isinstance(anchor_ids, torch.Tensor):
        aids = anchor_ids.cpu().numpy()
    else:
        aids = anchor_ids
    an = aids / (np.linalg.norm(aids, axis=1, keepdims=True) + 1e-9)
    pn = prototypes / (np.linalg.norm(prototypes, axis=1, keepdims=True) + 1e-9)
    sims = an @ pn.T
    labels = sims.argmax(axis=1)
    return labels, sims


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--object-id', type=int, required=True)
    p.add_argument('--prototype-npy', default=None)
    p.add_argument('--method', choices=['prototype','argmax'], default='prototype')
    p.add_argument('--rgb', nargs=3, type=float, default=[0.1, 0.0, 0.0], help='RGB color offset to add (in [-1,1])')
    p.add_argument('--scale', type=float, default=1.0, help='Scale multiplier for feature-space offset')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()

    model_params, it = load_checkpoint(args.ckpt)

    # anchor_id at last position
    try:
        anchor_id = model_params[-1]
    except Exception:
        print('Cannot find anchor_id in checkpoint')
        return

    if isinstance(anchor_id, torch.Tensor):
        anchor_id_np = anchor_id.detach().cpu().numpy()
    else:
        anchor_id_np = np.array(anchor_id)

    if args.method == 'prototype' and args.prototype_npy is not None:
        prototypes = np.load(args.prototype_npy)
        labels, sims = compute_assignments_by_prototypes(anchor_id_np, prototypes)
        selected = (labels == args.object_id)
    else:
        labels = anchor_id_np.argmax(axis=1)
        selected = (labels == args.object_id)

    N = selected.sum()
    print(f'Anchors assigned to object {args.object_id}: {N} / {anchor_id_np.shape[0]}')
    if N == 0:
        print('No anchors selected; aborting')
        return

    if args.dry_run:
        return

    # anchor feature (called _local in capture order, index 3)
    feat_idx = 3
    anchor_feat = model_params[feat_idx]
    if isinstance(anchor_feat, torch.Tensor):
        feat_np = anchor_feat.detach().cpu().numpy()
    else:
        feat_np = np.array(anchor_feat)

    # create a simple mapping from RGB offset to feature-space offset by scaling first 3 dims
    # This is heuristic: it assumes part of feature maps to color; adjust `scale` to tune strength.
    rgb = np.array(args.rgb, dtype=np.float32) * float(args.scale)

    # Build delta: shape (feat_dim,) with small values in first 3 dims and zeros elsewhere
    feat_dim = feat_np.shape[1]
    delta = np.zeros((feat_dim,), dtype=np.float32)
    delta[:3] = rgb

    sel_idx = np.where(selected)[0]
    for si in sel_idx:
        feat_np[si] = feat_np[si] + delta

    # wrap back into tensor / parameter if necessary
    if isinstance(anchor_feat, torch.nn.parameter.Parameter):
        new_feat = torch.nn.Parameter(torch.tensor(feat_np, dtype=torch.float, device='cpu').requires_grad_(True))
    else:
        new_feat = torch.tensor(feat_np, dtype=torch.float)

    model_params = list(model_params)
    model_params[feat_idx] = new_feat
    model_params = tuple(model_params)

    save_checkpoint(model_params, it, args.out)
    print('Saved colorized checkpoint to', args.out)

if __name__ == '__main__':
    main()
