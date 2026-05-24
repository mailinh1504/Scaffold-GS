#!/usr/bin/env python3
"""
Remove a 3D object by modifying gaussians checkpoint anchor opacities.

Usage examples:
  python tools/remove_object.py --ckpt out/chkpnt1000.pth --object-id 3 --out out/chkpnt1000_removed.pth
  python tools/remove_object.py --ckpt out/chkpnt1000.pth --prototype-npy masks/prototypes.npy --object-id 2 --out out/chkpnt1000_removed.pth

Behavior:
- Loads a checkpoint saved as (model_params, iter) where `model_params` is the tuple returned by `GaussianModel.capture()`.
- Determines anchors belonging to object `object_id` via either:
  * nearest-prototype matching (if --prototype-npy provided), or
  * argmax over anchor id vector (default fallback).
- Sets the anchors' opacity parameter to a near-zero value (so after sigmoid -> ~0).
- Saves modified checkpoint to --out (overwrites if exists).

Notes:
- This edits checkpoint file only; it does not run training or rendering.
- Prototype matching uses cosine similarity between anchor ids and provided prototypes.
"""

import argparse
import os
import math
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
    # anchor_ids: (N, D) numpy or tensor
    # prototypes: (K, D) numpy
    if isinstance(anchor_ids, torch.Tensor):
        aids = anchor_ids.cpu().numpy()
    else:
        aids = anchor_ids
    # normalize
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
    p.add_argument('--prototype-npy', default=None, help='optional prototypes npy file (K,D) to match anchors')
    p.add_argument('--method', choices=['prototype','argmax'], default='prototype')
    p.add_argument('--opacity', type=float, default=1e-4, help='final opacity value to set for removed anchors (in [0,1])')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()

    model_params, it = load_checkpoint(args.ckpt)

    # model_params tuple expected layout (see scene.gaussian_model.restore):
    # (active_sh_degree, _anchor, _offset, _local/_anchor_feat, _scaling, _rotation, _opacity, max_radii2D, denom, opt_dict, spatial_lr_scale, anchor_id)
    try:
        anchor_id = model_params[-1]
    except Exception:
        print('Checkpoint format unexpected: cannot find anchor_id at last position')
        return

    # ensure numpy
    if isinstance(anchor_id, torch.Tensor):
        anchor_id_np = anchor_id.detach().cpu().numpy()
    else:
        anchor_id_np = np.array(anchor_id)

    N = anchor_id_np.shape[0]

    if args.method == 'prototype' and args.prototype_npy is not None:
        prototypes = np.load(args.prototype_npy)
        labels, sims = compute_assignments_by_prototypes(anchor_id_np, prototypes)
        selected = (labels == args.object_id)
    else:
        # fallback: argmax along id vector
        labels = anchor_id_np.argmax(axis=1)
        selected = (labels == args.object_id)

    print(f'Anchors found for object {args.object_id}: {selected.sum()} / {N}')

    if args.dry_run:
        return

    # locate opacity entry in tuple
    # per restore ordering, opacity is at index 6
    opacity_idx = 6
    old_opacity = model_params[opacity_idx]
    # convert to tensor
    if not isinstance(old_opacity, torch.Tensor):
        old_opacity_t = torch.tensor(old_opacity)
    else:
        old_opacity_t = old_opacity.clone()

    # compute inverse-sigmoid (logit) for target opacity p
    pval = float(args.opacity)
    pval = min(max(pval, 1e-6), 1.0 - 1e-6)
    logit = math.log(pval / (1.0 - pval))

    # set selected anchors' opacity parameter to logit value
    # old_opacity_t shape expected [N, 1]
    try:
        old_opacity_t = old_opacity_t.clone()
        sel_idx = np.where(selected)[0]
        if old_opacity_t.ndim == 2:
            for si in sel_idx:
                old_opacity_t[si, 0] = logit
        else:
            for si in sel_idx:
                old_opacity_t[si] = logit
    except Exception as e:
        print('Failed to set opacity:', e)
        return

    # write back into model_params (must preserve types)
    # if original was a torch Parameter, we wrap into tensor
    if isinstance(old_opacity, torch.nn.parameter.Parameter):
        new_opacity = torch.nn.Parameter(old_opacity_t.requires_grad_(True))
    else:
        new_opacity = old_opacity_t

    model_params = list(model_params)
    model_params[opacity_idx] = new_opacity
    model_params = tuple(model_params)

    # save
    save_checkpoint(model_params, it, args.out)
    print('Saved modified checkpoint to', args.out)


if __name__ == '__main__':
    main()
