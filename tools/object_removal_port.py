#!/usr/bin/env python3
"""
Ported object removal utility for Scaffold-GS.

Features:
- Uses per-anchor `anchor_id` vectors (or optional prototype npy) to assign anchors to object IDs.
- Expands selection using convex hull of selected anchors (optional).
- Sets selected anchors' opacity to near-zero (logit) and saves a modified checkpoint under "_object_removal/iteration_{it}".
- Optionally renders train/test views after editing.

Usage:
  python tools/object_removal_port.py --model_path ./output/... --iteration 1000 --object-id 3 --out_dir ./output/edited --proto prototypes.npy --render

Notes:
- Requires the scene and GaussianModel APIs used in Scaffold-GS. This script edits the in-memory Gaussian parameters and saves a new checkpoint and ply.
"""

import os
import argparse
import numpy as np
import torch
from scene import Scene
from scene.gaussian_model import GaussianModel
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from render import render if False else None
from gaussian_renderer import render as grender
from os import makedirs
from tqdm import tqdm
from PIL import Image
import math


def points_inside_convex_hull(point_cloud, mask, remove_outliers=True, outlier_factor=1.0):
    import numpy as np
    from scipy.spatial import Delaunay
    pts = point_cloud[mask].cpu().numpy()
    if pts.shape[0] < 4:
        # convex hull can't be computed robustly for <4 points in 3D; return mask itself
        return mask
    if remove_outliers:
        Q1 = np.percentile(pts, 25, axis=0)
        Q3 = np.percentile(pts, 75, axis=0)
        IQR = Q3 - Q1
        outlier_mask = (pts < (Q1 - outlier_factor * IQR)) | (pts > (Q3 + outlier_factor * IQR))
        pts_f = pts[~np.any(outlier_mask, axis=1)]
        if pts_f.shape[0] < 4:
            pts_f = pts
    else:
        pts_f = pts
    try:
        delaunay = Delaunay(pts_f)
        inside = delaunay.find_simplex(point_cloud.cpu().numpy()) >= 0
        return torch.tensor(inside, device=point_cloud.device)
    except Exception:
        return mask


def load_checkpoint(path):
    data = torch.load(path, map_location='cpu')
    model_params = data[0]
    it = data[1] if len(data) > 1 else None
    return model_params, it


def save_checkpoint(model_params, it, outpath):
    torch.save((model_params, it), outpath)


def assign_by_prototypes(anchor_ids, prototypes):
    # anchor_ids: (N, D) numpy
    # prototypes: (K, D) numpy
    an = anchor_ids / (np.linalg.norm(anchor_ids, axis=1, keepdims=True) + 1e-9)
    pn = prototypes / (np.linalg.norm(prototypes, axis=1, keepdims=True) + 1e-9)
    sims = an @ pn.T
    labels = sims.argmax(axis=1)
    return labels, sims


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--iteration', required=True)
    parser.add_argument('--object_id', type=int, required=True)
    parser.add_argument('--prototype_npy', default=None)
    parser.add_argument('--use_convex', action='store_true')
    parser.add_argument('--opacity', type=float, default=1e-4)
    parser.add_argument('--out_dir', default=None)
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()

    # Prepare scene and gaussians
    dataset_cfg = ModelParams(argparse.ArgumentParser(), sentinel=True)
    # minimal config: set model_path so Scene can find files
    # We'll build a temporary args namespace for get_combined_args compatibility if needed
    # Instead, directly load scene

    gaussians = GaussianModel()
    scene = Scene(dataset_cfg, gaussians, load_iteration=args.iteration, shuffle=False)

    # try to find checkpoint file
    chkpnt_path = os.path.join(args.model_path, 'chkpnt' + str(args.iteration) + '.pth')
    if not os.path.exists(chkpnt_path):
        # try scene.model_path location
        chkpnt_path = scene.model_path + '/chkpnt' + str(args.iteration) + '.pth'
        if not os.path.exists(chkpnt_path):
            print('Checkpoint not found at', chkpnt_path)
            return

    model_params, it = load_checkpoint(chkpnt_path)

    # find anchor_id in model_params (last element expected)
    anchor_id = None
    try:
        anchor_id = model_params[-1]
    except Exception:
        print('Checkpoint format unexpected (no anchor_id). Aborting.')
        return

    if isinstance(anchor_id, torch.Tensor):
        anchor_id_np = anchor_id.detach().cpu().numpy()
    else:
        anchor_id_np = np.array(anchor_id)

    N = anchor_id_np.shape[0]
    print(f'Found {N} anchors, id dim {anchor_id_np.shape[1]}')

    if args.prototype_npy is not None and os.path.exists(args.prototype_npy):
        prototypes = np.load(args.prototype_npy)
        labels, sims = assign_by_prototypes(anchor_id_np, prototypes)
        selected = (labels == args.object_id)
    else:
        # fallback argmax
        labels = anchor_id_np.argmax(axis=1)
        selected = (labels == args.object_id)

    print('Initially selected anchors:', selected.sum(), '/', N)

    # expand via convex hull if requested
    # need anchors xyz from model_params: try to find _anchor in model_params tuple
    # capture ordering in gaussian_model.capture() we patched: (active_sh_degree, _anchor, _offset, _local, _scaling, _rotation, _opacity, max_radii2D, denom, opt_dict, spatial_lr_scale, anchor_id)
    try:
        anchor_idx = 1
        anchors = model_params[anchor_idx]
        if isinstance(anchors, torch.Tensor):
            anchors_np = anchors.detach().cpu()
        else:
            anchors_np = torch.tensor(anchors)
    except Exception:
        print('Could not extract anchor positions from checkpoint tuple. Aborting convex hull expansion.')
        anchors_np = None

    if args.use_convex and anchors_np is not None:
        mask = torch.tensor(selected, device=anchors_np.device)
        hull_mask = points_inside_convex_hull(anchors_np.cuda(), mask.cuda())
        selected = np.logical_or(selected, hull_mask.cpu().numpy())
        print('After convex expansion selected anchors:', selected.sum(), '/', N)

    # modify opacity values in model_params (index 6)
    opacity_idx = 6
    old_opacity = model_params[opacity_idx]
    if isinstance(old_opacity, torch.Tensor):
        old_opacity_t = old_opacity.clone()
    else:
        old_opacity_t = torch.tensor(old_opacity)

    pval = float(args.opacity)
    pval = min(max(pval, 1e-6), 1.0 - 1e-6)
    logit = math.log(pval / (1.0 - pval))

    sel_idx = np.where(selected)[0]
    if old_opacity_t.ndim == 2:
        for si in sel_idx:
            old_opacity_t[si, 0] = logit
    else:
        for si in sel_idx:
            old_opacity_t[si] = logit

    # write back
    new_opacity = old_opacity_t
    # keep same type as before
    if isinstance(old_opacity, torch.nn.parameter.Parameter):
        new_opacity = torch.nn.Parameter(new_opacity.requires_grad_(True))

    model_params = list(model_params)
    model_params[opacity_idx] = new_opacity
    model_params = tuple(model_params)

    # save modified checkpoint
    out_dir = args.out_dir if args.out_dir is not None else os.path.join(args.model_path, '_object_removal')
    makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'chkpnt' + str(args.iteration) + '_removed.pth')
    save_checkpoint(model_params, it, out_path)
    print('Saved modified checkpoint to', out_path)

    # save ply of modified anchors if possible
    try:
        # reinstantiate GaussianModel and load from model_params via restore
        new_gauss = GaussianModel()
        new_gauss.restore(model_params, OptimizationParams(argparse.ArgumentParser()))
        ply_dir = os.path.join(out_dir, 'point_cloud')
        makedirs(ply_dir, exist_ok=True)
        new_gauss.save_ply(os.path.join(ply_dir, 'point_cloud_removed.ply'))
        print('Saved point cloud ply to', ply_dir)
    except Exception as e:
        print('Could not save ply:', e)

    # Optional render
    if args.render:
        try:
            new_scene = Scene({}, new_gauss, load_iteration='_removed', shuffle=False)
            bg_color = [1,1,1]
            background = torch.tensor(bg_color, dtype=torch.float32, device='cuda')
            # render train and test
            views_train = new_scene.getTrainCameras()
            views_test = new_scene.getTestCameras()
            for idx, view in enumerate(tqdm(views_train)):
                res = grender(view, new_gauss, None, background)
                img = res['render']
                outf = os.path.join(out_dir, f'train_render_{idx:04d}.png')
                torchvision.utils.save_image(img, outf)
            for idx, view in enumerate(tqdm(views_test)):
                res = grender(view, new_gauss, None, background)
                img = res['render']
                outf = os.path.join(out_dir, f'test_render_{idx:04d}.png')
                torchvision.utils.save_image(img, outf)
            print('Rendered images saved to', out_dir)
        except Exception as e:
            print('Rendering failed:', e)


if __name__ == '__main__':
    main()
