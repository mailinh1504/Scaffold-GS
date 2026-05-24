"""
Generate SAM masks for a folder of images, compute per-mask appearance embeddings
(using CLIP on masked crop), and cluster masks across views into object IDs.
Saves: per-view mask PNGs, per-mask metadata (score, bbox, embedding) and a
global JSON mapping mask -> object_id.

Usage:
    python tools/generate_sam_masks.py --images_dir path/to/images --out_dir out/sam_masks

Dependencies:
    pip install git+https://github.com/facebookresearch/segment-anything.git
    pip install git+https://github.com/openai/CLIP.git
    pip install scikit-learn pillow numpy

Notes:
- This script crops each mask's bounding box, applies the mask to the crop,
  resizes to CLIP input and gets a CLIP embedding for that masked crop.
- Clustering across all mask embeddings uses DBSCAN (cosine distance).
"""

import os
import argparse
import json
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from tqdm import tqdm

# SAM
from segment_anything import sam_model_registry, SamPredictor

# CLIP
import clip

# clustering
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize


def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)


def load_image(path):
    return Image.open(path).convert('RGB')


def mask_to_bbox(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    return [int(x0), int(y0), int(x1), int(y1)]


def masked_crop_image(img, mask, bbox):
    x0, y0, x1, y1 = bbox
    crop = img.crop((x0, y0, x1+1, y1+1)).copy()
    mask_crop = mask[y0:y1+1, x0:x1+1]
    # apply mask: set background to black
    arr = np.array(crop)
    arr[mask_crop==0] = 0
    return Image.fromarray(arr)


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load SAM
    sam = sam_model_registry[args.sam_variant](checkpoint=args.sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    # Load CLIP
    clip_model, clip_preprocess = clip.load(args.clip_model, device=device, jit=False)
    clip_model.eval()

    images = sorted([p for p in Path(args.images_dir).iterdir() if p.suffix.lower() in ['.jpg', '.png', '.jpeg']])

    ensure_dir(args.out_dir)
    masks_dir = Path(args.out_dir) / 'masks'
    ensure_dir(masks_dir)

    metadata = []

    for img_idx, img_path in enumerate(tqdm(images, desc='Images')):
        img = load_image(img_path)
        predictor.set_image(np.array(img))

        # Run SAM prediction: use automatic boxes via grid prompts or multimask
        masks, scores, logits = predictor.predict(point_coords=None, point_labels=None, box=None, multimask_output=True)
        # masks: (n_masks, H, W)

        for midx in range(masks.shape[0]):
            mask = masks[midx].astype(np.uint8)
            score = float(scores[midx]) if scores is not None else 0.0

            bbox = mask_to_bbox(mask)
            if bbox is None:
                continue

            # save mask png
            mask_fname = f"{img_idx:06d}_mask_{midx:03d}.png"
            mask_path = masks_dir / mask_fname
            Image.fromarray((mask*255).astype(np.uint8)).save(mask_path)

            # compute CLIP embedding on masked crop
            try:
                crop = masked_crop_image(img, mask, bbox)
                crop_input = clip_preprocess(crop).unsqueeze(0).to(device)
                with torch.no_grad():
                    emb = clip_model.encode_image(crop_input)
                    emb = emb.cpu().numpy()[0]
                    emb = emb / (np.linalg.norm(emb) + 1e-9)
            except Exception as e:
                print('CLIP embedding failed for', mask_path, e)
                emb = np.zeros((512,), dtype=np.float32)

            # save metadata entry
            meta = {
                'image_index': int(img_idx),
                'image_path': str(img_path),
                'mask_path': str(mask_path),
                'mask_index': int(midx),
                'score': float(score),
                'bbox': bbox,
                'embedding': emb.tolist(),
            }
            metadata.append(meta)

    # cluster embeddings across all masks
    print('Clustering', len(metadata), 'masks with DBSCAN (cosine)')
    if len(metadata) == 0:
        print('No masks found.')
        return

    embs = np.array([m['embedding'] for m in metadata])
    # normalize already performed, but ensure
    embs = normalize(embs, axis=1)

    clustering = DBSCAN(eps=args.dbscan_eps, min_samples=args.dbscan_min_samples, metric='cosine')
    labels = clustering.fit_predict(embs)

    # attach labels and save global mapping
    for i, lbl in enumerate(labels):
        metadata[i]['object_id'] = int(lbl)  # -1 means noise

    # save metadata json
    meta_path = Path(args.out_dir) / 'masks_metadata.json'
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print('Saved masks and metadata to', args.out_dir)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--images_dir', required=True)
    p.add_argument('--out_dir', required=True)
    p.add_argument('--sam_variant', default='vit_h', help='vit_h | vit_l | vit_b')
    p.add_argument('--sam_checkpoint', required=True, help='path to sam checkpoint .pth')
    p.add_argument('--clip_model', default='ViT-B/32')
    p.add_argument('--dbscan_eps', type=float, default=0.25)
    p.add_argument('--dbscan_min_samples', type=int, default=1)
    args = p.parse_args()

    main(args)
