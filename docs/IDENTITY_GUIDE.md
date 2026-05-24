Identity supervision & editing guide

This document describes steps to validate and use identity renders, run unit tests, and perform object edits (removal/colorization).

Prerequisites
- Python environment with packages: torch, torchvision, numpy, scikit-learn, scipy, PIL, opencv-python.
- Optional: `segment-anything` and `clip` for SAM preprocessing.

1) Generate SAM masks (offline)
- Use `tools/generate_sam_masks.py` (if present) to produce per-image masks and optional CLIP prototypes.
- Output files (per image): `<image_name>_masks.npy` (K,H,W) and `<image_name>_mask_embs.npy` (K,D).
- A global `masks_metadata.json` and `mask_prototypes.npy` may be written when clustering is used.

2) Verify dataset loader picks up masks
- Place mask files in the same folder as images (or in `object_mask/` subfolder).
- `scene.dataset_readers` will attach `CameraInfo.label_map` or `CameraInfo.objects` when files exist.

3) Run unit tests (CPU-only smoke tests)
- Tests added at `tests/test_identity_render.py` validate prototype computation, CE loss, and 3D smoothness loss using a synthetic renderer.
- Run with pytest or python:

```bash
# using pytest
pytest -q tests/test_identity_render.py

# or run directly
python -m tests.test_identity_render
```

4) Train with identity supervision
- Enable identity hyperparameters via CLI (see `arguments/`): `--id_dim 32 --lambda_id2d 3.0 --lambda_idreg 0.001 --id_warmup_start 500 --id_warmup_iters 2000`
- Example:

```bash
python train.py --model_path ./output/scene --iterations 20000 \
  --id_dim 32 --lambda_id2d 3.0 --lambda_idreg 0.001 --id_warmup_start 500 --id_warmup_iters 2000
```

- Ensure the native rasterizer is built with `NUM_CHANNELS = 3 + id_dim` if using identity channels in the CUDA extension. If not rebuilt, identity gradients may not flow to anchors.

5) Edit operations (object removal / colorization)
- Scripts:
  - `tools/remove_object.py` — quick checkpoint edit (opacity set to near-zero) using either `anchor_id.argmax()` or `--prototype_npy` matching.
  - `tools/colorize_object.py` — heuristic color offset by modifying `anchor_feat`.
  - `tools/object_removal_port.py` — ported object removal that supports convex hull expansion and optional rendering.

Example remove command:

```bash
python tools/object_removal_port.py --model_path ./output/scene --iteration 1000 --object_id 3 --prototype_npy masks/prototypes.npy --use_convex --out_dir ./output/edited --render
```

6) Validate edits
- After editing, open the saved PLY (`point_cloud_removed.ply`) in MeshLab or CloudCompare to inspect removed anchors.
- Render edited scene (scripts attempt rendering if `--render`) or run your normal evaluation to confirm object removed in outputs.

7) Troubleshooting
- If identity_render is missing or zeros, check rasterizer build and `NUM_CHANNELS` match `id_dim`.
- If gradients to `anchor_id` are zero, you may need to extend CUDA backward kernels or use Python fallback accumulation to compute identity accumulation and backprop.

Questions / Next steps
- I can add an automated CI job to run the unit tests and a small rendering smoke test (if GPU available).
- I can also add a Jupyter notebook demonstrating SAM preprocessing → training → editing pipeline step-by-step.
