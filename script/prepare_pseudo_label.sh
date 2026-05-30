#!/bin/bash
set -euo pipefail

# Usage: script/prepare_pseudo_label.sh <dataset_name> [scale]
# This wrapper runs the DEVA mask generation inside the gaussian-grouping submodule

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
        echo "Usage: $0 <dataset_name> [scale]"
        exit 1
fi

dataset_name="$1"
if [ "$#" -eq 2 ]; then
        scale="$2"
else
        scale=1
fi

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
dataset_folder="$repo_root/data/$dataset_name"

if [ ! -d "$dataset_folder" ]; then
        echo "Error: Folder '$dataset_folder' does not exist." >&2
        exit 2
fi

deva_dir="$repo_root/submodules/gaussian-grouping/Tracking-Anything-with-DEVA"
if [ ! -d "$deva_dir" ]; then
        echo "Error: Tracking-Anything-with-DEVA not found at $deva_dir" >&2
        exit 3
fi

cd "$deva_dir"

if [ "$scale" = "1" ]; then
        img_path="$repo_root/data/${dataset_name}/images"
else
        img_path="$repo_root/data/${dataset_name}/images_${scale}"
fi

out_dir="./example/output_gaussian_dataset/${dataset_name}"
mkdir -p "${out_dir}"

# colored mask for visualization check
python demo/demo_automatic.py \
    --chunk_size 4 \
    --img_path "$img_path" \
    --amp \
    --temporal_setting semionline \
    --size 480 \
    --output "${out_dir}" \
    --suppress_small_objects  \
    --SAM_PRED_IOU_THRESHOLD 0.7

if [ -d "${out_dir}/Annotations" ]; then
    mv "${out_dir}/Annotations" "${out_dir}/Annotations_color"
fi

# gray mask for training
python demo/demo_automatic.py \
    --chunk_size 4 \
    --img_path "$img_path" \
    --amp \
    --temporal_setting semionline \
    --size 480 \
    --output "${out_dir}" \
    --use_short_id  \
    --suppress_small_objects  \
    --SAM_PRED_IOU_THRESHOLD 0.7

# copy gray mask to corresponding data path
if [ -d "${out_dir}/Annotations" ]; then
    mkdir -p "$repo_root/data/${dataset_name}/object_mask"
    cp -r "${out_dir}/Annotations" "$repo_root/data/${dataset_name}/object_mask"
else
    echo "Warning: ${out_dir}/Annotations not found, skipping copy." >&2
fi

cd "$repo_root"
