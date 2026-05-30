#!/bin/bash
set -euo pipefail

# Usage: script/train_lerf.sh <dataset_name> [scale]
# If scale omitted, defaults to 1
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

gg_dir="$repo_root/submodules/gaussian-grouping"
if [ ! -d "$gg_dir" ]; then
    echo "Error: gaussian-grouping submodule not found at $gg_dir" >&2
    exit 3
fi

cd "$gg_dir"

# Gaussian Grouping training (with --train_split for lerf datasets)
python train.py -s "$dataset_folder" -r ${scale} -m output/${dataset_name} --config_file config/gaussian_dataset/train.json --train_split

# Segmentation rendering using trained model (images folder)
python render.py -m output/${dataset_name} --num_classes 256 --images images

cd "$repo_root"
