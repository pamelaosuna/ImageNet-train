#!/bin/bash

#SBATCH --job-name=imagenet
#SBATCH --output=out_sbatch/%j.out
#SBATCH --partition=gpu2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=40G
#SBATCH --time=3-00:00:00

echo "load conda environment"
eval "$(/scratch/dldevel/osuna/miniconda3/bin/conda shell.bash hook)"
conda activate hug
which python3
echo "loaded conda environment"
echo "start extraction..."
date

python -u src/preprocess_data/extract_subset.py \
 --source "data_ImageNet/wds-imagenet/val*.tar" \
 --output-dir data_ImageNet/sub50_imagenet/val/ \
 --class-list src/sub50_imagenet_labels.txt \
 --split val \
 --imagenet-mapping data_ImageNet/imagenet1000_idx_to_wnid.txt
