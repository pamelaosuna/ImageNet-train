#!/bin/bash

#SBATCH --job-name=imagenet
#SBATCH --output=out_sbatch/%j.out
#SBATCH --partition=gpu2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=20G
#SBATCH --time=3-00:00:00

echo "load conda environment"
eval "$(/scratch/dldevel/osuna/miniconda3/bin/conda shell.bash hook)"
conda activate hug
which python3
echo "loaded conda environment"
echo "start training..."
date

export OMP_NUM_THREADS=1 
export MKL_NUM_THREADS=1

# python src/train.py \
#  --data_dir data_ImageNet/ \
#  --workers 8 \
#  --model_name alexnet \
# #  --debug

python -u src/train_timm_subset.py \
 --train-data "data_ImageNet/sub50_imagenet/train" \
 --val-data "data_ImageNet/sub50_imagenet/val" \
 --num-workers 8 \
 --model alexnet \
 --class-list "src/sub50_imagenet_labels.txt"

# python -u src/train_timm_wds.py \
#  --train-data "data_ImageNet/wds-imagenet/train-{00000..00128}.tar" \
#  --val-data "data_ImageNet/wds-imagenet/val-{00000..00004}.tar" \
#  --num-workers 4 \
#  --model resnet50

# python -u src/train_timm.py \
#  --train-data "data_ImageNet/wds-imagenet/train-{00000..00128}.tar" \
#  --val-data "data_ImageNet/wds-imagenet/val-{00000..00004}.tar" \
#  --num-workers 4 \
#  --model vit_base_patch16_224 \
#  --epochs 300 \
#  --batch-size 512 \
#  --lr 0.001 \
#  --optimizer adamw \
#  --weight-decay 5e-4 \
#  --schedule cosine


date
echo "Job ended"