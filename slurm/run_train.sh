#!/bin/bash

#SBATCH --job-name=imagenet
#SBATCH --output=out_sbatch/%j.out
#SBATCH --partition=gpu2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=40G
#SBATCH --time=3-00:00:00

echo "load conda environment"
eval "$(/scratch/dldevel/osuna/miniconda3/bin/conda shell.bash hook)"
conda activate hug
which python3
echo "loaded conda environment"
echo "start training..."
date

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python src/train.py \
 --data_dir data_ImageNet/ \
 --workers 8 \
 --model_name alexnet
#  --debug

date
echo "Job ended"