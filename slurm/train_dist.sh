#!/bin/bash

#SBATCH --job-name=alexnet
#SBATCH --output=out_sbatch/%j.out
#SBATCH --partition=gpu2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=40G
#SBATCH --time=3-00:00:00

echo "load conda environment"
eval "$(/scratch/dldevel/osuna/miniconda3/bin/conda shell.bash hook)"
conda activate hug
which python3
echo "loaded conda environment"
echo "start training..."
date

# optional: avoid thread oversubscription
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

torchrun --standalone --nnodes=1 --nproc_per_node=2 src/train.py \
 --data_dir data_ImageNet/ \
 --model_subdir alexnet_42 \
 --seed 42 \
 --workers 4 #\
#  --debug


date
echo "Job ended"