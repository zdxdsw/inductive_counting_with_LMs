#!/bin/sh
#SBATCH --job-name=mamba_training
#SBATCH --output=/home/yingshac/workspace/llms_do_math/slurm/logs/%x-%j.out
#SBATCH --err=/home/yingshac/workspace/llms_do_math/slurm/logs/%x-%j.err
#SBATCH --time=24:00:00
#SBATCH -c 2
#SBATCH --mem=10G
#SBATCH --gres=gpu:v100:1

scl enable gcc-toolset-11 bash && module load cuda-11.8 &&
cd /home/yingshac/workspace/mamba && source venv/bin/activate && cd ../llms_do_math/scripts/mamba

echo $1
python run.py --task $1 --accelerator "--main_process_port $2"