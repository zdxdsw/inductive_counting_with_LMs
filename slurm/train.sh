#!/bin/sh
#SBATCH --job-name=rnn_training
#SBATCH --output=/home/yingshac/workspace/llms_do_math/slurm/logs/%x-%j.out
#SBATCH --err=/home/yingshac/workspace/llms_do_math/slurm/logs/%x-%j.err
#SBATCH --time=24:00:00
#SBATCH -c 2
#SBATCH --mem=4G
#SBATCH --gres=gpu:A5000:1

cd /home/yingshac/workspace/llms_do_math
source venv/bin/activate
cd scripts/s4

echo $1
#--turnoff_accelerator
python run.py --task $1 --accelerator "--main_process_port $2"