#!/bin/sh
#SBATCH --job-name=yingshac_testing_bulk
#SBATCH --output=/home/yingshac/workspace/llms_do_math/slurm/logs/%x-%j.out
#SBATCH --err=/home/yingshac/workspace/llms_do_math/slurm/logs/%x-%j.err
#SBATCH --time=24:00:00
#SBATCH -c 2
#SBATCH --mem=4G
#SBATCH --gres=gpu:v100:1

cd /home/yingshac/workspace/llms_do_math
source venv/bin/activate
cd scripts/causal_transformer


python tester.py --handle 0423_211827 --loop 10 --test_files "val ood_test addtable_test"