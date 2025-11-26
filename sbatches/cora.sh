#!/bin/bash
#SBATCH -p main                  # Partition: main
#SBATCH --qos=main               # QoS: main
#SBATCH --output=/mnt/data/khosro/Graph_v2/sbatches/cora.txt
#SBATCH --job-name=cora
#SBATCH --gres=gpu:rtx6000ada:1        # Request 1 RTX 6000ada GPUs
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# Run your script 
/mnt/data/khosro/anaconda3/envs/my-env/bin/python /mnt/data/khosro/Graph_v2/run_v1/cora_v4.py


