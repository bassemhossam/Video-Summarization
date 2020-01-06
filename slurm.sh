#!/bin/bash
#SBATCH --time=6-00:00:00  # wall-clock time limit in minutes
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=bassem.mounir.elias@gmail.com    # Where to send mail
#SBATCH --cpus-per-task=10            # number of CPU cores
#SBATCH --partition=gpu
srun ./main.py --batch-size 80 --model HACAModel_bs80_maxmode --epoch 5000 --model-type haca --log-interval 1 --captions-per-vid 20 --lr 1  --optimizer adadelta
