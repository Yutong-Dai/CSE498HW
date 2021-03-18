#!/bin/tcsh
#SBATCH --partition=hawkgpu
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:2
#SBATCH --job-name adv-cifra10-resnet18
#SBATCH --output="adv-cifra10-resnet18.%j.%N.out"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<username>@lehigh.edu

cd ${SLURM_SUBMIT_DIR}
# cd /home/yud319/cse498_s2021_053121/yud319/CSE498HW/
/home/yud319/anaconda3/bin/python 3_trainAdvResnet18CIFAR10.py


# squeue -u yud319
# sbatch run.slurm