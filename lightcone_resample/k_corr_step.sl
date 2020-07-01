#!/bin/bash -l

#SBATCH -N 16
#SBATCH -t 78:00:00
#SBATCH -o "runs/k_corr_step.out"
#SBATCH -e "runs/k_corr_step.err"

mpiexec -N 1 ./k_corr_step.py params_kcorr/p_all.param
