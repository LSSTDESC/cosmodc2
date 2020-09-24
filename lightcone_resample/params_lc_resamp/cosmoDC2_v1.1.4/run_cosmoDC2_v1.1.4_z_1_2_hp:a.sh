#!/bin/bash 

#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --job-name=1.1.4_1_2
#SBATCH -o params_lc_resamp/protoDC2_v1.1.4/logs/v1.1.4_z_1_2_%j.out
#SBATCH -e params_lc_resamp/protoDC2_v1.1.4/logs/v1.1.4_z_1_2_%j.err



./lc_resample.py params_lc_resamp/cosmoDC2_v1.1.4/cosmoDC2_v1.1.4_z_1_2_hp:a.param
