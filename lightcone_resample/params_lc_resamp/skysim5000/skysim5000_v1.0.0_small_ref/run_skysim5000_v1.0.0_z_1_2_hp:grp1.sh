#!/bin/bash 

#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --job-name=1.0.0_1_2
#SBATCH -o params_lc_resamp/protoDC2_v1.0.0/logs/v1.0.0_z_1_2_%j.out
#SBATCH -e params_lc_resamp/protoDC2_v1.0.0/logs/v1.0.0_z_1_2_%j.err

/soft/libraries/anaconda-unstable/bin/python ./lc_resample.py params_lc_resamp/skysim5000_v1.0.0/skysim5000_v1.0.0_z_1_2_hp:grp1.param
