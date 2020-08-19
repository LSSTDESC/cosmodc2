#!/bin/bash 

#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --job-name=1.0.0_0_1
#SBATCH -o params_lc_resamp/protoDC2_v1.0.0/logs/v1.0.0_z_0_1_%j.out
#SBATCH -e params_lc_resamp/protoDC2_v1.0.0/logs/v1.0.0_z_0_1_%j.err

/soft/libraries/anaconda-unstable/bin/python ./lc_resample.py params_lc_resamp/skysim5000_v1.0.0/skysim5000_v1.0.0_z_0_1_hp:grp1.param
