#!/bin/bash 

#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --job-name=1.0.0_2_3
#SBATCH -o params_lc_resamp/protoDC2_v1.0.0/logs/v1.0.0_z_2_3_%j.out
#SBATCH -e params_lc_resamp/protoDC2_v1.0.0/logs/v1.0.0_z_2_3_%j.err

/soft/libraries/anaconda-unstable/bin/python ./lc_resample.py params_lc_resamp/skysim5000_v1.0.0/skysim5000_v1.0.0_z_2_3_hp:grp1.param
