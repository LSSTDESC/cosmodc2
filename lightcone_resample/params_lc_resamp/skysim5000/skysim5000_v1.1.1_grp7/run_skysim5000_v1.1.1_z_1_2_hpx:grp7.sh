#!/bin/bash 

#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --job-name=#v1#.#v2#.#v3#_1_2
#SBATCH -o params_lc_resamp/protoDC2_v#v1#.#v2#.#v3#/logs/v#v1#.#v2#.#v3#_z_1_2_%j.out
#SBATCH -e params_lc_resamp/protoDC2_v#v1#.#v2#.#v3#/logs/v#v1#.#v2#.#v3#_z_1_2_%j.err

/soft/libraries/anaconda-unstable/bin/python ./lc_resample.py params_lc_resamp/skysim5000/skysim5000_v1.1.1_grp7/skysim5000_v1.1.1_z_1_2_hpx:grp7.param
