#!/bin/bash 

#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --job-name=#v1#.#v2#.#v3#_0_1
#SBATCH -o params_lc_resamp/protoDC2_v#v1#.#v2#.#v3#/logs/v#v1#.#v2#.#v3#_z_0_1_%j.out
#SBATCH -e params_lc_resamp/protoDC2_v#v1#.#v2#.#v3#/logs/v#v1#.#v2#.#v3#_z_0_1_%j.err

/soft/libraries/anaconda-unstable/bin/python ./lc_resample.py params_lc_resamp/skysim5000/skysim5000_v1.0.0_small_worst/skysim5000_v1.0.0_small_worst_z_0_1_hpx:grp0.param
