#!/bin/bash 

#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --job-name=#v1#.#v2#.#v3#_2_3
#SBATCH -o params_lc_resamp/protoDC2_v#v1#.#v2#.#v3#/logs/v#v1#.#v2#.#v3#_z_2_3_%j.out
#SBATCH -e params_lc_resamp/protoDC2_v#v1#.#v2#.#v3#/logs/v#v1#.#v2#.#v3#_z_2_3_%j.err

/soft/libraries/anaconda-unstable/bin/python ./lc_resample.py params_lc_resamp/skysim5000/skysim5000_v1.1.1_grp15/skysim5000_v1.1.1_z_2_3_hpx:grp15.param
