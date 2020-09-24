#!/bin/bash 

#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --job-name=#v1#.#v2#.#v3#_#z_range#
#SBATCH -o params_lc_resamp/protoDC2_v#v1#.#v2#.#v3#/logs/v#v1#.#v2#.#v3#_z_#z_range#_%j.out
#SBATCH -e params_lc_resamp/protoDC2_v#v1#.#v2#.#v3#/logs/v#v1#.#v2#.#v3#_z_#z_range#_%j.err



./lc_resample.py params_lc_resamp/cosmoDC2_v#v1#.#v2#.#v3#/cosmoDC2_v#v1#.#v2#.#v3#_z_#z_range#_hp:#healpix_name#.param
