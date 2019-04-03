#!/bin/sh
export EMAIL=kovacs@anl.gov
echo "Submitting jobs for snaphot ${1}"

#qsub -n 1 -t 00:10:00 -A ExtCosmology_2 -M ${EMAIL} ./bundle_cosmodc2_z_0.sh
qsub -n 8 -t 02:00:00 -A ExtCosmology_2 -M ${EMAIL} ./bundle_cosmodc2_snapshot_blocks.sh ${1}
