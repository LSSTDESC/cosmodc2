#!/bin/sh
export EMAIL=kovacs@anl.gov
echo "Submitting jobs for snaphot ${1}"

#qsub -n 15 -t 00:10:00 -A ExtCosmology_2 -M ${EMAIL} ./bundle_cosmodc2_snapshot_blocks.sh ${1}
qsub -n 15 -t 02:00:00 -A ExtCosmology_2 -M ${EMAIL} ./bundle_cosmodc2_snapshot_blocks.sh ${1}
