#!/bin/sh
export EMAIL=kovacs@anl.gov

#qsub -n 2 -t 00:10:00 -A ExtCosmology_2 -M ${EMAIL} -I 
#qsub -n 1 -t 00:10:00 -A ExtCosmology_2 -M ${EMAIL} ./bundle_cosmodc2_z_0.sh
qsub -n 33 -t 06:00:00 -A ExtCosmology_2 -M ${EMAIL} ./bundle_cosmodc2_z_2.sh
