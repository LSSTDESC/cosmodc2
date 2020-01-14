#!/bin/sh
export EMAIL=kovacs@anl.gov

if [ "$#" -lt 1 ]
then
echo "Submit jobs for healpix group for all z ranges"
echo "Usage: submit_group hpx_group (0-11)"
exit
else
hpx_group=${1}
echo "hpx_group=${hpx_group}"
fi
#qsub -n 2 -t 00:10:00 -A ExtCosmology_2 -M ${EMAIL} -I 
#qsub -n 1 -t 00:10:00 -A ExtCosmology_2 -M ${EMAIL} ./bundle_cosmodc2_z_0.sh
qsub -n 30 -t 11:00:00 -A LastJourney -M ${EMAIL} ./bundle_skysim5000_hpx_z.sh ${hpx_group} 0
qsub -n 30 -t 11:00:00 -A LastJourney -M ${EMAIL} ./bundle_skysim5000_hpx_z.sh ${hpx_group} 1
qsub -n 30 -t 11:00:00 -A LastJourney -M ${EMAIL} ./bundle_skysim5000_hpx_z.sh ${hpx_group} 2
