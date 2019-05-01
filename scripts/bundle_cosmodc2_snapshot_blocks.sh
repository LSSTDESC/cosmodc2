#!/bin/sh
NODES=`cat $COBALT_NODEFILE | wc -l`
PROCS=1
# starting block number 
nblock=0
# setup
cd /gpfs/mira-home/ekovacs/cosmology/DC2/cosmoDC2/OR_Snapshots
export PYTHONDIRS=/gpfs/mira-home/ekovacs/.local/lib/python2.7/site-packages:/gpfs/mira-home/ekovacs/cosmology/cosmodc2:/gpfs/mira-home/ekovacs/cosmology/galsampler/build/lib.linux-x86_64-2.7:/gpfs/mira-home/ekovacs/cosmology/halotools/build/lib.linux-x86_64-2.7
PYTHONPATH=/gpfs/mira-home/ekovacs/.local/lib/python2.7/site-packages:/gpfs/mira-home/ekovacs/cosmology/cosmodc2:/gpfs/mira-home/ekovacs/cosmology/galsampler/build/lib.linux-x86_64-2.7:/gpfs/mira-home/ekovacs/cosmology/halotools/build/lib.linux-x86_64-2.7
export PYTHONPATH
vsnap="snapshots_v0.1"
xtra_args="-input_master_dirname cosmology/DC2/OR_Snapshots -output_mock_dirname baseDC2_${vsnap}"
snapshot=${1}
echo "Running snapshot ${1}"
total_block_num=1200
blocks_per_node=80
blocks_per_core=20
#change filenumber range below to match jobs_per_core - 1
jobs_per_node=4

script_name=run_cosmoDC2_snapshot_production.py
pythonpath=/soft/libraries/anaconda-unstable/bin/python

readarray nodenumbers < $COBALT_NODEFILE
subgrp=0
for nodenumber in "${nodenumbers[@]}"
#for nodenumber in {1..3}
do
  hostname1=$nodenumber
  hostname1=${hostname1%?}
  #hostname1=$(cat $COBALT_NODEFILE | awk 'NR=='${nodenumber})
  #filemax=$(expr $jobs_per_core - 1)
  for filenumber in {0..3}
  do
  if [ "$nblock" -lt "$total_block_num" ]
  then
  sublo=$(expr $subgrp + $filenumber \* $blocks_per_core )
  subhi=$(expr $sublo + $blocks_per_core - 1 )
  blockrange="$sublo-$subhi"
  echo "$blockrange"
  echo "${blockrange}" >> started_blocks_${snapshot}.txt
  args="${snapshot} -blocks ${blockrange} ${xtra_args}"
  #echo $args
  #   mpirun --host ${hostname1}
  #echo ${hostname1}_${COBALT_JOBID}_${blockelname}-err.log
  jobname="Step${snapshot}_${blockrange}_${vsnap}_${hostname1}_${COBALT_JOBID}.log"
  mpirun --host ${hostname1} -n $PROCS $pythonpath $script_name ${args} > $jobname 2>&1 & 
  #echo "mpirun --host ${hostname1} -n $PROCS $pythonpath $script_name ${args}" > $jobname 2>&1 &
  nblock=$(expr $nblock + $blocks_per_core)
  else
  echo "$nblock > maximum number of blocks"
  fi
  done
  subgrp=$(expr $subgrp + $blocks_per_node)
done
echo "$nblock" >> restart_nblock_${snapshot}.txt
wait
