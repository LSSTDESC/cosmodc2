#!/bin/sh

if [ "$#" -lt 2 ]
then
echo "Bundle jobs to selected nodes (6 pixels per node)"
echo "Usage: bundle_skysim5000_hpx_z hpx_group (0-11) z_range (0-2)"
exit
else
hpx_group=${1}
z_range=${2}
echo "hpx_group=${hpx_group}"
echo "z_range=${z_range}"
fi

NODES=`cat $COBALT_NODEFILE | wc -l`
PROCS=1
npix=1
# starting pixel number (1 is start of file)
cd /gpfs/mira-home/ekovacs/cosmology/DC2/cosmoDC2/OR_BaseDC2_5000_v0.1
export PYTHONDIRS=/gpfs/mira-home/ekovacs/.local/lib/python2.7/site-packages:/gpfs/mira-home/ekovacs/cosmology/cosmodc2:/gpfs/mira-home/ekovacs/cosmology/galsampler/build/lib.linux-x86_64-2.7:/gpfs/mira-home/ekovacs/cosmology/halotools/build/lib.linux-x86_64-2.7
PYTHONPATH=/gpfs/mira-home/ekovacs/.local/lib/python2.7/site-packages:/gpfs/mira-home/ekovacs/cosmology/cosmodc2:/gpfs/mira-home/ekovacs/cosmology/galsampler/build/lib.linux-x86_64-2.7:/gpfs/mira-home/ekovacs/cosmology/halotools/build/lib.linux-x86_64-2.7
export PYTHONPATH
#script_name="run_cosmoDC2_healpix_production.py"
vprod="5000_production_v0.1"
#xtra_args="-input_master_dirname cosmology/DC2/OR_Production -output_mock_dirname baseDC2_5000_v1.1.1 -gaussian_smearing 0.03 -nside 32 -ndebug_snaps 1"
xtra_args="-input_master_dirname cosmology/DC2/OR_5000 -output_mock_dirname baseDC2_5000_v1.1.1 -gaussian_smearing 0.03 -nside 32"
filename="cutout"
tot_pix_grp=16
if [ "$hpx_group" -lt "$tot_pix_grp" ]
then
# 128 pixels per file
total_pix_num=129
else
# 74 pixels per file
total_pix_num=75
fi
echo "total_pix_num=${total_pix_num}"

script_name=run_cosmoDC2_healpix_production.py
pythonpath=/soft/libraries/anaconda-unstable/bin/python

readarray nodenumbers < $COBALT_NODEFILE
for nodenumber in "${nodenumbers[@]}"
#for nodenumber in {1..3}
do
  hostname1=$nodenumber
  #hostname1=expr xargs $nodenumber
  #hostname1=${ xargs $nodenumber}
  hostname1=${hostname1%?}
  #echo $hostname1
  #hostname1=$(cat $COBALT_NODEFILE | awk 'NR=='${nodenumber})
  for pixnumber in {1..4}
  do
  if [ "$npix" -lt "$total_pix_num" ]
  then
  pixelname=$(cat pixels_${hpx_group}.txt | awk 'NR=='${npix})
  echo $pixelname
  echo "${pixelname}_${z_range}" >> started_pixels_${hpx_group}.txt
  filename2=${filename}_${pixelname}.hdf5
  args="${filename2} -zrange_value ${z_range} ${xtra_args}"
  #echo $args
  #   mpirun --host ${hostname1}
  #echo ${hostname1}_${COBALT_JOBID}_${pixelname}-err.log
  jobname="cutout_${pixelname}_z_${z_range}_${vprod}_${hostname1}_${COBALT_JOBID}.log"
  mpirun --host ${hostname1} -n $PROCS $pythonpath $script_name ${args} > $jobname 2>&1 &
  #mpirun --host ${hostname1} -n $PROCS $pythonpath $script_name ${args} > "${jobname}_1.log" 2>&1 
  npix=$(expr $npix + 1 )
  else
  echo "$npix > maximum number of pixels"
  fi
  done
done
echo "$npix $z_range" >> restart_npix_${hpx_group}.txt
wait
