#!/bin/bash

#this script runs in cat_name and makes scripts to be run from lightcone_resample(_copy)
python="/soft/libraries/anaconda-unstable/bin/python"
output_file_path="/gpfs/mira-fs0/projects/DarkUniverse_esp/kovacs/OR_5000"
params_parent_dir="params_lc_resamp"
cat_basename="skysim5000"
cat_minor_name=""
pixel_basename="pixels"
pixel_lists_dir="../pixel_lists"
hpx_group_name="grp"
if [ "$#" -ge 1 ]; then
    pixel_group="${1}"
    pix_suffix="_${hpx_group_name}${pixel_group}"
    echo "Running pixel group ${pixel_group}"
    clean=""
else
    pixel_group=""
    pix_suffix=""
    clean="yes"
    echo "No pixel group selected; cleaning any old files"
fi
template_param_file="template.param"
match="ran"
v1=1
v2=1
v3=1
logs="logs"
plots="plots"

#setup filenames and paths
cat_name="${cat_basename}_v${v1}.${v2}.${v3}"
if [ ! -z "${cat_minor_name}" ]; then
    cat_name="${cat_name}_${cat_minor_name}"
    pixel_basename="${pixel_basename}_${cat_minor_name}"
fi
# output directory is cat name and run directory is cat name+hpx group if supplied
out_dir="${cat_name}"
run_dir="${cat_name}${pix_suffix}"
mkdir -p ${run_dir}
if [ ! -z "${pixel_group}" ]; then
    pixel_filename="${pixel_basename}_${pixel_group}.txt"
else
    pixel_filename="${pixel_basename}.txt"
fi
pixel_file="${pixel_lists_dir}/${pixel_filename}"
readarray -t healpix_groups < ${pixel_file}
#for j in "${!healpix_groups[@]}";do
#    echo "$j ${healpix_groups[j]}" 
#done
output_dir=${output_file_path}/${cat_name}

echo "Creating scripts for producing ${run_dir} using pixels in ${pixel_filename}"
echo "Pixels to be run: `cat ${pixel_file}`"
echo "Logs and plots saved in ${output_dir}"

run_filename="${run_dir}/run_${healpix_group_name}"
submit_filename="${run_dir}/submit_${healpix_group_name}"
run_all_file="${run_dir}/run_all.sh"
submit_all_file="${run_dir}/submit_all.sh"

# path for output scripts
params_dir=${params_parent_dir}/${cat_basename}/${run_dir}
#path for logs 
logdir="${output_dir}/${logs}"
logdir_old="${output_dir}/${logs}/old"
mkdir -p ${logdir}
#path for plots
plotdir="${output_dir}/${plots}"
mkdir -p ${plotdir}

#clean
if [ ! -z "${clean}" ]; then

    if [ ! -z "`ls -p ${logdir} | grep -v /`" ]; then
	mkdir -p ${logdir_old}
	echo "Archiving old log files"
	mv ${logdir}/*.err ${logdir_old}/.
	mv ${logdir}/*.out ${logdir_old}/.
	mv ${logdir}/*.log ${logdir_old}/.
	mv ${logdir}/*.cobalt ${logdir_old}/.
    else
	echo "No old logfiles to archive"
    fi

    echo "Erasing all ${run_dir}/*${cat_name}* and *all.sh files"
    rm ${run_dir}/*${cat_name}*
    rm ${run_dir}/*_all.sh
else
    echo "Not moving earlier logfiles"
fi

# parameters for z ranges and Galacticus libraries
z_ranges=("0_1" "1_2" "2_3")
steps_list=("499 487 475 464 453 442 432 421 411 401 392 382 373 365 355 347 338 331 323 315 307 300 293 286 279 272 266 259 253 247"  "247 241 235 230 224 219 213 208 203 198 194 189 184 180 176 171 167" "167 163 159 155 151 148 144 141 137 134 131 127 124 121")
gltcs_files=("/gpfs/mira-fs0/projects/DarkUniverse_esp/dkorytov/data/Galacticus/low_z/galaxy_library/\${step}_mod.hdf5" "/gpfs/mira-fs0/projects/DarkUniverse_esp/dkorytov/data/Galacticus/high_z/galaxy_library/\${step}_mod.hdf5" "/gpfs/mira-fs0/projects/DarkUniverse_esp/dkorytov/data/Galacticus/high_z/galaxy_library/\${step}_mod.hdf5")
z_types=("low_z" "high_z" "high_z")

# begin *all files if they don't exist; previous log files should have been moved
if [ ! -e ${submit_all_file} ]; then 
    echo "#!/bin/bash" >${submit_all_file}
    #echo 'if [ ! -z "`ls -p '"${logdir}"' | grep -v /`" ]; then mv '"${logdir}/*.err ${logdir}/old/.; fi" >> ${submit_all_file}
fi
if [ ! -e ${run_all_file} ]; then 
    echo "#!/bin/bash" >${run_all_file}
    #echo 'if [ ! -z "`ls -p '"${logdir}"' | grep -v /`" ]; then mv '"${logdir}/*.out ${logdir}/old/.; fi" >> ${run_all_file}
fi

for i in "${!z_ranges[@]}";do
    #echo "z ${i}"
    z_range=${z_ranges[$i]}
    steps=${steps_list[$i]}
    gltcs_file=${gltcs_files[$i]}
    z_type=${z_types[$i]}
    plotdir_z="${plotdir}/z_${z_range}"
    echo "z_range:${z_range}"
    echo "steps:${steps}"
    echo "glcts:${gltcs_file}"
    echo "z_type:${z_type}"
    echo "plotdir:${plotdir_z}"
    echo "select_match:${match}"
    for j in "${!healpix_groups[@]}";do
	#auto assign pixel group or use input value (assumes only 1 group will be input)
	if [ ! -z "${pixel_group}" ]; then
	    group_name="${hpx_group_name}${pixel_group}"
	else
	    group_name="${hpx_group_name}${j}"
	fi
	#get rid of leading and trailing extra quotes in healpix_groups
	healpix_group=`sed -e 's/^"//' -e 's/"$//' <<<${healpix_groups[j]}`
        #echo ${healpix_groups[j]}
	echo "healpix group $group_name count: `echo ${healpix_group} | wc -w`"
	filename="${cat_name}_z_${z_range}_hpx:${group_name}"
	param_file="${run_dir}/${filename}.param"
	run_file="${run_dir}/run_${filename}.sh"
	#plot subdirectories
	plotdir_z_grp="${plotdir_z}/${group_name}"
	mkdir -p ${plotdir_z_grp}
	if [ ! -z "${clean}" ]; then
	    if [ ! -z "`ls -p ${plotdir_z_grp} | grep -v /`" ]; then
		plotdir_z_grp_old="${plotdir_z_grp}/old"
		mkdir -p ${plotdir_z_grp_old}
		echo "Archiving old plot files in ${plotdir_z_grp}"
		mv ${plotdir_z_grp}/*.png ${plotdir_z_grp_old}/.
	    else
		echo "No old files in ${plotdir_z_grp} to archive"
	    fi
	fi
	#group submit file
	submit_file="${run_dir}/submit_${cat_name}_hpx:${group_name}.sh"
	if [ ! -e ${submit_file} ]; then
	    echo "#!/bin/bash" >${submit_file}
	fi
	sed "s/#z_range#/${z_range}/g; s/#step_list#/${steps}/g; s@#gltcs_file#@${gltcs_file}@g; s/#z_type#/${z_type}/g; s/#healpix_group#/${healpix_group}/g; s%#path_to_plotdir#%${plotdir_z_grp}%g; s/#match#/${match}/g; s/#out_dir#/${out_dir}/g; s/#v1#/${v1}/g; s/#v2#/${v2}/g; s/#v3#/${v3}/g"<${template_param_file} > ${param_file}
	sed "s%#python#%${python}%g; s/#z_range#/${z_range}/g; s/#healpix_name#/${group_name}/g; s%#param_file_path#%${params_dir}%g; s/#cat_name#/${cat_name}/g"<template.sh > ${run_file}
	chmod +x ${run_file}
	log_fname=${logdir}/${z_range}_${group_name}
	echo "qsub -n 1 -t 720 -A LastJourney -o ${log_fname}.out -e ${log_fname}.err  --debuglog=${log_fname}.cobalt  `tail -1 $run_file`" >> ${submit_all_file}
	echo "qsub -n 1 -t 720 -A LastJourney -o ${log_fname}.out -e ${log_fname}.err  --debuglog=${log_fname}.cobalt  `tail -1 $run_file`" >> ${submit_file}
	echo "`tail -1 $run_file`" >> ${run_all_file} 
    done
done
chmod +x ${submit_all_file}
chmod +x ${run_all_file}
