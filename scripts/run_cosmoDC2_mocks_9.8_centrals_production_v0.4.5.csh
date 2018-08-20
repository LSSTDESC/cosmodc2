#!/bin/csh
#
if ($#argv < 1 ) then
    echo 'Usage: run_cosmoDC2_mocks filename z_range [mode] [timelimit]'
    echo 'Script runs or submits job for creating cosmoDC2 healpix mock'
    echo 'filename = filename of healpix cutout (cutout_xxx) to run (.hdf5 assumed)'
    echo 'z_range = z-range value to process [0, 1, 2, all]'
    echo 'mode = test or qsub (default) or qtest'
    echo 'timelimit = timelimit for qsub mode (default = 5 minutes)'
    echo 'qsub runs production script in batch mode'
    echo 'test runs production script interactively'
    echo 'qtest runs production script in batch mode with -h option'
    echo 'output is written to the submit directory'
    exit
endif

set jobname = "${1}_z_${2}"
set filename = "${1}.hdf5"
set z_range = ${2}
set mode = "qsub"
set timelimit = "5"
if ($#argv > 2 ) then
    if(${3} == "qsub" || ${3} == "test" || ${3} == "qtest") then
	set mode = "${3}"
    else
	set timelimit = "${3}"
    endif
    if ($#argv > 3 ) then
	if(${4} == "qsub" || ${4} == "test" || ${4} == "qtest") then
	    set mode = "${4}"
	else
	    set timelimit = "${4}"
	endif
    endif
endif

set script_name = "run_cosmoDC2_healpix_production.py"
set python = "/soft/libraries/anaconda-unstable/bin/python"
set xtra_args = "-input_master_dirname cosmology/DC2/OR_Production -output_mock_dirname baseDC2_min_9.8_centrals_v0.4.5 -gaussian_smearing 0.03"
set xtra_label = "9.8_centrals_production_v0.4.5"
if(${xtra_label} != "") then
    set jobname = ${jobname}_${xtra_label}
endif
set args = "${filename} -zrange_value ${z_range} ${xtra_args}"

set pythondirs = /gpfs/mira-home/ekovacs/.local/lib/python2.7/site-packages:/gpfs/mira-home/ekovacs/cosmology/cosmodc2:/gpfs/mira-home/ekovacs/cosmology/galsampler/build/lib.linux-x86_64-2.7:/gpfs/mira-home/ekovacs/cosmology/halotools/build/lib.linux-x86_64-2.7

setenv PYTHONPATH ${pythondirs}
echo "Using PYTHONPATH $PYTHONPATH"

if(${mode} == "test") then
    echo "Running ${script_name} interactively to create ${filename} for z-range ${z_range} in ${mode} mode"
    ${python} ${script_name} ${args}
else
    if(${mode} == "qtest") then
	echo "Running ${script_name} -h in ${mode} mode"
	qsub -n 1 -t 5 -A ExtCosmology_2 -O ${jobname}.\$jobid --env PYTHONPATH=${pythondirs} ${python} ./${script_name} -h
    else
	echo "Running ${script_name} to create ${filename} for z-range ${z_range} in ${mode} mode with time limit of ${timelimit} minutes"
	qsub -n 1 -t ${timelimit} -A ExtCosmology_2 -O ${jobname}.\$jobid --env PYTHONPATH=${pythondirs} ${python} ./${script_name} ${args}
    endif
endif
