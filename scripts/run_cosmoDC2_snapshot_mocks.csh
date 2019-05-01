#!/bin/csh
#
if ($#argv < 1 ) then
    echo 'Usage: run_cosmoDC2_mocks timesteps blocks [mode] [timelimit]'
    echo 'Script runs or submits job for creating cosmoDC2 snapshot mock'
    echo 'timesteps = timesteps to run'
    echo 'blocks = blocks to process [0-255]'
    echo 'mode = test or qsub (default) or qtest'
    echo 'timelimit = timelimit for qsub mode (default = 5 minutes)'
    echo 'qsub runs production script in batch mode'
    echo 'test runs production script interactively'
    echo 'qtest runs production script in batch mode with -h option'
    echo 'output is written to the submit directory'
    exit
endif

set jobname = "$Step{1}_#${2}"
set timesteps = "${1}"
set blocks = ${2}
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

set script_name = "run_cosmoDC2_snapshot_production.py"
set python = "/soft/libraries/anaconda-unstable/bin/python"
set version = "0.1"
set xtra_label = "snapshots_v${version}"
set xtra_args = "-input_master_dirname cosmology/DC2/OR_Snapshots -output_mock_dirname baseDC2_${xtra_label}"
if(${xtra_label} != "") then
    set jobname = ${jobname}_${xtra_label}
endif
set args = "${timesteps} -blocks ${block} ${xtra_args}"

set pythondirs = /gpfs/mira-home/ekovacs/.local/lib/python2.7/site-packages:/gpfs/mira-home/ekovacs/cosmology/cosmodc2:/gpfs/mira-home/ekovacs/cosmology/galsampler/build/lib.linux-x86_64-2.7:/gpfs/mira-home/ekovacs/cosmology/halotools/build/lib.linux-x86_64-2.7

setenv PYTHONPATH ${pythondirs}
echo "Using PYTHONPATH $PYTHONPATH"

msg1 = "Running ${script_name}"
msg2 = "to create snapshot for Step ${timestep} for block(s) ${block} in ${mode} mode"

if(${mode} == "test") then
    echo "${msg1} interactively ${msg2}"
    ${python} ${script_name} ${args}
else
    if(${mode} == "qtest") then
	echo "${msg1} -h in ${mode} mode"
	qsub -n 1 -t 5 -A ExtCosmology_2 -O ${jobname}.\$jobid --env PYTHONPATH=${pythondirs} ${python} ./${script_name} -h
    else
	echo "${msg1} ${msg2} with time limit of ${timelimit} minutes"
	qsub -n 1 -t ${timelimit} -A ExtCosmology_2 -O ${jobname}.\$jobid --env PYTHONPATH=${pythondirs} ${python} ./${script_name} ${args}
    endif
endif
