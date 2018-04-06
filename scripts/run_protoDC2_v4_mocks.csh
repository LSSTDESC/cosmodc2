#!/bin/csh
#
if ($#argv < 1 ) then
    echo 'Usage: run_protoDC2_v4_mocks commit_hash nsnap [mode] [timelimit]'
    echo 'Script runs or submits job for creating protoDC2 mock catalogs'
    echo 'nsnap = number of snapshots to run'
    echo 'mode = test or qsub (default) or qtest'
    echo 'timelimit = timelimit for qsub mode (default = 5 minutes)'
    echo 'qsub runs production script in batch mode'
    echo 'test runs production script interactively'
    echo 'qtest runs production script in batch mode with -h option'
    echo 'output is written to the submit directory'
    exit
endif

set commit_hash = ${1}
set nsnap = ${2}
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

set script_name = "run_v4_snapshot_production.py"

#set target_halo_dirname = "/projects/DarkUniverse_esp/heitmann/OuterRim/M000/L360/HACC001/analysis/Halos/b0168/fofp_new"
#set um_dirname = "/projects/DarkUniverse_esp/kovacs/AlphaQ/protoDC2_v4_um_sfr_catalogs_and_halos"
#set output_mocks_dirname = "/projects/DarkUniverse_esp/kovacs/AlphaQ/galsampler_alphaq_outputs_v4"

set python = "/soft/libraries/anaconda/bin/python"
#set args = "${target_halo_dirname} ${um_dirname} ${output_mocks_dirname}"
set args = ${commit_hash}

set pythondirs = /gpfs/mira-home/ekovacs/.local/lib/python2.7/site-packages:/gpfs/mira-home/ekovacs/cosmology/cosmodc2:/gpfs/mira-home/ekovacs/cosmology/galsampler/build/lib.linux-x86_64-2.7:/gpfs/mira-home/ekovacs/cosmology/halotools/build/lib.linux-x86_64-2.7

setenv PYTHONPATH ${pythondirs}
echo "Using PYTHONPATH $PYTHONPATH"

if(${mode} == "test") then
    echo "Running ${script_name} interactively to create ${nsnap} snapshots in ${mode} mode"
    ${python} ${script_name} ${args} -nsnap ${nsnap}
else
    if(${mode} == "qtest") then
	echo "Running ${script_name} in ${mode} mode"
	qsub -n 1 -t 5 -A ExtCosmology --env PYTHONPATH=${pythondirs} /soft/libraries/anaconda/bin/python ./${script_name} -h
    else
	echo "Running ${script_name} to create ${nsnap} snapshots in ${mode} mode with time limit of ${timelimit} minutes"
	qsub -n 1 -t ${timelimit} -A ExtCosmology --env PYTHONPATH=${pythondirs} /soft/libraries/anaconda/bin/python ./${script_name} ${args} -nsnap ${nsnap}
    endif
endif
