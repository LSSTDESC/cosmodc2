#!/bin/csh
#
if ($#argv < 1 ) then
    echo 'Usage: run_protoDC2_v3_mocks nsnap [mode] [timelimit]'
    echo 'Script runs or submits job for creating protoDC2 mock catalogs'
    echo 'nsnap = number of snapshots to run'
    echo 'mode = test or qsub (default) or qtest'
    echo 'timelimit = timelimit for qsub mode (default = 5 minutes)'
    echo 'qsub runs generate_protoDC2_v3_snapshot_mocks.py in batch mode'
    echo 'test runs generate_protoDC2_v3_snapshot_mocks.py interactively'
    echo 'qtest runs generate_protoDC2_v3_snapshot_mocks.py in batch mode with -h option'
    echo 'output is written to the submit directory'
    exit
endif

set nsnap = ${1}
set mode = "qsub"
set timelimit = "5"
if ($#argv > 1 ) then
    if(${2} == "qsub" || ${2} == "test" || ${2} == "qtest") then
	set mode = "${2}"
    else
	set timelimit = "${2}"
    endif
    if ($#argv > 2 ) then
	if(${3} == "qsub" || ${3} == "test" || ${3} == "qtest") then
	    set mode = "${3}"
	else
	    set timelimit = "${3}"
	endif
    endif
endif

set script_name = "generate_protoDC2_v3_snapshot_mocks.py"

set pkldirname = "/home/ekovacs/cosmology/cosmodc2/cosmodc2"
set halocat_dirname = "/home/ahearin/protoDC2_v3_mocks/bpl_halos"
set um_dirname = "/home/ahearin/protoDC2_v3_mocks/um_snaps"
set umachine_z0p1_color_mock_fname = "/home/ahearin/protoDC2_v3_mocks/um_z0p1_color_mock/um_z0p1_color_mock.hdf5"
#set output_mocks_dirname = "~/cosmology/DC2/test"
set output_mocks_dirname = "./"

set python = "/soft/libraries/anaconda/bin/python"
set args = "${pkldirname} ${halocat_dirname} ${um_dirname} ${umachine_z0p1_color_mock_fname} ${output_mocks_dirname}"

set pythondirs = /gpfs/mira-home/ekovacs/cosmology/cosmodc2:/gpfs/mira-home/ekovacs/cosmology/galsampler/build/lib.linux-x86_64-2.7:/gpfs/mira-home/ekovacs/cosmology/halotools/build/lib.linux-x86_64-2.7

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
