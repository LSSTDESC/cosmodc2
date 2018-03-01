#!/bin/csh
#
if ($#argv < 2) then
    echo 'Usage: run_protoDC2_v3_mocks nsnap commit_hash [mode] [timelimit]'
    echo 'Script runs or submits job for creating protoDC2 mock catalogs'
    echo 'nsnap = number of snapshots to run'
    echo 'commit_hash = commit hash to write to output file'
    echo 'mode = test or qsub (default) or qtest'
    echo 'timelimit = timelimit for qsub mode (default = 5 minutes)'
    echo 'qsub runs generate_protoDC2_v3_snapshot_mocks.py in batch mode'
    echo 'test runs generate_protoDC2_v3_snapshot_mocks.py interactively'
    echo 'qtest runs generate_protoDC2_v3_snapshot_mocks.py in batch mode with -h option'
    echo 'output is written to the submit directory'
    exit
endif

set nsnap = ${1}
set commit_hash = ${2}
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

set script_name = "generate_protoDC2_v3_snapshot_mocks.py"

set pkldirname = "/home/ekovacs/cosmology/cosmodc2/cosmodc2"
set halocat_dirname = "/home/ahearin/protoDC2_v3_mocks/bpl_halos"
set um_dirname = "/home/ahearin/protoDC2_v3_mocks/um_snaps"
set umachine_z0p1_color_mock_fname = "/home/ahearin/protoDC2_v3_mocks/um_z0p1_color_mock/um_z0p1_color_mock.hdf5"
if (${mode} == "test") then
    set output_mocks_dirname = "./"
else
    set output_mocks_dirname = "/projects/DarkUniverse_esp/kovacs/AlphaQ/production_${commit_hash}"
endif
if (-e ${output_mocks_dirname}) then
else
    mkdir ${output_mocks_dirname}
endif
echo "Output directory is ${output_mocks_dirname}"

set python = "/soft/libraries/anaconda/bin/python"
set args = "${pkldirname} ${halocat_dirname} ${um_dirname} ${umachine_z0p1_color_mock_fname} ${output_mocks_dirname} ${commit_hash}"

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




