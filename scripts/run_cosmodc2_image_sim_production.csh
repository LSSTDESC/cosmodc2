#!/bin/csh
#
if ($#argv < 2 ) then
    echo 'Usage: run_cosmodc2_image_sim_production version set'
    echo 'Runs run_cosmoDC2_mocks_production_vxxx.csh over the 14 healpixels required for image sims'
    echo 'version: version of production script'
    echo 'set: A or B for first 10 or last 4 healpix files, respectively (30 job limit on cooley)' 
    echo 'Please supply version and set number'
    exit
endif

set version = "${1}"
set set = "${2}"
echo 'Running run_cosmoDC2_mocks_9.8_centrals_production_v${version}.csh on set ${set} healpix files'
set time0 = 310
set time1 = 280
set time2 = 280

#main regions
if (${set} == 'A') then
    ./run_cosmoDC2_mocks_9.8_centrals_production_v${version}.csh cutout_564 0 qsub ${time0}
    ./run_cosmoDC2_mocks_9.8_centrals_production_v${version}.csh cutout_564 1 qsub ${time1}
    ./run_cosmoDC2_mocks_9.8_centrals_production_v${version}.csh cutout_564 2 qsub ${time2}

    ./run_cosmoDC2_mocks_9.8_centrals_production_v${version}.csh cutout_565 0 qsub ${time0}
    ./run_cosmoDC2_mocks_9.8_centrals_production_v${version}.csh cutout_565 1 qsub ${time1}
    ./run_cosmoDC2_mocks_9.8_centrals_production_v${version}.csh cutout_565 2 qsub ${time2}

    ./run_cosmoDC2_mocks_9.8_centrals_production_v${version}.csh cutout_566 0 qsub ${time0}
    ./run_cosmoDC2_mocks_9.8_centrals_production_v${version}.csh cutout_566 1 qsub ${time1}
    ./run_cosmoDC2_mocks_9.8_centrals_production_v${version}.csh cutout_566 2 qsub ${time2}

    ./run_cosmoDC2_mocks_9.8_centrals_production_v${version}.csh cutout_597 0 qsub ${time0}
    ./run_cosmoDC2_mocks_9.8_centrals_production_v${version}.csh cutout_597 1 qsub ${time1}
    ./run_cosmoDC2_mocks_9.8_centrals_production_v${version}.csh cutout_597 2 qsub ${time2}

    ./run_cosmoDC2_mocks_9.8_centrals_production_v${version}.csh cutout_598 0 qsub ${time0}
    ./run_cosmoDC2_mocks_9.8_centrals_production_v${version}.csh cutout_598 1 qsub ${time1}
    ./run_cosmoDC2_mocks_9.8_centrals_production_v${version}.csh cutout_598 2 qsub ${time2}

    ./run_cosmoDC2_mocks_9.8_centrals_production_v${version}.csh cutout_628 0 qsub ${time0}
    ./run_cosmoDC2_mocks_9.8_centrals_production_v${version}.csh cutout_628 1 qsub ${time1}
    ./run_cosmoDC2_mocks_9.8_centrals_production_v${version}.csh cutout_628 2 qsub ${time2}

    ./run_cosmoDC2_mocks_9.8_centrals_production_v${version}.csh cutout_629 0 qsub ${time0}
    ./run_cosmoDC2_mocks_9.8_centrals_production_v${version}.csh cutout_629 1 qsub ${time1}
    ./run_cosmoDC2_mocks_9.8_centrals_production_v${version}.csh cutout_629 2 qsub ${time2}

    ./run_cosmoDC2_mocks_9.8_centrals_production_v${version}.csh cutout_630 0 qsub ${time0}
    ./run_cosmoDC2_mocks_9.8_centrals_production_v${version}.csh cutout_630 1 qsub ${time1}
    ./run_cosmoDC2_mocks_9.8_centrals_production_v${version}.csh cutout_630 2 qsub ${time2}

     #overlaps
    ./run_cosmoDC2_mocks_9.8_centrals_production_v${version}.csh cutout_533 0 qsub ${time0}
    ./run_cosmoDC2_mocks_9.8_centrals_production_v${version}.csh cutout_533 1 qsub ${time1}
    ./run_cosmoDC2_mocks_9.8_centrals_production_v${version}.csh cutout_533 2 qsub ${time2}

    ./run_cosmoDC2_mocks_9.8_centrals_production_v${version}.csh cutout_534 0 qsub ${time0}
    ./run_cosmoDC2_mocks_9.8_centrals_production_v${version}.csh cutout_534 1 qsub ${time1}
    ./run_cosmoDC2_mocks_9.8_centrals_production_v${version}.csh cutout_534 2 qsub ${time2}
else
    ./run_cosmoDC2_mocks_9.8_centrals_production_v${version}.csh cutout_596 0 qsub ${time0}
    ./run_cosmoDC2_mocks_9.8_centrals_production_v${version}.csh cutout_596 1 qsub ${time1}
    ./run_cosmoDC2_mocks_9.8_centrals_production_v${version}.csh cutout_596 2 qsub ${time2}

    ./run_cosmoDC2_mocks_9.8_centrals_production_v${version}.csh cutout_599 0 qsub ${time0}
    ./run_cosmoDC2_mocks_9.8_centrals_production_v${version}.csh cutout_599 1 qsub ${time1}
    ./run_cosmoDC2_mocks_9.8_centrals_production_v${version}.csh cutout_599 2 qsub ${time2}

    ./run_cosmoDC2_mocks_9.8_centrals_production_v${version}.csh cutout_660 0 qsub ${time0}
    ./run_cosmoDC2_mocks_9.8_centrals_production_v${version}.csh cutout_660 1 qsub ${time1}
    ./run_cosmoDC2_mocks_9.8_centrals_production_v${version}.csh cutout_660 2 qsub ${time2}

    ./run_cosmoDC2_mocks_9.8_centrals_production_v${version}.csh cutout_661 0 qsub ${time0}
    ./run_cosmoDC2_mocks_9.8_centrals_production_v${version}.csh cutout_661 1 qsub ${time1}
    ./run_cosmoDC2_mocks_9.8_centrals_production_v${version}.csh cutout_661 2 qsub ${time2}
endif
