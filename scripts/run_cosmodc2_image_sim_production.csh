#!/bin/csh
#
if ($#argv < 1 ) then
    echo 'Usage: run_cosmodc2_image_sim_production version'
    echo 'Runs run_cosmoDC2_mocks_production_vxxx.csh'
    echo 'over the 14 healpixels required for image sims'
    echo 'Please supply version number'
    exit
endif

set version = "${1}"
echo 'Running run_cosmoDC2_mocks_9.8_centrals_production_v${version}.csh'
set time0 = 310
set time1 = 280
set time2 = 280

#main regions
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
