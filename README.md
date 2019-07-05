# CosmoDC2 python package

This python package generates the cosmoDC2 synthetic galaxy catalog used by LSST-DESC for Data Challenge 2.


## Code Overview


### Mock production scripts

The scripts driving the generation of the cosmoDC2 catalog are contained in the cosmodc2/scripts directory. In particular: 

* `run_cosmoDC2_healpix_production.py`produces the synthetic lightcones. This script primarily functions as a wrapper around the `write_umachine_healpix_mock_to_disk.py` module in the top-level of the package. 
* `run_cosmoDC2_snapshot_production.py` produces synthetic catalogs at a single redshift. This script primarily functions as a wrapper around the `write_umachine_snapshot_mock_to_disk.py` module in the top-level of the package. 

### Modules implementing models for the galaxy-halo connection

* Code implementing the model used in cosmoDC2 for broadband optical color is located in the `sdss_colors` directory. The primary callable of interest is the `assign_restframe_sdss_gri` function defined in the `v4_sdss_assign_gri.py` module. 
* Code implementing modifications to stellar masses is located in the `stellar_mass_remapping` directory. The primary callable of interest is the `remap_stellar_mass_in_snapshot` function defined in the `remap_high_mass_smhm.py` module. 
