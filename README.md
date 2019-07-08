# CosmoDC2 python package

This python package generates the cosmoDC2 synthetic galaxy catalog used by LSST-DESC for Data Challenge 2.


## Code Overview


### Mock production scripts

The scripts driving the generation of the cosmoDC2 catalog are contained in the cosmodc2/scripts directory. In particular: 

* `run_cosmoDC2_healpix_production.py`produces the synthetic lightcones. This script primarily functions as a wrapper around the `write_umachine_healpix_mock_to_disk.py` module in the top-level of the package. 
* `run_cosmoDC2_snapshot_production.py` produces synthetic catalogs at a single redshift. This script primarily functions as a wrapper around the `write_umachine_snapshot_mock_to_disk.py` module in the top-level of the package. 

### Models for the galaxy-halo connection

#### Stellar mass
Code implementing the model used in cosmoDC2 for stellar mass is located in the `stellar_mass_remapping` directory. The primary callable of interest is the `remap_stellar_mass_in_snapshot` function defined in the `remap_high_mass_smhm.py` module. 

#### Restframe optical color
Code implementing the model used in cosmoDC2 for broadband optical color is located in the `sdss_colors` directory. The primary callable of interest is the `assign_restframe_sdss_gri` function defined in the `v4_sdss_assign_gri.py` module. 

#### Black holes
Code implementing the model used in cosmoDC2 for black hole mass and accretion rate is located in the `black_hole_modeling ` directory. The primary callables of interest are:

* The `monte_carlo_black_hole_mass` function models black hole mass and is located in the `black_hole_mass.py` module. 
* The `monte_carlo_bh_acc_rate` function models black hole accretion rate and is located in the `black_hole_accretion_rate.py` module. 

#### Sizes 
Code implementing the model used in cosmoDC2 to map half-light radii onto the galaxy's bulge and disk component is located in the `size_modeling ` directory. The primary callables of interest are the `mc_size_vs_luminosity_early_type` function and the `mc_size_vs_luminosity_late_type` function in the `zhang_yang17.py` module.

#### Ultra-faint population 
Code implementing the model used in cosmoDC2 to extend the resolution limits of the Outer Rim simulation to capture the ultra-faint galaxy population is located in the `synthetic_subhalos ` directory. The primary callable of interest is the `model_extended_mpeak` function located in the `extend_subhalo_mpeak_range.py` module.


