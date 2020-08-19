There is one main script for the galacticus matchup: lc_resample.py (light cone resample)
There are several precompute scritps:
All scirpts run with one argrument: their parameter file. 
i.e: "./lc_resample.py param_lc_resamp/v4.1.dust136.param"
There are dependencies on dtk, Eve's galmatcher and the 
cosmoDC2 package.


Setup Scripts
============
      
1) lightcone_matchup.py param_lc_match/*.param:
  This script takes in parameters from param_lc_matchup folder
  This code takes in the binary outputs from the lightcone cutout cod and the
  UMachine + Color snapshot catalog and creates a lightcone catalog with all 
  snapshot catalog properties. If the positions don't change, then all one has
  to do is change the "snapshot_galaxy_fname" parameter to point to the new 
  UMachine + color snapshot catalog. This is a fast script. 

2) k_cor_step.py param_kcorr/*.param: 
   This scirpt generates the index to relate galaxies from step B to step A, where
    step B is later in time. 

3) precompute_mask.py param_lc_resamp/*.param:
   This script precomputes the masks on snaps shot (eg: color cuts, mstar cuts, etc)
   It should run on teh same parameter file as lc_resamp


Main Script:
============
     ./lc_resample.py param_lc_resampe/*.param
     This function matches up the lightcone UMachine galaxies to Galacticus galaxies. 
     These are parameters for the script:
     
     #The output form lightcone_matchup.py
     lightcone_fname /cosmo/homes/dkorytov/data/cosmoDC2/protoDC2_v4.1/lc_mock/lc_mock_v4.1_${step}.hdf5
     #galacticus snapshots
     gltcs_fname     /cosmo/homes/dkorytov/proj/protoDC2/output/ANL_box_v2.1.3_nocut_steps/${step}.hdf5
     # a galacticus file from which the metadata is copied from
     gltcs_metadata_ref  /cosmo/homes/dkorytov/proj/protoDC2/output/ANL_box_v2.1.3_nocut_mod.hdf5
     # not sued
     gltcs_slope_fname output/k_corr_${step}.hdf5
     # host halo info
     halo_shape_fname     /cosmo/homes/dkorytov/data/AlphaQ/halo_shapes/m000-${step}.bighaloshapes
     halo_shape_red_fname /cosmo/homes/dkorytov/data/AlphaQ/halo_shapes/m000-${step}.bighaloshapes_red
     sod_fname /cosmo/homes/dkorytov/data/AlphaQ/M200/STEP${step}/m000-${step}.sodproperties
     # which steps to run on. 
     #steps       315     307      300      293      286      279      272      266      259      253      247
     #steps           487      475      464      453      442      432      421      411      401      392      382      373      365      355      347      338      331      323      315      307      300      293      286      279      272      266      259      253      247
     #steps  331      323      315     307     
     #steps     315     307     
     steps   432      421

     # output location 
     output_fname  /cosmo/homes/dkorytov/data/cosmoDC2/protoDC2_test/ptest${step}.hdf5

     # Where to load the precomputed galmatcher mask & if to load it
     load_mask true
     mask_loc output/masks/mask.hdf5

     # Where to load indexing to match galacticus snapshots
     index_loc output/index/index_${step}.hdf5

     # Use_slope must be ste to true. OTher paths are depricated.
     # Use_slope turns on the interpolation bewteen snapshots. Substeps indiciates
     # into how many slices each step is divided into. The kdtree match is done 
     # on each slice. "use_substep_redshift" controls if galaxies use the substep's 
     # redshift, which is discrete, or their own individual redshift. 
     use_slope true
     substeps  1
     use_substep_redshift true

     # assignes new colors to galaxies using the assign color function
     # in teh cosmodc2 repo.
     recolor false

     # The parameters are used to speed up the run of the catalog. if "short" and "supershort"
     # are turned on, all columns but SDSS filters, stellar mass and new calculated quantites
     # are not output to disk. "short" only excludes LSST and SEDs. Cut_small_galaxies cuts out
     # small galaxies from the KDTree & matchup so should speed things up as well. It hasn't been
     # carefuly tested. 
     short     true
     supershort true
     cut_small_galaxies true
     cut_small_galaxies_mass 8 

     # Use_dust_factor must be turned on. Other paths are depcripated
     # "dust_factors" list the extra multiplicitive factors of dust.
     # a x1 factor is automatically included. 
     use_dust_factor true
     dust_factors 

     # To match on stellar or not
     ignore_mstar true
     # To match on observed colors (g-r, r-i, i-z)  for red squence 
     # galaxies in clusters (M_halo > 10**13.5). 
     match_obs_color_red_seq true
     
     # "plot" plots after each step of the input UM catalog vs the catalog on disk.
     # "plot_substep" plots the input UM catalog vs the matched galacticus for each
     # substep whil matching. 
     plot      true
     plot_substep true
     
     # The version number to write into the catalog metadata
     version_major       4
     version_minor       1
     version_minor_minor 0