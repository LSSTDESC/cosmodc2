""" Module storing the primary driver script used for the v1 release of cosmoDC2.
"""
import os
import psutil
import numpy as np
import h5py
import re
import healpy as hp
from time import time
from astropy.table import Table, vstack
from astropy.cosmology import FlatLambdaCDM
from astropy.utils.misc import NumpyRNGContext
from cosmodc2.load_gio_halos import load_gio_halo_snapshot
from cosmodc2.sdss_colors import assign_restframe_sdss_gri
from cosmodc2.sdss_colors.sigmoid_magr_model import magr_monte_carlo

from cosmodc2.stellar_mass_remapping import remap_stellar_mass_in_snapshot
from galsampler import halo_bin_indices, source_halo_index_selection
from galsampler.cython_kernels import galaxy_selection_kernel
from halotools.utils import crossmatch

from cosmodc2.synthetic_subhalos import map_mstar_onto_lowmass_extension
from cosmodc2.synthetic_subhalos import model_synthetic_cluster_satellites
#from cosmodc2.synthetic_subhalos import synthetic_logmpeak

fof_halo_mass = 'fof_halo_mass'
mass = 'mass'
fof_max = 14.5
H0 = 71.0
OmegaM = 0.2648
OmegaB = 0.0448

#unique galaxy_id
galaxy_id_factor = int(1e4)  #  factor to guarantee unique galaxy_id across blocks in snapshot

desired_logm_completeness=9.8

def write_umachine_snapshot_mock_to_disk(
        umachine_mock_fname, umachine_halo_fname,
        target_halo_catalog_fname, snapshot, blocks, output_snapshot_mock_fname_list,
        redshift, commit_hash, Lbox=3000.):
    """
    Main driver function used to paint SDSS fluxes onto UniverseMachine,
    GalSample the mock into the halo snapshot and write the snapshot mock to disk.

    Parameters
    ----------
    umachine_mock_fname : str 
        the absolute path to the value-added UniverseMachine snapshot mock

    umachine_halo_fname : str
        the absolute path to the
        value-added host halo catalog hosting the UniverseMachine snapshot mock

    target_halo_catalog_fname : str
        the absolute path to the gio file(s) of
        source halos into which UniverseMachine will be GalSampled

    snapshot : str
        snapshot being processed

    blocks : list  
        list of blocks (str format) of halo-catalog file being processed

    output_snapshot_mock_fname_list : list
        list of absolute paths to the output snapshot mock filenames (1 per block)

    redshift : str
        value of the redshift for the target halo catalog

    commit_hash : string
        Commit hash of the version of the cosmodc2 repo used when
        calling this function.

        After updating the cosmodc2 repo to the desired version,
        the commit_hash can be determined by navigating to the root
        directory and typing ``git log --pretty=format:'%h' -n 1``

    """

    start_time = time()
    process = psutil.Process(os.getpid())
    
    #  determine factor for number of synthetic galaxies for this snapshot
    #  synthetic_number_factor = (OR_size/MDPL2_size)**3

    #  initialize book-keeping variables
    fof_halo_mass_max = 0.
    Ngals_total = 0

    print('\nStarting snapshot processing')
    print('Redshift for halo catalog = {}'.format(redshift))

    #  Get galaxy properties from UM catalogs 
    print("\n...loading z = {0:.2f} galaxy catalog into memory".format(redshift))
    um_mock = Table.read(umachine_mock_fname, path='data')
    print('.....{} galaxies read in'.format(len(um_mock)))

    ### Get source halos 
    print("\n...loading z = {0:.2f} source-halo catalogs into memory".format(redshift))
    source_halos = Table.read(umachine_halo_fname, path='data')

    #  Bin the halos in source simulation by mass
    dlogM = 0.15
    mass_bins = 10.**np.arange(10.5, 14.5+dlogM, dlogM)
    source_halos['mass_bin'] = halo_bin_indices(
        mass=(source_halos['mvir'], mass_bins))

    for block, output_snap_fname  in zip(blocks, output_snapshot_mock_fname_list):
        new_time_stamp = time()
        #  determine seed from output filename (includes snapshot and block)
        seed = get_random_seed(os.path.basename(output_snap_fname))
        print('\n.Processing block {}'.format(block))
        print('.Using galaxy_id factor {} + galaxy_id offset {}'.format(galaxy_id_factor, block))
        print('.Using seed = {} (for block {})'.format(seed, block))

        # copy um_mock to new table for this block
        mock = um_mock.copy()
        
        print("\n...loading step {} fof target-halo catalogs into memory".format(snapshot))
        target_halos = load_gio_halo_snapshot(target_halo_catalog_fname, block=block)
        target_halos.rename_column('fof_halo_tag', 'fof_halo_id')
        target_halos.rename_column('fof_halo_center_x', 'x')
        target_halos.rename_column('fof_halo_center_y', 'y')
        target_halos.rename_column('fof_halo_center_z', 'z')
        target_halos.rename_column('fof_halo_mean_vx', 'vx')
        target_halos.rename_column('fof_halo_mean_vy', 'vy')
        target_halos.rename_column('fof_halo_mean_vz', 'vz')
        max_fof_halo_mass = np.max(target_halos[fof_halo_mass].quantity.value)
        fof_halo_mass_max = max(max_fof_halo_mass, fof_halo_mass_max)
        print('.....Maximum fof halo mass = {:.3e}'.format(max_fof_halo_mass))

        print("\n...Finding halo--halo correspondence with GalSampler")
        #  Bin the halos in target simulation by mass
        target_halos['mass_bin'] = halo_bin_indices(
            mass=(target_halos[fof_halo_mass], mass_bins))

        #  Randomly draw halos from corresponding mass bins
        nhalo_min = 10
        source_halo_bin_numbers = source_halos['mass_bin']
        target_halo_bin_numbers = target_halos['mass_bin']
        target_halo_ids = target_halos['fof_halo_id']
        _result = source_halo_index_selection(source_halo_bin_numbers,
                      target_halo_bin_numbers, target_halo_ids, nhalo_min, mass_bins, seed=seed)
        source_halo_indx, matching_target_halo_ids = _result

        #  Transfer quantities from the source halos to the corresponding target halo
        target_halos['source_halo_id'] = source_halos['halo_id'][source_halo_indx]
        target_halos['matching_mvir'] = source_halos['mvir'][source_halo_indx]
        target_halos['richness'] = source_halos['richness'][source_halo_indx]
        target_halos['first_galaxy_index'] = source_halos['first_galaxy_index'][source_halo_indx]

        ################################################################################
        #  Use GalSampler to calculate the indices of the galaxies that will be selected
        ################################################################################
        print("\n...GalSampling z={0:.2f} galaxies to OuterRim halos".format(redshift))

        source_galaxy_indx = np.array(galaxy_selection_kernel(
            target_halos['first_galaxy_index'].astype('i8'),
            target_halos['richness'].astype('i4'), target_halos['richness'].sum()))

        ########################################################################
        #  Correct stellar mass for low-mass subhalos and create synthetic mpeak
        ########################################################################
        #print("...correcting low mass mpeak and assigning synthetic mpeak values")
        #  First generate the appropriate number of synthetic galaxies for the snapshot
        #mpeak_synthetic_snapshot = 10**synthetic_logmpeak(
        #    mock['mpeak'], seed=seed, desired_logm_completeness=synthetic_halo_minimum_mass)
        #print('...assembling {} synthetic galaxies'.format(len(mpeak_synthetic_snapshot)))

        ########################################################################
        #  Assign stellar mass
        ########################################################################
        print("...re-assigning high-mass mstar values")

        #  Map stellar mass onto mock using target halo mass instead of UM Mpeak for cluster BCGs
        new_mstar = remap_stellar_mass_in_snapshot(redshift, mock['mpeak'], mock['obs_sm'])
        mock.rename_column('obs_sm', '_obs_sm_orig_um_snap')
        mock['obs_sm'] = new_mstar

        #  Add call to map_mstar_onto_lowmass_extension function after pre-determining low-mass slope
        print("...re-assigning low-mass mstar values")
        min_obs_sm = np.min(mock['obs_sm'])
        mpeak_synthetic_snapshot = np.asarray([])
        new_mstar_real, mstar_synthetic_snapshot = map_mstar_onto_lowmass_extension(
            mock['mpeak'], mock['obs_sm'], mpeak_synthetic_snapshot,
            desired_logm_completeness=desired_logm_completeness)
        mock['obs_sm'] = new_mstar_real
        new_min_obs_sm = np.min(mock['obs_sm'])
        print('.....New min(obs_sm) = {:.3e}; old min(obs_sm) = {:.3e}'.format(new_min_obs_sm, min_obs_sm))
        print('.....Number of shifted values = {}'.format(np.count_nonzero(mock['obs_sm'] < min_obs_sm)))

        ###################################################
        #  Map restframe Mr, g-r, r-i onto mock
        ###################################################
        #  use the redshift of the snapshot of the target simulation
        print("...assigning rest-frame Mr and colors")
        check_time = time()
        redshift_mock = np.zeros(len(mock)) + redshift
        msg = (".....using snapshot redshift to assign restframe colors")
        print(msg)

        magr, gr_mock, ri_mock, is_red_gr, is_red_ri = assign_restframe_sdss_gri(
            mock['upid'], mock['obs_sm'], mock['sfr_percentile'],
            mock['host_halo_mvir'], redshift_mock, seed=seed, use_substeps=False)
        #  check for bad values
        for m_id, m in zip(['magr', 'gr', 'ri'], [magr, gr_mock, ri_mock]):
            num_infinite = np.sum(~np.isfinite(m))
            if num_infinite > 0:
                print('.....Warning: {} infinite values in mock {}'.format(num_infinite, m_id))

        mock['restframe_extincted_sdss_abs_magr'] = magr
        mock['restframe_extincted_sdss_gr'] = gr_mock
        mock['restframe_extincted_sdss_ri'] = ri_mock
        mock['is_on_red_sequence_gr'] = is_red_gr
        mock['is_on_red_sequence_ri'] = is_red_ri
        print('.....time to assign_restframe_sdss_gri = {:.2f} secs'.format(time()-check_time))

        ########################################################################
        #  Assemble the output mock by snapshot
        ########################################################################

        print("\n...building output snapshot mock for snapshot {}".format(snapshot))
        output_mock = build_output_snapshot_mock(redshift, mock, target_halos,
                                                          source_galaxy_indx, galaxy_id_factor,
                                                          int(block), Lbox=Lbox)
        Ngals_total += len(output_mock['galaxy_id'])
        print('...saved {} galaxies to dict'.format(len(output_mock['galaxy_id'])))

        ########################################################################
        #  Write the output mock to disk
        ########################################################################
        if len(output_mock) > 0:
            check_time = time()
            write_output_mock_to_disk(output_snap_fname, output_mock, commit_hash, seed,
                                      redshift, snapshot, block, Lbox)
            print('...time to write mock to disk = {:.2f} minutes'.format((time()-check_time)/60.))

        time_stamp = time()
        msg = "\n.Block runtime = {0:.2f} minutes"
        print(msg.format((time_stamp-new_time_stamp)/60.))
        mem = ".Memory usage =  {0:.2f} GB"
        print(mem.format(process.memory_info().rss/1.e9))

                                      
    file_info = 'snapshot {}, blocks {}'.format(snapshot, ', '.join(blocks))
    print('\n.Maximum halo mass in {} = {}\n'.format(file_info, fof_halo_mass_max))
    print('.Number of galaxies in {} = {}\n'.format(file_info, Ngals_total))

    time_stamp = time()
    msg = "\nEnd-to-end runtime = {0:.2f} minutes\n"
    print(msg.format((time_stamp-start_time)/60.))


def get_random_seed(filename, seed_max=4294967095):  #reduce max seed by 200 to allow for 60 z shells
    import hashlib
    s = hashlib.md5(filename).hexdigest()
    seed = int(s, 16)

    #  enforce seed is below seed_max and odd
    seed = seed%seed_max
    if seed%2 == 0:
        seed = seed + 1
    return seed


def build_output_snapshot_mock(
            snapshot_redshift, umachine, target_halos, galaxy_indices, galaxy_id_factor,
            galaxy_id_offset, Lbox=0.):
    """
    Collect the GalSampled snapshot mock into an astropy table

    Parameters
    ----------
    snapshot_redshift : float
        Float of the snapshot redshift

    umachine : astropy.table.Table
        Astropy Table of shape (num_source_gals, )
        storing the UniverseMachine snapshot mock

    target_halos : astropy.table.Table
        Astropy Table of shape (num_target_halos, )
        storing the target halo catalog

    galaxy_indices: ndarray
        Numpy indexing array of shape (num_target_gals, )
        storing integers valued between [0, num_source_gals)

    galaxy_id_factor: integer
        Multiplicative factor to ensure unique galaxy id's in a snapshot

    galaxy_id_offset: integer
        Offset to ensure unique galaxy id's in a snapshot

    Returns
    -------
    dc2 : astropy.table.Table
        Astropy Table of shape (num_target_gals, )
        storing the GalSampled galaxy catalog
    """
    dc2 = Table()
    dc2['source_halo_id'] = umachine['hostid'][galaxy_indices]
    dc2['target_halo_id'] = np.repeat(
        target_halos['fof_halo_id'], target_halos['richness'])
    #needed for synthetic cluster satellite assignment
    dc2['target_halo_redshift'] = np.repeat(snapshot_redshift, len(dc2['target_halo_id']))
    dc2['target_halo_fof_halo_id'] = dc2['target_halo_id']

    #  copy target halo information
    dc2['source_halo_mvir'] = np.repeat(
        target_halos['matching_mvir'], target_halos['richness'])

    idxA, idxB = crossmatch(dc2['target_halo_id'], target_halos['fof_halo_id'])

    msg = "target IDs do not match!"
    assert np.all(dc2['source_halo_id'][idxA] == target_halos['source_halo_id'][idxB]), msg

    dc2['target_halo_x'] = 0.
    dc2['target_halo_y'] = 0.
    dc2['target_halo_z'] = 0.
    dc2['target_halo_vx'] = 0.
    dc2['target_halo_vy'] = 0.
    dc2['target_halo_vz'] = 0.

    dc2['target_halo_x'][idxA] = target_halos['x'][idxB]
    dc2['target_halo_y'][idxA] = target_halos['y'][idxB]
    dc2['target_halo_z'][idxA] = target_halos['z'][idxB]

    dc2['target_halo_vx'][idxA] = target_halos['vx'][idxB]
    dc2['target_halo_vy'][idxA] = target_halos['vy'][idxB]
    dc2['target_halo_vz'][idxA] = target_halos['vz'][idxB]

    dc2['target_halo_mass'] = 0.
    dc2['target_halo_mass'][idxA] = target_halos['fof_halo_mass'][idxB]

    source_galaxy_keys = ('host_halo_mvir', 'upid', 'mpeak',
            'host_centric_x', 'host_centric_y', 'host_centric_z',
            'host_centric_vx', 'host_centric_vy', 'host_centric_vz',
            'obs_sm', 'obs_sfr', 'sfr_percentile',
            'restframe_extincted_sdss_abs_magr',
            'restframe_extincted_sdss_gr', 'restframe_extincted_sdss_ri',
            'is_on_red_sequence_gr', 'is_on_red_sequence_ri',
            '_obs_sm_orig_um_snap', 'halo_id')
    for key in source_galaxy_keys:
        try:
            dc2[key] = umachine[key][galaxy_indices]
        except KeyError:
            msg = ("The build_output_snapshot_mock function was passed a umachine mock\n"
                "that does not contain the ``{0}`` key")
            raise KeyError(msg.format(key))

    max_umachine_halo_mass = np.max(umachine['mpeak'])
    ultra_high_mvir_halo_mask = (dc2['upid'] == -1) & (dc2['target_halo_mass'] > max_umachine_halo_mass)
    num_to_remap = np.count_nonzero(ultra_high_mvir_halo_mask)
    if num_to_remap > 0:
        print("...remapping stellar mass of {0} BCGs in ultra-massive halos".format(num_to_remap))

        halo_mass_array = dc2['target_halo_mass'][ultra_high_mvir_halo_mask]
        mpeak_array = dc2['mpeak'][ultra_high_mvir_halo_mask]
        mhalo_ratio = halo_mass_array/mpeak_array
        mstar_array = dc2['obs_sm'][ultra_high_mvir_halo_mask]
        redshift_array = dc2['target_halo_redshift'][ultra_high_mvir_halo_mask]
        upid_array = dc2['upid'][ultra_high_mvir_halo_mask]

        assert np.shape(halo_mass_array) == (num_to_remap, ), "halo_mass_array has shape = {0}".format(np.shape(halo_mass_array))
        assert np.shape(mstar_array) == (num_to_remap, ), "mstar_array has shape = {0}".format(np.shape(mstar_array))
        assert np.shape(redshift_array) == (num_to_remap, ), "redshift_array has shape = {0}".format(np.shape(redshift_array))
        assert np.shape(upid_array) == (num_to_remap, ), "upid_array has shape = {0}".format(np.shape(upid_array))
        assert np.all(mhalo_ratio >= 1), "Bookkeeping error: all values of mhalo_ratio ={0} should be >= 1".format(mhalo_ratio)

        dc2['obs_sm'][ultra_high_mvir_halo_mask] = mstar_array*(mhalo_ratio**0.5)
        dc2['restframe_extincted_sdss_abs_magr'][ultra_high_mvir_halo_mask] = magr_monte_carlo(
            dc2['obs_sm'][ultra_high_mvir_halo_mask], upid_array, redshift_array)
        idx = np.argmax(dc2['obs_sm'])
        halo_id_most_massive = dc2['halo_id'][idx]
        assert dc2['obs_sm'][idx] < 10**13.5, "halo_id = {0} has stellar mass {1:.3e}".format(
            halo_id_most_massive, dc2['obs_sm'][idx])

    dc2['x'] = dc2['target_halo_x'] + dc2['host_centric_x']
    dc2['vx'] = dc2['target_halo_vx'] + dc2['host_centric_vx']

    dc2['y'] = dc2['target_halo_y'] + dc2['host_centric_y']
    dc2['vy'] = dc2['target_halo_vy'] + dc2['host_centric_vy']

    dc2['z'] = dc2['target_halo_z'] + dc2['host_centric_z']
    dc2['vz'] = dc2['target_halo_vz'] + dc2['host_centric_vz']

    print('...number of galaxies before adding synthetic satellites = {}'.format(len(dc2['halo_id'])))
    print("...generating and stacking any synthetic cluster satellites")
    fake_cluster_sats = model_synthetic_cluster_satellites(dc2, Lbox=Lbox, snapshot=True)
    if len(fake_cluster_sats) > 0:
        check_time = time()
        dc2 = vstack((dc2, fake_cluster_sats))
        print('...time to create {} galaxies in fake_cluster_sats = {:.2f} secs'.format(len(fake_cluster_sats['target_halo_id']), time()-check_time))

    # delete duplicate/unnecessary column after fake cluster satellites are added
    dc2.remove_column('target_halo_fof_halo_id')
    dc2.remove_column('target_halo_redshift')

    dc2['galaxy_id'] = np.arange(len(dc2['target_halo_id'])).astype(int)*galaxy_id_factor + galaxy_id_offset
    print('...Min and max galaxy_id = {} -> {}'.format(np.min(dc2['galaxy_id']), np.max(dc2['galaxy_id'])))

    #  Use gr and ri color to compute gi flux
    dc2['restframe_extincted_sdss_abs_magg'] = (
        dc2['restframe_extincted_sdss_gr'] +
        dc2['restframe_extincted_sdss_abs_magr'])
    dc2['restframe_extincted_sdss_abs_magi'] = (
        -dc2['restframe_extincted_sdss_ri'] +
        dc2['restframe_extincted_sdss_abs_magr'])

    #convert table to dict
    check_time = time()
    output_dc2 = {}
    for k in dc2.keys():
        output_dc2[k] = dc2[k].quantity.value

    print('...time to new dict = {:.4f} secs'.format(time()-check_time))

    return output_dc2


def write_output_mock_to_disk(output_snapshot_mock_fname, output_mock, commit_hash, seed,
                              redshift, snapshot, block, Lbox,
                              versionMajor=0, versionMinor=1, versionMinorMinor=0):
    """
    """

    print("\n...writing to file {} using commit hash {}".format(output_snapshot_mock_fname, commit_hash))
    hdfFile = h5py.File(output_snapshot_mock_fname, 'w')
    hdfFile.create_group('metaData')
    gkey = 'galaxyProperties'

    hdfFile['metaData']['commit_hash'] = commit_hash
    hdfFile['metaData']['seed'] = seed
    hdfFile['metaData']['versionMajor'] = versionMajor
    hdfFile['metaData']['versionMinor'] = versionMinor
    hdfFile['metaData']['versionMinorMinor'] = versionMinorMinor
    hdfFile['metaData']['H_0'] = H0
    hdfFile['metaData']['Omega_matter'] = OmegaM
    hdfFile['metaData']['Omega_b'] = OmegaB
    hdfFile['metaData']['box_size'] = Lbox
    hdfFile['metaData']['redshift'] = redshift
    hdfFile['metaData']['timestep'] = snapshot
    hdfFile['metaData']['block_number'] = block
    for d in ['x', 'y', 'z']:
        hdfFile['metaData'][d + '_max'] = np.max(output_mock[d]) 
        hdfFile['metaData'][d + '_min'] = np.min(output_mock[d]) 

    gGroup = hdfFile.create_group(gkey)
    check_time = time()
    for k, v in output_mock.items():
        gGroup[k] = v

    print('.....time to write group {} = {:.4f} secs'.format(gkey, time()-check_time))

    check_time = time()
    hdfFile.close()
    print('.....time to close file {:.4f} secs'.format(time()-check_time))
