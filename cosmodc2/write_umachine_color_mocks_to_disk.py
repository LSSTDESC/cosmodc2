""" Module storing the primary driver script used for the v4 release of DC2.
"""
import os
import numpy as np
from time import time
from astropy.table import Table
from cosmodc2.stellar_mass_remapping import remap_stellar_mass_in_snapshot
from cosmodc2.sdss_colors import assign_restframe_sdss_gri
from galsampler import halo_bin_indices, source_halo_index_selection
from galsampler.cython_kernels import galaxy_selection_kernel
from cosmodc2.load_gio_halos import load_gio_halo_snapshot
from halotools.empirical_models import enforce_periodicity_of_box
from halotools.utils import crossmatch
from cosmodc2.lightcone_id import append_lightcone_id, astropy_table_to_lightcone_hdf5


def write_snapshot_mocks_to_disk(
            umachine_mstar_ssfr_mock_fname_list, umachine_host_halo_fname_list,
            target_halo_fname_list, output_color_mock_fname_list,
            redshift_list, commit_hash, target_halo_loader, Lbox_target_halos):
    """
    Main driver function used to paint SDSS fluxes onto UniverseMachine,
    GalSample the mock into AlphaQ, and write each snapshot to disk.

    Parameters
    ----------
    umachine_mstar_ssfr_mock_fname_list : list
        List of length num_snaps storing the absolute path to the
        value-added UniverseMachine snapshot mock

    umachine_host_halo_fname_list : list
        List of length num_snaps storing the absolute path to the
        value-added host halo catalog hosting the UniverseMachine snapshot mock

    target_halo_fname_list : list
        List of length num_snaps storing the absolute path to the
        source halos into which UniverseMachine will be GalSampled

    output_color_mock_fname_list : list
        List of length num_snaps storing the absolute path to the
        output snapshot mocks

    redshift_list : list
        List of length num_snaps storing the value of the redshift
        of the target halo catalog

    commit_hash : string
        Commit hash of the version of the cosmodc2 repo used when
        calling this function.

        After updating the cosmodc2 repo to the desired version,
        the commit_hash can be determined by navigating to the root
        directory and typing ``git log --pretty=format:'%h' -n 1``

    target_halo_loader : string
        Format of the target halo catalog. Current options are
        "gio" and "hdf5"

    Lbox_target_halos : float
        Box size of the target halos in comoving units of Mpc/h.
        Should be 256 for AlphaQ and 4.5 for Outer Rim

    """

    gen = zip(umachine_mstar_ssfr_mock_fname_list, umachine_host_halo_fname_list,
            target_halo_fname_list, output_color_mock_fname_list, redshift_list)

    for a, b, c, d, e in gen:
        umachine_mock_fname = a
        umachine_halos_fname = b
        target_halo_fname = c
        output_color_mock_fname = d
        redshift = e

        new_time_stamp = time()
        print("\n...loading z = {0:.2f} galaxy catalog into memory".format(redshift))

        mock = Table.read(umachine_mock_fname, path='data')

        #  Remap stellar mass
        mock.rename_column('obs_sm', '_obs_sm_orig_um_snap')
        mock['obs_sm'] = remap_stellar_mass_in_snapshot(
            redshift, mock['mpeak'], mock['_obs_sm_orig_um_snap'])

        upid_mock = mock['upid']
        mstar_mock = mock['obs_sm']
        sfr_percentile_mock = mock['sfr_percentile']
        host_halo_mvir_mock = mock['host_halo_mvir']
        redshift_mock = np.random.uniform(redshift-0.1, redshift+0.1, len(mock))
        redshift_mock = np.where(redshift_mock < 0, 0, redshift_mock)

        print("\n...assigning SDSS restframe colors")

        magr, gr_mock, ri_mock, is_red_gr, is_red_ri = assign_restframe_sdss_gri(
            upid_mock, mstar_mock, sfr_percentile_mock, host_halo_mvir_mock, redshift_mock)
        mock['restframe_extincted_sdss_abs_magr'] = magr
        mock['restframe_extincted_sdss_gr'] = gr_mock
        mock['restframe_extincted_sdss_ri'] = ri_mock
        mock['is_on_red_sequence_gr'] = is_red_gr
        mock['is_on_red_sequence_ri'] = is_red_ri

        ###  GalSampler
        print("\n...loading z = {0:.2f} halo catalogs into memory".format(redshift))
        source_halos = Table.read(umachine_halos_fname, path='data')
        target_halos = load_alphaQ_halos(target_halo_fname, target_halo_loader)

        print("...Finding halo--halo correspondence with GalSampler")
        #  Bin the halos in each simulation by mass
        dlogM = 0.15
        mass_bins = 10.**np.arange(10.5, 14.5+dlogM, dlogM)
        source_halos['mass_bin'] = halo_bin_indices(
            mass=(source_halos['mvir'], mass_bins))
        target_halos['mass_bin'] = halo_bin_indices(
            mass=(target_halos['fof_halo_mass'], mass_bins))

        #  Randomly draw halos from corresponding mass bins
        nhalo_min = 10
        source_halo_bin_numbers = source_halos['mass_bin']
        target_halo_bin_numbers = target_halos['mass_bin']
        target_halo_ids = target_halos['halo_id']
        _result = source_halo_index_selection(source_halo_bin_numbers,
            target_halo_bin_numbers, target_halo_ids, nhalo_min, mass_bins)
        source_halo_indx, matching_target_halo_ids = _result

        #  Transfer quantities from the source halos to the corresponding target halo
        target_halos['source_halo_id'] = source_halos['halo_id'][source_halo_indx]
        target_halos['matching_mvir'] = source_halos['mvir'][source_halo_indx]
        target_halos['richness'] = source_halos['richness'][source_halo_indx]
        target_halos['first_galaxy_index'] = source_halos['first_galaxy_index'][source_halo_indx]

        ################################################################################
        #  Use GalSampler to calculate the indices of the galaxies that will be selected
        ################################################################################
        print("...GalSampling z={0:.2f} galaxies to AlphaQ halos".format(redshift))

        source_galaxy_indx = np.array(galaxy_selection_kernel(
            target_halos['first_galaxy_index'].astype('i8'),
            target_halos['richness'].astype('i4'), target_halos['richness'].sum()))

        ########################################################################
        #  Assemble the output protoDC2 mock
        ########################################################################

        print("...building output snapshot mock")
        output_snapshot_mock = build_output_snapshot_mock(
                mock, target_halos, source_galaxy_indx, commit_hash, Lbox_target_halos)

        ########################################################################
        #  Adding a unqiue id to each galaxy
        ########################################################################
        step_num = int(os.path.basename(
            output_color_mock_fname).replace(".hdf5", "").split("m000-")[-1])

        append_lightcone_id(0, step_num, output_snapshot_mock)

        ########################################################################
        #  Write the output protoDC2 mock to disk
        ########################################################################
        print("...writing to disk using commit hash {}".format(commit_hash))
        output_snapshot_mock.meta['cosmodc2_commit_hash'] = commit_hash
        output_snapshot_mock.write(output_color_mock_fname, path='data', overwrite=True)
        output_lightcone_fname = output_color_mock_fname.replace('.hdf5','') + "_lightcone.hdf5"
        astropy_table_to_lightcone_hdf5(output_snapshot_mock, output_lightcone_fname, commit_hash)

        time_stamp = time()
        msg = "End-to-end runtime for redshift {0:.1f} = {1:.2f} minutes"
        print(msg.format(redshift, (time_stamp-new_time_stamp)/60.))


def load_alphaQ_halos(fname, target_halo_loader):
    """
    """
    if target_halo_loader == 'hdf5':
        t = Table.read(fname, path='data')
    elif target_halo_loader == 'gio':
        t = load_gio_halo_snapshot(fname)
    else:
        raise ValueError("Options for ``loader`` are ``hdf5`` or ``gio``")

    t.rename_column('fof_halo_tag', 'halo_id')

    t.rename_column('fof_halo_center_x', 'x')
    t.rename_column('fof_halo_center_y', 'y')
    t.rename_column('fof_halo_center_z', 'z')

    t.rename_column('fof_halo_mean_vx', 'vx')
    t.rename_column('fof_halo_mean_vy', 'vy')
    t.rename_column('fof_halo_mean_vz', 'vz')

    return t


def build_output_snapshot_mock(
            umachine, target_halos, galaxy_indices, commit_hash, Lbox_target):
    """
    Collect the GalSampled snapshot mock into an astropy table

    Parameters
    ----------
    umachine : astropy.table.Table
        Astropy Table of shape (num_source_gals, )
        storing the UniverseMachine snapshot mock

    target_halos : astropy.table.Table
        Astropy Table of shape (num_target_halos, )
        storing the target halo catalog

    galaxy_indices: ndarray
        Numpy indexing array of shape (num_target_gals, )
        storing integers valued between [0, num_source_gals)

    commit_hash : string
        Commit hash of the version of the cosmodc2 repo used when
        calling this function.

        After updating the cosmodc2 repo to the desired version,
        the commit_hash can be determined by navigating to the root
        directory and typing ``git log --pretty=format:'%h' -n 1``

    Lbox_target : float
        Box size of the target halos in comoving units of Mpc/h.
        Should be 256 for AlphaQ and 4.5 for Outer Rim

    Returns
    -------
    dc2 : astropy.table.Table
        Astropy Table of shape (num_target_gals, )
        storing the GalSampled galaxy catalog
    """
    dc2 = Table(meta={'commit_hash': commit_hash})
    dc2['source_halo_id'] = umachine['hostid'][galaxy_indices]
    dc2['target_halo_id'] = np.repeat(
        target_halos['halo_id'], target_halos['richness'])

    idxA, idxB = crossmatch(dc2['target_halo_id'], target_halos['halo_id'])

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
        dc2[key] = umachine[key][galaxy_indices]

    x_init = dc2['target_halo_x'] + dc2['host_centric_x']
    vx_init = dc2['target_halo_vx'] + dc2['host_centric_vx']
    dc2_x, dc2_vx = enforce_periodicity_of_box(x_init, Lbox_target, velocity=vx_init)
    dc2['x'] = dc2_x
    dc2['vx'] = dc2_vx

    y_init = dc2['target_halo_y'] + dc2['host_centric_y']
    vy_init = dc2['target_halo_vy'] + dc2['host_centric_vy']
    dc2_y, dc2_vy = enforce_periodicity_of_box(y_init, Lbox_target, velocity=vy_init)
    dc2['y'] = dc2_y
    dc2['vy'] = dc2_vy

    z_init = dc2['target_halo_z'] + dc2['host_centric_z']
    vz_init = dc2['target_halo_vz'] + dc2['host_centric_vz']
    dc2_z, dc2_vz = enforce_periodicity_of_box(z_init, Lbox_target, velocity=vz_init)
    dc2['z'] = dc2_z
    dc2['vz'] = dc2_vz

    return dc2

