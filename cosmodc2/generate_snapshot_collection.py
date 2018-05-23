""" Module storing write_snapshot_mocks_to_disk, the end-to-end function
that generatee the full collection of AlphaQ halos that have been populated
with model galaxies with the following properties: {M*, SFR, Mr, g-r, r-i},
where colors are restframe extincted SDSS colors k-corrected to z=0.1.

This is the module used to generate the mocks in the v3 release of protoDC2.
"""
import os
from time import time
import numpy as np
import string
from astropy.table import Table
from galsampler import halo_bin_indices, source_halo_index_selection
from galsampler.utils import compute_richness
from galsampler.cython_kernels import galaxy_selection_kernel
from galsampler.source_galaxy_selection import _galaxy_table_indices
from halotools.utils import crossmatch
from halotools.empirical_models import enforce_periodicity_of_box
from halotools.mock_observables import relative_positions_and_velocities
from .load_gio_halos import load_gio_halo_snapshot
from .get_fof_info import get_fof_info
from .lightcone_id import append_lightcone_id, astropy_table_to_lightcone_hdf5
from .umachine_processing.load_umachine_outputs import retrieve_list_of_filenames


def get_filename_lists_of_protoDC2(pkldirname, halocat_dirname, um_dirname):
    """
    """
    _x = get_fof_info(pkldirname)
    redshift_strings, snapshots, alphaQ_halos_fname_list = _x
    redshift_list = [float(z) for z in redshift_strings]
    _y = retrieve_list_of_filenames(redshift_list, halocat_dirname, um_dirname)
    umachine_mstar_ssfr_mock_fname_list, bpl_halos_fname_list = _y

    dirname_alphaQ_halos = os.path.dirname(alphaQ_halos_fname_list[0])
    output_color_mock_basename_list = list(
        'protoDC2_v3_galaxies_' + string.replace(os.path.basename(fname), '.fofproperties', '')+ '.hdf5'
        for fname in alphaQ_halos_fname_list)
    # output_color_mock_fname_list = list(os.path.join(dirname_alphaQ_halos, basename)
    #     for basename in output_color_mock_basename_list)

    return (alphaQ_halos_fname_list, umachine_mstar_ssfr_mock_fname_list,
            bpl_halos_fname_list, output_color_mock_basename_list, redshift_list)


def write_snapshot_mocks_to_disk(
            umachine_z0p1_color_mock_fname, alphaQ_halos_fname_list,
            umachine_mstar_ssfr_mock_fname_list, bpl_halos_fname_list,
            output_color_mock_fname_list, redshift_list, commit_hash,
            Lbox, overwrite=False):
    """
    Function writes to disk a set of extragalactic snapshot catalogs by GalSampling UniverseMachine.

    Parameters
    ----------
    umachine_z0p1_color_mock_fname : string
        Absolute path to the z=0.1 UniverseMachine baseline mock that includes
        M*, SFR, Mr, g-r, r-i.

    alphaQ_halos_fname_list : list of strings
        List of absolute paths to the snapshot catalogs of FoF host halos in AlphaQ
        that will host galaxies in the output collection of mocks.

        These are the "target halos" in the language of GalSampler.

        The ordering of this list should be consistent with
        the orderings of the other filename lists.

    umachine_mstar_ssfr_mock_fname_list : list of strings
        List of absolute paths to the snapshot catalogs of
        UniverseMachine mock galaxies with M* and SFR.

        Values of Mr, g-r, and r-i will be painted onto these galaxies using
        the mock stored in the umachine_z0p1_color_mock_fname argument,
        and then these galaxies will be GalSampled into the AlphaQ snapshot.

        These are the "source galaxies" in the language of GalSampler.

        The ordering of this list should be consistent with
        the orderings of the other filename lists.

    bpl_halos_fname_list : list of strings
        List of absolute paths to the snapshot catalogs of
        Rockstar host halos hosting UniverseMachine mock galaxies.

        These are the "source halos" in the language of GalSampler.

        The ordering of this list should be consistent with
        the orderings of the other filename lists.

    output_color_mock_fname_list : list of strings
        List of absolute paths to the output catalogs

        The ordering of this list should be consistent with
        the orderings of the other filename lists.

    redshift_list : list
        List of floats storing the redshift of each protoDC2 snapshot

    Lbox : float
        Size of the simulation storing the UniverseMachine mocks

    """
    assert Lbox in (250, 1000), "Positional Lbox argument should be either BPl or MDPL2"
    raise NotImplementedError("Unfinished for v4")

    umachine_z0p1_color_mock = load_umachine_z0p1_color_mock(umachine_z0p1_color_mock_fname)

    gen = zip(alphaQ_halos_fname_list, umachine_mstar_ssfr_mock_fname_list,
            bpl_halos_fname_list, output_color_mock_fname_list, redshift_list)

    for fname1, fname2, fname3, output_color_mock_fname, redshift in gen:
        new_time_stamp = time()

        print("\n...loading z = {0:.3f} catalogs into memory".format(redshift))

        #  Load all three catalogs into memory
        alphaQ_halos = load_alphaQ_halos(fname1)
        umachine_mock = load_umachine_mstar_ssfr_mock(fname2)
        bpl_halos = load_bpl_halos(fname3, Lbox)

        #  Throw out the small number of galaxies for which there is no matching host
        idxA, idxB = crossmatch(umachine_mock['hostid'], bpl_halos['halo_id'])
        umachine_mock = umachine_mock[idxA]

        ########################################################################
        #  Create value-added catalogs
        ########################################################################

        print("          Computing host-centric positions for UniverseMachine galaxies")
        idxA, idxB = crossmatch(umachine_mock['hostid'], bpl_halos['halo_id'])

        #  Compute host halo position for every UniverseMachine galaxy
        umachine_mock['host_halo_x'] = np.nan
        umachine_mock['host_halo_y'] = np.nan
        umachine_mock['host_halo_z'] = np.nan
        umachine_mock['host_halo_vx'] = np.nan
        umachine_mock['host_halo_vy'] = np.nan
        umachine_mock['host_halo_vz'] = np.nan
        umachine_mock['host_halo_mvir'] = np.nan

        umachine_mock['host_halo_x'][idxA] = bpl_halos['x'][idxB]
        umachine_mock['host_halo_y'][idxA] = bpl_halos['y'][idxB]
        umachine_mock['host_halo_z'][idxA] = bpl_halos['z'][idxB]
        umachine_mock['host_halo_vx'][idxA] = bpl_halos['vx'][idxB]
        umachine_mock['host_halo_vy'][idxA] = bpl_halos['vy'][idxB]
        umachine_mock['host_halo_vz'][idxA] = bpl_halos['vz'][idxB]
        umachine_mock['host_halo_mvir'][idxA] = bpl_halos['mvir'][idxB]

        #  Throw out the small number of UniverseMachine galaxies with no matching host halo
        umachine_mock = umachine_mock[idxA]

        #  Compute halo-centric position for every UniverseMachine galaxy
        result = calculate_host_centric_position_velocity(umachine_mock, Lbox)
        xrel, yrel, zrel, vxrel, vyrel, vzrel = result
        umachine_mock['host_centric_x'] = xrel
        umachine_mock['host_centric_y'] = yrel
        umachine_mock['host_centric_z'] = zrel
        umachine_mock['host_centric_vx'] = vxrel
        umachine_mock['host_centric_vy'] = vyrel
        umachine_mock['host_centric_vz'] = vzrel

        print("          Computing galaxy--halo correspondence for UniverseMachine galaxies/halos\n")

        #  Sort the mock by host halo ID, putting centrals first within each grouping
        umachine_mock.sort(('hostid', 'upid'))

        #  Compute the number of galaxies in each Bolshoi-Planck halo
        bpl_halos['richness'] = compute_richness(
            bpl_halos['halo_id'], umachine_mock['hostid'])

        #  For every Bolshoi-Planck halo, compute the index of the galaxy table
        #  storing the central galaxy, reserving -1 for unoccupied halos
        bpl_halos['first_galaxy_index'] = _galaxy_table_indices(
            bpl_halos['halo_id'], umachine_mock['hostid'])
        #  Because of the particular sorting order of umachine_mock,
        #  knowledge of `first_galaxy_index` and `richness` gives sufficient
        #  information to map the correct members of umachine_mock to each halo

        ########################################################################
        #  Transfer the colors from the z=0.1 UniverseMachine mock
        #  to the other UniverseMachine mock
        ########################################################################

        print("          Matching z=0.1 color mock to z={0:.3f} mock".format(redshift))

        #  For every galaxy in umachine_mock, find a galaxy in umachine_z0p1_color_mock
        #  with a closely matching stellar mass and SFR-percentile
        source_mstar = umachine_z0p1_color_mock['obs_sm']
        source_percentile = umachine_z0p1_color_mock['sfr_percentile_fixed_sm']
        target_mstar = umachine_mock['obs_sm']
        target_percentile = umachine_mock['sfr_percentile_fixed_sm']
        um_matching_indx = um1_to_um2_matching_indices(
            source_mstar, source_percentile, target_mstar, target_percentile)

        keys_to_transfer = ('restframe_extincted_sdss_abs_magr',
                'restframe_extincted_sdss_gr', 'restframe_extincted_sdss_ri')
        for key in keys_to_transfer:
            umachine_mock[key] = umachine_z0p1_color_mock[key][um_matching_indx]

        #  Shift colors according to redshift
        gr_new, ri_new = shift_gr_ri_colors_at_high_redshift(
                umachine_mock['restframe_extincted_sdss_gr'],
                umachine_mock['restframe_extincted_sdss_ri'], redshift)
        umachine_mock['restframe_extincted_sdss_gr'] = gr_new
        umachine_mock['restframe_extincted_sdss_ri'] = ri_new

        ########################################################################
        #  For every host halo in the AlphaQ halo catalog,
        #  use GalSampler to find a matching halo in the Bolshoi-Planck catalog
        ########################################################################

        print("          Matching AlphaQ halos to Bolshoi-Planck halos")

        #  Bin the halos in each simulation by mass
        dlogM = 0.15
        mass_bins = 10.**np.arange(10.5, 14.5+dlogM, dlogM)
        bpl_halos['mass_bin'] = halo_bin_indices(
            mass=(bpl_halos['mvir'], mass_bins))
        alphaQ_halos['mass_bin'] = halo_bin_indices(
            mass=(alphaQ_halos['fof_halo_mass'], mass_bins))

        #  Randomly draw halos from corresponding mass bins
        nhalo_min = 10
        source_halo_bin_numbers = bpl_halos['mass_bin']
        target_halo_bin_numbers = alphaQ_halos['mass_bin']
        target_halo_ids = alphaQ_halos['halo_id']
        _result = source_halo_index_selection(source_halo_bin_numbers,
            target_halo_bin_numbers, target_halo_ids, nhalo_min, mass_bins)
        source_halo_indx, matching_target_halo_ids = _result

        #  Transfer quantities from the source halos to the corresponding target halo
        alphaQ_halos['source_halo_id'] = bpl_halos['halo_id'][source_halo_indx]
        alphaQ_halos['matching_mvir'] = bpl_halos['mvir'][source_halo_indx]
        alphaQ_halos['richness'] = bpl_halos['richness'][source_halo_indx]
        alphaQ_halos['first_galaxy_index'] = bpl_halos['first_galaxy_index'][source_halo_indx]

        ################################################################################
        #  Use GalSampler to calculate the indices of the galaxies that will be selected
        ################################################################################

        print("          Mapping z={0:.3f} galaxies to AlphaQ halos".format(redshift))

        source_galaxy_indx = np.array(galaxy_selection_kernel(
            alphaQ_halos['first_galaxy_index'].astype('i8'),
            alphaQ_halos['richness'].astype('i4'), alphaQ_halos['richness'].sum()))

        ########################################################################
        #  Assemble the output protoDC2 mock
        ########################################################################
        print("          Assembling z={0:.3f} output snapshot mock".format(redshift))

        output_snapshot_mock = build_output_snapshot_mock(
                umachine_mock, alphaQ_halos, source_galaxy_indx, commit_hash)

        ########################################################################
        #  Adding a unqiue id to each galaxy
        ########################################################################
        step_num = int(os.path.basename(output_color_mock_fname).replace(".hdf5","").split("m000-")[-1])

        append_lightcone_id(0, step_num, output_snapshot_mock)

        ########################################################################
        #  Write the output protoDC2 mock to disk
        ########################################################################
        print("          Writing to disk using commit hash {}".format(commit_hash))
        output_snapshot_mock.write(output_color_mock_fname, path='data', overwrite=overwrite)
        output_lightcone_fname = output_color_mock_fname.replace('.hdf5','') + "_lightcone.hdf5"
        astropy_table_to_lightcone_hdf5(output_snapshot_mock, output_lightcone_fname, commit_hash)
        old_time_stamp = time()
        msg = "Snapshot creation runtime = {0:.2f} minutes"
        print(msg.format((old_time_stamp-new_time_stamp)/60.))


def load_umachine_z0p1_color_mock(umachine_z0p1_color_mock_fname):
    """
    """
    return Table.read(umachine_z0p1_color_mock_fname, path='data')


def load_umachine_mstar_ssfr_mock(umachine_mstar_ssfr_mock_fname, Lbox):
    """
    """
    t = Table.read(umachine_mstar_ssfr_mock_fname, path='data')

    t['x'] = enforce_periodicity_of_box(t['x'], Lbox)
    t['y'] = enforce_periodicity_of_box(t['y'], Lbox)
    t['z'] = enforce_periodicity_of_box(t['z'], Lbox)

    #  Correct for edge case where position is exactly on box boundary
    epsilon = 0.0001
    t['x'][t['x'] == Lbox] = Lbox-epsilon
    t['y'][t['y'] == Lbox] = Lbox-epsilon
    t['z'][t['z'] == Lbox] = Lbox-epsilon

    return t


def load_alphaQ_halos(alphaQ_halos_fname):
    """
    """
    t = load_gio_halo_snapshot(alphaQ_halos_fname)

    t.rename_column('fof_halo_tag', 'halo_id')

    t.rename_column('fof_halo_center_x', 'x')
    t.rename_column('fof_halo_center_y', 'y')
    t.rename_column('fof_halo_center_z', 'z')

    t.rename_column('fof_halo_mean_vx', 'vx')
    t.rename_column('fof_halo_mean_vy', 'vy')
    t.rename_column('fof_halo_mean_vz', 'vz')

    return t


def load_bpl_halos(bpl_halos_fname, Lbox):
    """
    """
    t = Table.read(bpl_halos_fname, path='data')

    #  Correct for edge case where position is exactly on box boundary
    epsilon = 0.0001
    t['x'][t['x'] == Lbox] = Lbox-epsilon
    t['y'][t['y'] == Lbox] = Lbox-epsilon
    t['z'][t['z'] == Lbox] = Lbox-epsilon

    return t


def um1_to_um2_matching_indices(source_mstar, source_percentile, target_mstar, target_percentile):
    """
    """
    from scipy.spatial import cKDTree

    X1 = np.vstack((source_mstar, source_percentile)).T
    tree = cKDTree(X1)

    X2 = np.vstack((target_mstar, target_percentile)).T
    nn_distinces, nn_indices = tree.query(X2)

    return nn_indices


def transfer_colors_to_umachine_mstar_ssfr_mock(
        umachine_mstar_ssfr_mock, umachine_z0p1_color_mock, redshift,
        keys_to_match, keys_to_transfer):
    """
    """
    from scipy.spatial import cKDTree

    X = np.vstack((umachine_z0p1_color_mock[key] for key in keys_to_match)).T
    tree = cKDTree(X)

    Y = np.vstack((umachine_mstar_ssfr_mock[key] for key in keys_to_match)).T
    nn_distinces, nn_indices = tree.query(Y)

    for key in keys_to_transfer:
        umachine_mstar_ssfr_mock[key] = umachine_z0p1_color_mock[key][nn_indices]


def build_output_snapshot_mock(umachine, target_halos, galaxy_indices, commit_hash,
            Lbox_target=256.):
    """
    """
    dc2 = Table(meta={'commit_hash':commit_hash})
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

    source_galaxy_keys = ('host_halo_mvir', 'upid',
            'host_centric_x', 'host_centric_y', 'host_centric_z',
            'host_centric_vx', 'host_centric_vy', 'host_centric_vz',
            'obs_sm', 'obs_sfr', 'sfr_percentile_fixed_sm',
            'restframe_extincted_sdss_abs_magr',
            'restframe_extincted_sdss_gr', 'restframe_extincted_sdss_ri')
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


def calculate_host_centric_position_velocity(mock, Lbox):
    """
    """
    xrel, vxrel = relative_positions_and_velocities(mock['x'], mock['host_halo_x'],
        v1=mock['vx'], v2=mock['host_halo_vx'], period=Lbox)
    yrel, vyrel = relative_positions_and_velocities(mock['y'], mock['host_halo_y'],
        v1=mock['vy'], v2=mock['host_halo_vy'], period=Lbox)
    zrel, vzrel = relative_positions_and_velocities(mock['z'], mock['host_halo_z'],
        v1=mock['vz'], v2=mock['host_halo_vz'], period=Lbox)

    return xrel, yrel, zrel, vxrel, vyrel, vzrel


def shift_gr_ri_colors_at_high_redshift(gr, ri, redshift):
    """ Apply a simple multiplicative shift to the g-r and r-i color distributions
    to crudely mock up redshift evolution in the colors.

    Parameters
    ----------
    gr : ndarray
        Array of shape (ngals, ) storing the g-r colors

    ri : ndarray
        Array of shape (ngals, ) storing the r-i colors

    redshift : float
        Redshift of the snapshot

    Returns
    -------
    gr_new : ndarray
        Array of shape (ngals, ) storing the shifted g-r colors

    ri_new : ndarray
        Array of shape (ngals, ) storing the shifted r-i colors

    Examples
    --------
    >>> gr = np.random.uniform(0, 1.25, 1000)
    >>> ri = np.random.uniform(0.25, 0.75, 1000)
    >>> gr_new, ri_new = shift_gr_ri_colors_at_high_redshift(gr, ri, 0.8)
    >>> gr_new, ri_new = shift_gr_ri_colors_at_high_redshift(gr, ri, 8.)
    """
    gr_shift = np.interp(redshift, [0.1, 0.3, 1], [1., 1.15, 1.3])
    ri_shift = np.interp(redshift, [0.1, 0.3, 1], [1., 1.05, 1.1])
    gr_new = gr/gr_shift
    ri_new = ri/ri_shift
    return gr_new, ri_new
