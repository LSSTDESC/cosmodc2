"""
"""
from scipy.spatial import cKDTree
import numpy as np
from astropy.table import Table
from galsampler import halo_bin_indices, source_halo_index_selection, source_galaxy_selection_indices
from galsampler.utils import compute_richness
from galsampler.source_galaxy_selection import _galaxy_table_indices
from halotools.utils import crossmatch
from halotools.empirical_models import enforce_periodicity_of_box
from halotools.mock_observables import relative_positions_and_velocities


def write_sdss_restframe_color_snapshot_mocks_to_disk(
            umachine_z0p1_color_mock_fname, alphaQ_halos_fname_list,
            umachine_mstar_ssfr_mock_fname_list, bpl_halos_fname_list,
            output_color_mock_fname_list, redshift_list, overwrite=False):
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
        List storing the redshift of each protoDC2 snapshot

    """
    umachine_z0p1_color_mock = load_umachine_z0p1_color_mock(umachine_z0p1_color_mock_fname)

    gen = zip(alphaQ_halos_fname_list, umachine_mstar_ssfr_mock_fname_list,
            bpl_halos_fname_list, output_color_mock_fname_list, redshift_list)

    for fname1, fname2, fname3, output_color_mock_fname, redshift in gen:
        print("...working on z = {0:.2f}".format(redshift))

        #  Load all three catalogs into memory
        alphaQ_halos = load_alphaQ_halos(fname1)
        umachine_mock = load_umachine_mstar_ssfr_mock(fname2)
        bpl_halos = load_bpl_halos(fname3)

        ########################################################################
        #  Create value-added catalogs
        ########################################################################

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

        #  Throw out the small number of halos with no matching host halo
        umachine_mock = umachine_mock[idxA]

        #  Compute halo-centric position for every UniverseMachine galaxy
        result = calculate_host_centric_position_velocity(umachine_mock)
        xrel, yrel, zrel, vxrel, vyrel, vzrel = result
        umachine_mock['host_centric_x'] = xrel
        umachine_mock['host_centric_y'] = yrel
        umachine_mock['host_centric_z'] = zrel
        umachine_mock['host_centric_vx'] = vxrel
        umachine_mock['host_centric_vy'] = vyrel
        umachine_mock['host_centric_vz'] = vzrel

        #  Sort the mock by host halo ID, putting centrals first
        umachine_mock.sort(('hostid', 'upid'))

        #  Compute the number of galaxies in each Bolshoi-Planck halo
        bpl_halos['richness'] = compute_richness(
            bpl_halos['halo_id'], umachine_mock['hostid'])

        #  For every Bolshoi-Planck halo,
        #  compute the index of the galaxy table storing the resident central galaxy
        bpl_halos['first_galaxy_index'] = _galaxy_table_indices(
            bpl_halos['halo_id'], umachine_mock['hostid'])

        ########################################################################
        ########################################################################

        #  Transfer the colors from the z=0.1 UniverseMachine mock to the other UniverseMachine mock
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

        #  For every host halo in the AlphaQ halo catalog,
        #  find a matching halo in the Bolshoi-Planck catalog
        dlogM = 0.15
        mass_bins = 10.**np.arange(10.5, 14.5+dlogM, dlogM)
        bpl_halos['mass_bin'] = halo_bin_indices(
            mass=(bpl_halos['mvir'], mass_bins))
        alphaQ_halos['mass_bin'] = halo_bin_indices(
            mass=(alphaQ_halos['fof_halo_mass'], mass_bins))

        nhalo_min = 10
        source_halo_bin_numbers = bpl_halos['mass_bin']
        target_halo_bin_numbers = alphaQ_halos['mass_bin']
        target_halo_ids = alphaQ_halos['halo_id']
        _result = source_halo_index_selection(source_halo_bin_numbers,
            target_halo_bin_numbers, target_halo_ids, nhalo_min, mass_bins)
        source_halo_indx, matching_target_halo_ids = _result

        alphaQ_halos['source_halo_id'] = bpl_halos['halo_id'][source_halo_indx]
        alphaQ_halos['matching_mvir'] = bpl_halos['mvir'][source_halo_indx]
        alphaQ_halos['richness'] = bpl_halos['richness'][source_halo_indx]
        alphaQ_halos['first_galaxy_index'] = bpl_halos['first_galaxy_index'][source_halo_indx]

        alphaQ_halos = value_add_matched_target_halos(
            bpl_halos, alphaQ_halos, source_halo_indx)

        #  Calculate the indices of the UniverseMachine galaxies that will be selected
        nhalo_min = 10
        source_galaxies_host_halo_id = umachine_mock['hostid']
        source_halos_bin_number = bpl_halos['mass_bin']
        source_halos_halo_id = bpl_halos['halo_id']
        target_halos_bin_number = alphaQ_halos['mass_bin']
        target_halo_ids = alphaQ_halos['halo_id']
        _result = source_galaxy_selection_indices(source_galaxies_host_halo_id,
            source_halos_bin_number, source_halos_halo_id,
            target_halos_bin_number, target_halo_ids, nhalo_min, mass_bins)
        source_galaxy_indx, target_galaxy_target_halo_ids, target_galaxy_source_halo_ids = _result

        #  Assemble the output protoDC2 mock
        output_snapshot_mock = build_output_snapshot_mock(
                umachine_mock, alphaQ_halos,
                source_halo_indx, source_galaxy_indx)

        #  Use DTK code to cross-match with Galacticus galaxies
        output_snapshot_mock = remap_mock_galaxies_with_galacticus_properties(output_snapshot_mock)

        #  Write the output protoDC2 mock to disk
        output_snapshot_mock.write(output_color_mock_fname, path='data', overwrite=overwrite)


def load_umachine_z0p1_color_mock(umachine_z0p1_color_mock_fname):
    """
    """
    return Table.read(umachine_z0p1_color_mock_fname, path='data')


def load_umachine_mstar_ssfr_mock(umachine_mstar_ssfr_mock_fname, Lbox=250.):
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
    t = Table.read(alphaQ_halos_fname, path='data')
    t.rename_column('fof_halo_tag', 'halo_id')

    t.rename_column('fof_halo_center_x', 'x')
    t.rename_column('fof_halo_center_y', 'y')
    t.rename_column('fof_halo_center_z', 'z')

    t.rename_column('fof_halo_mean_vx', 'vx')
    t.rename_column('fof_halo_mean_vy', 'vy')
    t.rename_column('fof_halo_mean_vz', 'vz')
    t.remove_column('fof_halo_vel_disp')
    return t


def load_bpl_halos(bpl_halos_fname):
    """
    """
    return Table.read(bpl_halos_fname, path='data')


def um1_to_um2_matching_indices(source_mstar, source_percentile, target_mstar, target_percentile):
    """
    """
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
    X = np.vstack((umachine_z0p1_color_mock[key] for key in keys_to_match)).T
    tree = cKDTree(X)

    Y = np.vstack((umachine_mstar_ssfr_mock[key] for key in keys_to_match)).T
    nn_distinces, nn_indices = tree.query(Y)

    for key in keys_to_transfer:
        umachine_mstar_ssfr_mock[key] = umachine_z0p1_color_mock[key][nn_indices]


def remap_mock_galaxies_with_galacticus_properties(mock):
    """
    """
    return mock


def value_add_matched_target_halos(source_halos, target_halos, indices):
    """
    """
    target_halos['source_halo_id'] = source_halos['halo_id'][indices]
    target_halos['matching_mvir'] = source_halos['mvir'][indices]
    target_halos['richness'] = source_halos['richness'][indices]
    target_halos['first_galaxy_index'] = source_halos['first_galaxy_index'][indices]
    return target_halos


def build_output_snapshot_mock():
    """
    """
    raise NotImplementedError()


def calculate_host_centric_position_velocity(mock, Lbox=250.):
    """
    """
    Lbox = 250.
    xrel, vxrel = relative_positions_and_velocities(mock['x'], mock['host_halo_x'],
        v1=mock['vx'], v2=mock['host_halo_vx'], period=Lbox)
    yrel, vyrel = relative_positions_and_velocities(mock['y'], mock['host_halo_y'],
        v1=mock['vy'], v2=mock['host_halo_vy'], period=Lbox)
    zrel, vzrel = relative_positions_and_velocities(mock['z'], mock['host_halo_z'],
        v1=mock['vz'], v2=mock['host_halo_vz'], period=Lbox)

    return xrel, yrel, zrel, vxrel, vyrel, vzrel

