"""
"""
from scipy.spatial import cKDTree
import numpy as np
from astropy.table import Table
from galsampler import matched_value_selection_indices, source_galaxy_selection_indices


def write_sdss_restframe_color_snapshot_mocks_to_disk(
            umachine_z0p1_color_mock_fname, protoDC2_fof_halo_catalog_fname_list,
            umachine_mstar_ssfr_mock_fname_list, bolshoi_planck_halo_catalog_fname_list,
            output_color_mock_fname_list, redshift_list, overwrite=False):
    """
    Function writes to disk a set of extragalactic snapshot catalogs by GalSampling UniverseMachine.

    Parameters
    ----------
    umachine_z0p1_color_mock_fname : string
        Absolute path to the z=0.1 UniverseMachine baseline mock that includes
        M*, SFR, Mr, g-r, r-i.

    protoDC2_fof_halo_catalog_fname_list : list of strings
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

    bolshoi_planck_halo_catalog_fname_list : list of strings
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

    gen = zip(protoDC2_fof_halo_catalog_fname_list, umachine_mstar_ssfr_mock_fname_list,
            bolshoi_planck_halo_catalog_fname_list, output_color_mock_fname_list, redshift_list)

    for fname1, fname2, fname3, output_color_mock_fname, redshift in gen:
        print("...working on creating {0}".format(output_color_mock_fname))

        #  Load all three catalogs into memory
        protoDC2_fof_halo_catalog = load_protoDC2_fof_halo_catalog(fname1)
        umachine_mstar_ssfr_mock = load_umachine_mstar_ssfr_mock(fname2)
        bolshoi_planck_halo_catalog = load_bolshoi_planck_halo_catalog(fname3)

        #  Transfer the colors from the z=0.1 UniverseMachine mock to the other UniverseMachine mock
        umachine_mstar_ssfr_mock_with_colors = transfer_colors_to_umachine_mstar_ssfr_mock(
            umachine_mstar_ssfr_mock, umachine_z0p1_color_mock, redshift)

        #  For every host halo in the halo catalog hosting the protoDC2 galaxies,
        #  find a matching halo in halo catalog hosting the UniverseMachine galaxies
        source_halo_indx = matched_value_selection_indices(
            bolshoi_planck_halo_catalog['mvir'],
            protoDC2_fof_halo_catalog['fof_mass'])

        protoDC2_fof_halo_catalog = value_add_matched_target_halos(
            bolshoi_planck_halo_catalog, protoDC2_fof_halo_catalog, source_halo_indx)

        #  Calculate the indices of the UniverseMachine galaxies that will be selected
        #  find a matching halo in halo catalog hosting the UniverseMachine galaxies
        source_galaxy_indx = source_galaxy_selection_indices(
            protoDC2_fof_halo_catalog, source_halo_indx)

        #  Assemble the output protoDC2 mock
        output_snapshot_mock = build_output_snapshot_mock(
                umachine_mstar_ssfr_mock_with_colors, protoDC2_fof_halo_catalog,
                source_halo_indx, source_galaxy_indx)

        #  Use DTK code to cross-match with Galacticus galaxies
        output_snapshot_mock = remap_mock_galaxies_with_galacticus_properties(output_snapshot_mock)

        #  Write the output protoDC2 mock to disk
        output_snapshot_mock.write(output_color_mock_fname, path='data', overwrite=overwrite)


def load_umachine_z0p1_color_mock(umachine_z0p1_color_mock_fname):
    """
    """
    return Table.read(umachine_z0p1_color_mock_fname, path='data')


def load_umachine_mstar_ssfr_mock(umachine_mstar_ssfr_mock_fname):
    """
    """
    return Table.read(umachine_mstar_ssfr_mock_fname, path='data')


def load_protoDC2_fof_halo_catalog(protoDC2_fof_halo_catalog_fname):
    """
    """
    return Table.read(protoDC2_fof_halo_catalog_fname, path='data')


def load_bolshoi_planck_halo_catalog(bolshoi_planck_halo_catalog_fname):
    """
    """
    return Table.read(bolshoi_planck_halo_catalog_fname, path='data')


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


def remap_mock_galaxies_with_galacticus_properties():
    """
    """
    raise NotImplementedError()


def value_add_matched_target_halos():
    """
    """
    raise NotImplementedError()


def build_output_snapshot_mock():
    """
    """
    raise NotImplementedError()


