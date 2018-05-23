"""
"""
import numpy as np
from halotools.utils import crossmatch
from halotools.mock_observables import relative_positions_and_velocities
from galsampler.utils import compute_richness
from galsampler.source_galaxy_selection import _galaxy_table_indices


__all__ = ('calculate_value_added_galaxy_halo_catalogs',
        'calculate_host_centric_position_velocity')


def calculate_value_added_galaxy_halo_catalogs(
        galaxies, halos, umachine_color_mock, Lbox):
    """
    """
    #  Correct for edge case where position is exactly on box boundary
    halos['x'] = np.where(halos['x'] == Lbox, 0., halos['x'])
    halos['y'] = np.where(halos['y'] == Lbox, 0., halos['y'])
    halos['z'] = np.where(halos['z'] == Lbox, 0., halos['z'])

    #  Throw out the small number of galaxies for which there is no matching host
    idxA, idxB = crossmatch(galaxies['hostid'], halos['halo_id'])
    galaxies = galaxies[idxA]

    #  Compute host halo position for every UniverseMachine galaxy
    galaxies['host_halo_x'] = np.nan
    galaxies['host_halo_y'] = np.nan
    galaxies['host_halo_z'] = np.nan
    galaxies['host_halo_vx'] = np.nan
    galaxies['host_halo_vy'] = np.nan
    galaxies['host_halo_vz'] = np.nan
    galaxies['host_halo_mvir'] = np.nan

    galaxies['host_halo_x'][idxA] = halos['x'][idxB]
    galaxies['host_halo_y'][idxA] = halos['y'][idxB]
    galaxies['host_halo_z'][idxA] = halos['z'][idxB]
    galaxies['host_halo_vx'][idxA] = halos['vx'][idxB]
    galaxies['host_halo_vy'][idxA] = halos['vy'][idxB]
    galaxies['host_halo_vz'][idxA] = halos['vz'][idxB]
    galaxies['host_halo_mvir'][idxA] = halos['mvir'][idxB]

    #  Compute halo-centric position for every UniverseMachine galaxy
    result = calculate_host_centric_position_velocity(galaxies, Lbox)
    xrel, yrel, zrel, vxrel, vyrel, vzrel = result
    galaxies['host_centric_x'] = xrel
    galaxies['host_centric_y'] = yrel
    galaxies['host_centric_z'] = zrel
    galaxies['host_centric_vx'] = vxrel
    galaxies['host_centric_vy'] = vyrel
    galaxies['host_centric_vz'] = vzrel

    #  Sort the mock by host halo ID, putting centrals first within each grouping
    galaxies.sort(('hostid', 'upid'))

    #  Compute the number of galaxies in each Bolshoi-Planck halo
    halos['richness'] = compute_richness(
        halos['halo_id'], galaxies['hostid'])

    #  For every Bolshoi-Planck halo, compute the index of the galaxy table
    #  storing the central galaxy, reserving -1 for unoccupied halos
    halos['first_galaxy_index'] = _galaxy_table_indices(
        halos['halo_id'], galaxies['hostid'])

    ########################################################################
    #  Transfer the colors from the z=0.1 UniverseMachine mock
    #  to the other UniverseMachine mock
    ########################################################################

    #  For every galaxy in galaxies, find a galaxy in umachine_color_mock
    #  with a closely matching stellar mass and SFR-percentile
    source_mstar = umachine_color_mock['obs_sm']
    source_percentile = umachine_color_mock['sfr_percentile_fixed_sm']
    target_mstar = galaxies['obs_sm']
    target_percentile = galaxies['sfr_percentile_fixed_sm']
    um_matching_indx = um1_to_um2_matching_indices(
        source_mstar, source_percentile, target_mstar, target_percentile)

    keys_to_transfer = ('restframe_extincted_sdss_abs_magr',
            'restframe_extincted_sdss_gr', 'restframe_extincted_sdss_ri')
    for key in keys_to_transfer:
        galaxies[key] = umachine_color_mock[key][um_matching_indx]

    return galaxies, halos


def um1_to_um2_matching_indices(source_mstar, source_percentile,
            target_mstar, target_percentile):
    """
    """
    from scipy.spatial import cKDTree

    X1 = np.vstack((source_mstar, source_percentile)).T
    tree = cKDTree(X1)

    X2 = np.vstack((target_mstar, target_percentile)).T
    nn_distinces, nn_indices = tree.query(X2)

    return nn_indices


def calculate_host_centric_position_velocity(mock, Lbox):
    """
    """
    xrel, vxrel = relative_positions_and_velocities(
        mock['x'], mock['host_halo_x'],
        v1=mock['vx'], v2=mock['host_halo_vx'], period=Lbox)
    yrel, vyrel = relative_positions_and_velocities(
        mock['y'], mock['host_halo_y'],
        v1=mock['vy'], v2=mock['host_halo_vy'], period=Lbox)
    zrel, vzrel = relative_positions_and_velocities(
        mock['z'], mock['host_halo_z'],
        v1=mock['vz'], v2=mock['host_halo_vz'], period=Lbox)

    return xrel, yrel, zrel, vxrel, vyrel, vzrel

