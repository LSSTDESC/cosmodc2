import numpy as np
from scipy.spatial import cKDTree
from astropy.table import Table
from halotools.empirical_models import NFWPhaseSpace


__all__ = ('nearby_hostmass_selection_indices', 'calculate_synthetic_richness',
        'create_synthetic_cluster_satellites')


def nearby_hostmass_selection_indices(hostmass, desired_hostmass):
    """
    Parameters
    ----------
    hostmass : ndarray
        Numpy array of shape (ntot_available_sats, ) storing the host halo mass
        of every satellite in the selection sample

    desired_hostmass : ndarray
        Numpy array of shape (ntot_desired_fakesats, ) storing the host halo mass
        of every desired synthetic satellite

    Returns
    -------
    selection_indices : ndarray
        Numpy integer array of shape (ntot_desired_fakesats, ) storing the indices
        that select the satellites from the original sample
    """
    noisy_hostmass = 10**np.random.normal(loc=np.log10(hostmass), scale=0.02)
    X1 = np.vstack((noisy_hostmass, )).T
    tree = cKDTree(X1)

    X2 = np.vstack((desired_hostmass, )).T
    nn_distinces, nn_indices = tree.query(X2)
    nn_indices = np.minimum(len(hostmass)-1, nn_indices)
    nn_indices = np.maximum(0, nn_indices)

    return nn_indices


def calculate_synthetic_richness(halo_richness, logmhalo,
        cluster_satboost_logm_table, cluster_satboost_table, **kwargs):
    """
    Parameters
    ----------
    halo_richness : ndarray
        Numpy array of shape (nhalos, ) storing the richness of each halo

    logmhalo : ndarray
        Numpy array of shape (nhalos, ) storing the log mass of each halo

    Returns
    -------
    synthetic_richness : ndarray
        Numpy integer array of shape (nhalos, ) storing the synthetic richness
    """
    boost_factor = np.interp(logmhalo, cluster_satboost_logm_table, cluster_satboost_table)
    return np.array(halo_richness*boost_factor, dtype=int)


def create_synthetic_cluster_satellites(mock, Lbox=256.,
        cluster_satboost_logm_table=[13.5, 13.75, 14],
        cluster_satboost_table=[0., 0.15, 0.2], **kwargs):
    """
    """
    host_halo_id, idx, counts = np.unique(
        mock['target_halo_id'], return_counts=True, return_index=True)
    host_mass = mock['target_halo_mass'][idx]
    host_richness = counts

    synthetic_richness = calculate_synthetic_richness(
        host_richness, np.log10(host_mass), **kwargs)
    if np.sum(synthetic_richness) > 0:
        synthetic_hostmass = np.repeat(host_mass, synthetic_richness)

        sat_sample_mask = mock['upid'] != -1
        selection_indices = nearby_hostmass_selection_indices(
            mock['target_halo_mass'][sat_sample_mask], synthetic_hostmass)

        sats = Table()
        for key in mock.keys():
            sats[key] = mock[key][sat_sample_mask][selection_indices]

        nfw = NFWPhaseSpace()
        nfw_sats = nfw.mc_generate_nfw_phase_space_points(
            mass=sats['target_halo_mass'])
        sats['host_centric_x'] = nfw_sats['x']
        sats['host_centric_y'] = nfw_sats['y']
        sats['host_centric_z'] = nfw_sats['z']
        sats['host_centric_vx'] = nfw_sats['vx']
        sats['host_centric_vy'] = nfw_sats['vy']
        sats['host_centric_vz'] = nfw_sats['vz']

        sats['x'] = sats['host_centric_x'] + sats['target_halo_x']
        sats['y'] = sats['host_centric_y'] + sats['target_halo_y']
        sats['z'] = sats['host_centric_z'] + sats['target_halo_z']
        sats['vx'] = sats['host_centric_vx'] + sats['target_halo_vx']
        sats['vy'] = sats['host_centric_vy'] + sats['target_halo_vy']
        sats['vz'] = sats['host_centric_vz'] + sats['target_halo_vz']

        sats['x'] = np.mod(sats['x'], Lbox)
        sats['y'] = np.mod(sats['y'], Lbox)
        sats['z'] = np.mod(sats['z'], Lbox)

        return sats
    else:
        return Table()
