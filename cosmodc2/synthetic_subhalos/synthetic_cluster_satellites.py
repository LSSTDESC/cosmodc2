import numpy as np
from scipy.spatial import cKDTree


__all__ = ('nearby_hostmass_selection_indices', )


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
        cluster_satboost_logm_table=[13.5, 13.75, 14],
        cluster_satboost_table=[0., 0.1, 0.2], **kwargs):
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








