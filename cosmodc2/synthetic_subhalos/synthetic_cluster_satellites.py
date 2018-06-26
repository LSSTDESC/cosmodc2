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

    return nn_indices
