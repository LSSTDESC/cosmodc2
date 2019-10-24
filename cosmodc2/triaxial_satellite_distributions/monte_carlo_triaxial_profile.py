"""
"""
import numpy as np
from .monte_carlo_nfw import nfw_profile_realization
from halotools.utils import elementwise_norm
from halotools.utils import rotation_matrices_from_vectors, rotate_vector_collection


def generate_triaxial_satellite_distribution(
            host_conc, host_Ax, host_Ay, host_Az, b_to_a, c_to_a, seed=43):
    """ Generate xyz positions according to an ellipsoidal NFW profile.

    Parameters
    ----------
    host_conc : float or ndarray
        Concentration of the parent halo

    host_Ax : float or ndarray
        x-coordinate of the principal axis

    host_Ay : float or ndarray
        y-coordinate of the principal axis

    host_Az : float or ndarray
        z-coordinate of the principal axis

    b_to_a : float or ndarray
        B/A axis ratio. Should respect 0 < B/A < 1

    c_to_a : float or ndarray
        C/A axis ratio. Should respect 0 < C/A < 1

    Returns
    -------
    x, y, z : ndarrays
        Arrays storing xyz positions

    Notes
    -----
    The normalization of the input vector host_A determines the length of the principal axis

    """
    host_conc, host_Ax, host_Ay, host_Az, b_to_a, c_to_a = _format_args_as_ndarrays(
            host_conc, host_Ax, host_Ay, host_Az, b_to_a, c_to_a)
    npts = host_conc.size

    x, y, z = _mc_unit_ellipsoid(npts, c_to_a, b_to_a, host_Ax, host_Ay, host_Az, seed=seed)
    r = nfw_profile_realization(host_conc, seed=seed-1)
    return r*x, r*y, r*z


def _mc_unit_ellipsoid(Npts, c_to_a=1., b_to_a=1., Ax=0., Ay=1., Az=0., seed=43):
    """
    """
    cos_t = np.random.RandomState(seed).uniform(-1., 1., Npts)
    phi = np.random.RandomState(seed+1).uniform(0, 2*np.pi, Npts)
    sin_t = np.sqrt((1.-cos_t*cos_t))

    c_to_a = np.zeros(Npts) + c_to_a
    b_to_a = np.zeros(Npts) + b_to_a
    Ax = np.zeros(Npts) + Ax
    Ay = np.zeros(Npts) + Ay
    Az = np.zeros(Npts) + Az
    A_vector = np.vstack((Ax, Ay, Az)).T
    A_length = elementwise_norm(A_vector)

    B_length = A_length*b_to_a
    C_length = A_length*c_to_a
    c_to_b = C_length/B_length

    x = (C_length/c_to_a)*sin_t * np.cos(phi)
    y = (C_length/c_to_b)*sin_t * np.sin(phi)
    z = C_length*cos_t
    points_X_frame = np.vstack((x, y, z)).T

    X_vector = np.tile((1., 0., 0.), Npts).reshape((Npts, 3))
    rotmat_X_to_A = rotation_matrices_from_vectors(X_vector, A_vector)
    points_A_frame = rotate_vector_collection(rotmat_X_to_A, points_X_frame)
    return points_A_frame[:, 0], points_A_frame[:, 1], points_A_frame[:, 2]


def _format_args_as_ndarrays(*args):
    args = [np.atleast_1d(x) for x in args]
    npts = int(max((x.size for x in args)))
    args = [np.zeros(npts) + arg for arg in args]
    return args
