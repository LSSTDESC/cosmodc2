""" Polynomial fits to the mean color vs. redshift scaling relation
of DES red sequence galaxies provided by Joe DeRose and Eli Rykoff.
"""
import numpy as np


c3_gr, c2_gr, c1_gr, c0_gr = 2.91, -6.63, 4.98, 0.48
c3_ri, c2_ri, c1_ri, c0_ri = -3.54, 5.66, -1.42, 0.49
c3_iz, c2_iz, c1_iz, c0_iz = 1.83, -1.90, 0.79, 0.21

__all__ = ('mean_des_red_sequence_gr_color_vs_redshift',
    'mean_des_red_sequence_ri_color_vs_redshift',
    'mean_des_red_sequence_iz_color_vs_redshift')


def mean_des_red_sequence_gr_color_vs_redshift(redshift):
    """
    Parameters
    ----------
    redshift : ndarray
        Numpy array of shape (npts, )

    Returns
    -------
    mean_observed_color : ndarray
        Numpy array of shape (npts, )

    """
    z = np.atleast_1d(redshift)
    return c0_gr + c1_gr*z + c2_gr*z**2 + c3_gr*z**3


def mean_des_red_sequence_ri_color_vs_redshift(redshift):
    """
    Parameters
    ----------
    redshift : ndarray
        Numpy array of shape (npts, )

    Returns
    -------
    mean_observed_color : ndarray
        Numpy array of shape (npts, )
    """
    z = np.atleast_1d(redshift)
    return c0_ri + c1_ri*z + c2_ri*z**2 + c3_ri*z**3


def mean_des_red_sequence_iz_color_vs_redshift(redshift):
    """
    Parameters
    ----------
    redshift : ndarray
        Numpy array of shape (npts, )

    Returns
    -------
    mean_observed_color : ndarray
        Numpy array of shape (npts, )
    """
    z = np.atleast_1d(redshift)
    return c0_iz + c1_iz*z + c2_iz*z**2 + c3_iz*z**3


def _read_fits_files_provided_by_joe(fname_redshift, fname_mean_color):
    """ This function is entirely for internal use while fitting the relations
    with convenient polynomial approximations.

    For an example showing how the polynomial fitting was done:

    c3_ri, c2_ri, c1_ri, c0_ri = np.polyfit(z_table, mean_colors['ri'], 3)
    """
    from astropy.table import Table
    from astropy.io import fits

    with fits.open(fname_redshift) as hdulist:
        redshift_table = Table(np.array(hdulist[0].data))
    z_table = np.copy(
        np.array(list(redshift_table[key] for key in redshift_table.keys())).flatten()[:-1])

    #  colors are g-r, r-i and i-z
    with fits.open(fname_mean_color) as hdulist:
        des_mean_colors = Table(np.array(hdulist[0].data))
    des_mean_colors.rename_column('col0', 'gr')
    des_mean_colors.rename_column('col1', 'ri')
    des_mean_colors.rename_column('col2', 'iz')
    mean_colors = np.copy(des_mean_colors[:-1])

    return z_table, mean_colors
