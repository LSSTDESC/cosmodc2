"""
"""
import numpy as np


sdss_completeness_table_magr = [-17.75, -18.25, -18.5, -19, -19.5, -20, -20.5, -21, -21.5, -22]
sdss_completeness_table_redshift = [0.0175, 0.025, 0.03, 0.035, 0.045, 0.055, 0.07, 0.08, 0.09, 0.1]


__all__ = ('retrieve_sdss_sample_mask', )


def sdss_completeness_cut(magr_max, magr_table=sdss_completeness_table_magr,
            redshift=sdss_completeness_table_redshift):
    """
    """
    assert np.all(np.diff(magr_table) < 0), "magnitude table should be decreasing {0}".format(magr_table)
    assert np.all(np.diff(redshift) > 0), "redshift table should be increasing"

    return np.interp(magr_max, magr_table[::-1], redshift[::-1])


def retrieve_sdss_sample_mask(sdss_redshift, sdss_magr, magr_min, magr_max,
        magr_table=sdss_completeness_table_magr):
    """
    """
    msg = "magr_min = {0}, magr_max = {1}"
    assert magr_min < magr_max, msg.format(magr_min, magr_max)

    if magr_min < np.min(magr_table):
        magr_min = -np.inf
    elif magr_max > np.max(magr_table):
        magr_max = np.max(magr_table)
        magr_min = -np.inf

    mask = sdss_redshift < sdss_completeness_cut(magr_max)
    mask *= (sdss_magr > magr_min)
    mask *= (sdss_magr < magr_max)
    return mask
