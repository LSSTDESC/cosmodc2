"""
"""
import numpy as np
from halotools.mock_observables import return_xyz_formatted_array, wp


rp_zehavi = np.array(
    (0.17, 0.27, 0.42, 0.67, 1.1, 1.7, 2.7, 4.2, 6.7, 10.6, 16.9, 26.8))

log_rp_zehavi = np.log10(rp_zehavi)
dlog_rp = np.mean(np.diff(log_rp_zehavi))

log_rp_low = log_rp_zehavi.min() - dlog_rp/2.
log_rp_high = log_rp_zehavi.max() + dlog_rp/2.
log_rp_bins = np.arange(log_rp_low, log_rp_high, dlog_rp)
rp_bins = 10**log_rp_bins
rp_mids = 10**(0.5*(np.log10(rp_bins[:-1]) + np.log10(rp_bins[1:])))
pi_max = 40.


__all__ = ('cumulative_nd', 'zehavi_wp')


def cumulative_nd(magr, volume, assumed_littleh_for_magr, lumthresh_h1p0):
    """
    """
    magr_h1p0 = magr - 5.*np.log10(assumed_littleh_for_magr)
    counts = np.fromiter((np.count_nonzero(magr_h1p0 < lum)
        for lum in lumthresh_h1p0), dtype=int)
    return counts/float(volume)


def zehavi_wp(x, y, z, vz, period, magr, magr_thresh, assumed_littleh_for_magr,
            subsample='all', gr_colors=None):
    """
    """
    magr_h1p0 = magr - 5*np.log10(assumed_littleh_for_magr)

    mask = magr_h1p0 < magr_thresh
    if subsample != 'all':
        gr_cut = 0.21 - 0.03*magr_h1p0
        assert gr_colors is not None, "Must pass g-r colors when subsample = {0}".format(subsample)
        blue_mask = gr_colors < gr_cut
        if subsample == 'blue':
            mask *= blue_mask
        elif subsample == 'red':
            mask *= ~blue_mask

    ngals = np.count_nonzero(mask)

    downsampling_factor = 1.e5/float(ngals)
    if downsampling_factor < 1.:
        # print("...downsampling sample from original counts = {0:.2e}".format(ngals))
        mask *= (np.random.rand(len(x)) < downsampling_factor)
        ngals = np.count_nonzero(mask)

    pos = return_xyz_formatted_array(x, y, z, velocity=vz,
            velocity_distortion_dimension='z', period=period, mask=mask)

    print("...calculating wp for {0:.2e} galaxies and Mr < {1:.1f}".format(ngals, magr_thresh))
    wp_sample = wp(pos, rp_bins, pi_max, period=period, num_threads='max')
    return rp_mids, wp_sample









