"""
"""
import numpy as np
from halotools.utils import resample_x_to_match_y, sliding_conditional_percentile


def gr_ri_matching_indices(r_in, gr_in, ri_in, gr_desired, ri_desired,
            gr_bins, ri_bins, weights_gr_ri=(0.75, 0.25), nwindow=201):
    """
    """
    idx_fake_gr = resample_x_to_match_y(gr_in, gr_desired, gr_bins)
    gr_out = gr_in[idx_fake_gr]
    ri_out_temp = ri_in[idx_fake_gr]
    idx_fake_ri = resample_x_to_match_y(ri_out_temp, ri_desired, ri_bins)
    ri_out = ri_out_temp[idx_fake_ri]

    gr_out_p = sliding_conditional_percentile(r_in, gr_out, nwindow)
    ri_out_p = sliding_conditional_percentile(r_in, ri_out, nwindow)
    weighted_gr_ri_out = weights_gr_ri[0]*gr_out_p + weights_gr_ri[1]*ri_out_p
    weighted_gr_ri_out_p = sliding_conditional_percentile(r_in, weighted_gr_ri_out, nwindow)

    idx_sorted_weighted_gr_ri_out_p = np.argsort(weighted_gr_ri_out_p)
    return idx_sorted_weighted_gr_ri_out_p


def remap_weighted_gr_ri(r_in, gr_in, ri_in, gr_desired, ri_desired,
                         gr_bins, ri_bins, weights_gr_ri=(0.75, 0.25), nwindow=201):
    """
    Rescale g-r and r-i to match some other desired distributions,
    preserving the rank-order of some weighted combination of g-r and r-i.

    Parameters
    ----------
    r_in : ndarray
        Numpy array of shape (ngals_in, ) storing r-band Absolute magnitude
        of galaxy sample which needs its colors corrected

    gr_in : ndarray
        Numpy array of shape (ngals_in, ) storing g-r color
        of galaxy sample which needs its colors corrected

    ri_in : ndarray
        Numpy array of shape (ngals_in, ) storing r-i color
        of galaxy sample which needs its colors corrected

    gr_desired : ndarray
        Numpy array of shape (ngals_data, ) storing target g-r distribution

    ri_desired : ndarray
        Numpy array of shape (ngals_data, ) storing target r-i distribution

    gr_bins : ndarray
        Numpy array of shape (num_gr_bins, ) storing the bin edges in g-r
        used to define the PDF to which `gr_in` will be matched

    ri_bins : ndarray
        Numpy array of shape (num_ri_bins, ) storing the bin edges in r-i
        used to define the PDF to which `ri_in` will be matched

    weights_gr_ri : tuple, optional
        Two-element tuple storing (gr_weight, ri_weight), which regulates how
        how the g-r vs. r-i rank-ordering is in correspondence with the data
        Default is (0.75, 0.25)

    nwindow : int, optional
        Sliding window length used to estimate the conditional CDF. Default is 201.

    Returns
    -------
    corrected_gr : ndarray
        Numpy array of shape (ngals_in, ) storing remapped g-r values

    corrected_ri : ndarray
        Numpy array of shape (ngals_in, ) storing remapped r-i values

    """
    idx_fake_gr = resample_x_to_match_y(gr_in, gr_desired, gr_bins)
    gr_out = gr_in[idx_fake_gr]
    ri_out_temp = ri_in[idx_fake_gr]
    idx_fake_ri = resample_x_to_match_y(ri_out_temp, ri_desired, ri_bins)
    ri_out = ri_out_temp[idx_fake_ri]

    gr_in_p = sliding_conditional_percentile(r_in, gr_in, nwindow)
    ri_in_p = sliding_conditional_percentile(r_in, ri_in, nwindow)
    weighted_gr_ri_in = weights_gr_ri[0]*gr_in_p + weights_gr_ri[1]*ri_in_p
    weighted_gr_ri_in_p = sliding_conditional_percentile(r_in, weighted_gr_ri_in, nwindow)

    gr_out_p = sliding_conditional_percentile(r_in, gr_out, nwindow)
    ri_out_p = sliding_conditional_percentile(r_in, ri_out, nwindow)
    weighted_gr_ri_out = weights_gr_ri[0]*gr_out_p + weights_gr_ri[1]*ri_out_p
    weighted_gr_ri_out_p = sliding_conditional_percentile(r_in, weighted_gr_ri_out, nwindow)

    idx_sorted_weighted_gr_ri_in_p = np.argsort(weighted_gr_ri_in_p)
    idx_sorted_weighted_gr_ri_out_p = np.argsort(weighted_gr_ri_out_p)

    corrected_gr = np.zeros_like(gr_out)
    corrected_ri = np.zeros_like(ri_out)
    corrected_gr[idx_sorted_weighted_gr_ri_in_p] = gr_out[idx_sorted_weighted_gr_ri_out_p]
    corrected_ri[idx_sorted_weighted_gr_ri_in_p] = ri_out[idx_sorted_weighted_gr_ri_out_p]

    return corrected_gr, corrected_ri
