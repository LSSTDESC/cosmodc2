"""
"""
import numpy as np
from astropy.table import Table
from astropy.utils.misc import NumpyRNGContext


__all__ = ('random_linear_combo_spectra', 'matching_spectrum_search')


default_seed = 43


def random_linear_combo_spectra(spectra, num_random=None,
            coeff_low=0, coeff_high=2, seed=default_seed):
    ngals = len(spectra)
    if num_random is None:
        num_random = ngals

    a = np.arange(ngals)
    with NumpyRNGContext(seed):
        indx1 = np.random.choice(a, size=num_random)
        indx2 = np.random.choice(a, size=num_random)
        w1 = np.random.uniform(coeff_low, coeff_high, num_random)
        w2 = np.random.uniform(coeff_low, coeff_high, num_random)

    result = Table()
    for key in ('u', 'g', 'r', 'i', 'z', 'age', 'metallicity'):
        result[key] = spectra[key][indx1]*w1 + spectra[key][indx2]*w2
    result['specID1'] = indx1
    result['specID2'] = indx2
    result['w1'] = w1
    result['w2'] = w2
    return result


def matching_spectrum_search(gr, ri, fake_sed_library):
    """
    """
    from scipy.spatial import cKDTree

    gr_tree = fake_sed_library['g']-fake_sed_library['r']
    ri_tree = fake_sed_library['r']-fake_sed_library['i']
    sed_tree = cKDTree(np.vstack((gr_tree, ri_tree)).T)

    mock_gr_ri = np.vstack((gr, ri)).T
    d, idx = sed_tree.query(mock_gr_ri, k=1)
    return d, idx
