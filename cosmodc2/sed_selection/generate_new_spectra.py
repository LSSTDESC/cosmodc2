"""
"""
import numpy as np
from astropy.table import Table


__all__ = ('random_linear_combo_spectra', )


def random_linear_combo_spectra(spectra, num_random=None, coeff_low=0, coeff_high=2):
    ngals = len(spectra)
    if num_random is None:
        num_random = ngals

    a = np.arange(ngals)
    indx1 = np.random.choice(a, size=num_random)
    indx2 = np.random.choice(a, size=num_random)
    w1 = np.random.uniform(coeff_low, coeff_high, num_random)
    w2 = np.random.uniform(coeff_low, coeff_high, num_random)

    result = Table()
    for key in ('u', 'g', 'r', 'i', 'z'):
        result[key] = spectra[key][indx1]*w1 + spectra[key][indx2]*w2
    result['specID1'] = indx1
    result['specID2'] = indx2
    result['w1'] = w1
    result['w2'] = w2
    return result
