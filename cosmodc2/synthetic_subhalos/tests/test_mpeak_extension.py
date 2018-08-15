"""
"""
import pytest
import numpy as np
from scipy.stats import powerlaw
from ..extend_subhalo_mpeak_range import model_extended_mpeak
from ..extend_subhalo_mpeak_range import map_mstar_onto_lowmass_extension


@pytest.mark.xfail
def test1():
    mpeak = 10**(5*(1-powerlaw.rvs(2, size=40000)) + 10.)
    desired_logm_completeness = 9.5
    corrected_mpeak, mpeak_extension = model_extended_mpeak(mpeak, desired_logm_completeness)
    assert np.all(mpeak_extension < corrected_mpeak.min())
    assert np.all(mpeak_extension >= 10**desired_logm_completeness)
    assert np.any(mpeak_extension < 10**(desired_logm_completeness+0.02))


def test_nonzero_synthetic_stellar_mass():
    """ Regression test for GitHub Issue #39 - https://github.com/LSSTDESC/cosmodc2/issues/39
    """
    nreal, nfake = int(1e5), int(5e5)
    corrected_mpeak = 10**np.random.uniform(9.5, 12, nreal)
    obs_sm_orig = corrected_mpeak/100.
    mpeak_extension = 10**np.random.uniform(9, 12, nfake)
    new_mstar_real, new_mstar_synthetic = map_mstar_onto_lowmass_extension(
        corrected_mpeak, obs_sm_orig, mpeak_extension)
    assert np.all(new_mstar_synthetic) > 0
