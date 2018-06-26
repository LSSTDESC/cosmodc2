"""
"""
import numpy as np
from halotools.utils import unsorting_indices
from astropy.table import Table
from scipy.stats import norm


default_mpeak_mstar_fit_low_mpeak, default_mpeak_mstar_fit_high_mpeak = 11, 11.5
default_desired_logm_completeness = 9.75

__all__ = ('model_extended_mpeak', 'map_mstar_onto_lowmass_extension', 'create_synthetic_mock')


def model_extended_mpeak(mpeak, desired_logm_completeness=default_desired_logm_completeness,
            logm_min_fit=11.75, logm_max_fit=12.25, Lbox=256.):
    """ Given an input set of subhalo mpeak values, and a desired completeness limit,
    fit the input distribution with a power law at the low mass end,
    extrapolate subhalo abundance to lower masses, and return a set of subhalos
    whose abundance obeys the best-fit power law down to the desired completeness limit.

    Parameters
    ----------
    mpeak : ndarray
        Numpy array of shape (nsubs_orig, )

    new_logmpeak_low : float
        Desired completeness limit in log10 units

    Returns
    -------
    corrected_mpeak : ndarray
        Numpy array of shape (nsubs_orig, ). Values greater than the midpoint
        of the fitted range will be unchanged from the original mpeak. Values less than this
        will be altered to correct for the power law departure

    mpeak_extension : ndarray
        Numpy array of shape (num_new_subs, ) storing values of Mpeak for synthetic subhalos

    Examples
    --------
    >>> from scipy.stats import powerlaw
    >>> mpeak = 10**(5*(1-powerlaw.rvs(2, size=40000)) + 10.)
    >>> desired_logm_completeness = 9.5
    >>> corrected_mpeak, mpeak_extension = model_extended_mpeak(mpeak, desired_logm_completeness)
    """
    logmpeak = np.log10(mpeak)
    idx_sorted = np.argsort(logmpeak)[::-1]
    sorted_logmpeak = logmpeak[idx_sorted]

    Vbox = Lbox**3
    npts_total = len(logmpeak)
    logndarr = np.log10(np.arange(1, 1 + npts_total)/Vbox)

    logm_mid = 0.5*(logm_min_fit + logm_max_fit)

    mask = sorted_logmpeak >= logm_min_fit
    mask &= sorted_logmpeak < logm_max_fit

    c1, c0 = np.polyfit(sorted_logmpeak[mask][::100], logndarr[mask][::100], deg=1)
    model_lognd = c0 + c1*sorted_logmpeak

    model_logmpeak = np.interp(logndarr, model_lognd, sorted_logmpeak)
    model_logmpeak[sorted_logmpeak > logm_mid] = sorted_logmpeak[sorted_logmpeak > logm_mid]

    lognd_extension_max = c0 + c1*desired_logm_completeness
    new_ngals_tot = int((10**lognd_extension_max)*Vbox)
    logndarr_extension = np.log10(np.arange(1 + npts_total, new_ngals_tot)/Vbox)
    logmpeak_extension = (logndarr_extension - c0)/c1
    mpeak_extension = 10**logmpeak_extension

    corrected_mpeak = 10**model_logmpeak[unsorting_indices(idx_sorted)]
    return corrected_mpeak, mpeak_extension


def fit_lowmass_mstar_mpeak_relation(mpeak_orig, mstar_orig,
            mpeak_mstar_fit_low_mpeak=default_mpeak_mstar_fit_low_mpeak,
            mpeak_mstar_fit_high_mpeak=default_mpeak_mstar_fit_high_mpeak):
    """
    """
    mid = 0.5*(mpeak_mstar_fit_low_mpeak + mpeak_mstar_fit_high_mpeak)
    mask = (mpeak_orig >= 10**mpeak_mstar_fit_low_mpeak)
    mask &= (mpeak_orig < 10**mpeak_mstar_fit_high_mpeak)
    #  Add noise to mpeak to avoid particle discreteness effects in the fit
    _x = np.random.normal(loc=np.log10(mpeak_orig[mask])-mid, scale=0.002)
    _y = np.log10(mstar_orig[mask])
    c1, c0 = np.polyfit(_x, _y, deg=1)
    return c0, c1, mid


def map_mstar_onto_lowmass_extension(corrected_mpeak, obs_sm_orig, mpeak_extension,
            c0=9., c1=2.2, mpeak_mstar_fit_low_mpeak=default_mpeak_mstar_fit_low_mpeak,
            mpeak_mstar_fit_high_mpeak=default_mpeak_mstar_fit_high_mpeak, synthetic_scatter=0.4,
            **kwargs):
    """
    c1 controls the new low-mass slope. Smaller values of c1 puts more stellar mass
    into galaxies in low-mass subhalos.
    """
    mid = 0.5*(mpeak_mstar_fit_low_mpeak + mpeak_mstar_fit_high_mpeak)
    composite_mpeak = np.concatenate((corrected_mpeak, mpeak_extension))
    new_median_logsm = c0 + c1*(np.log10(composite_mpeak)-mid)

    new_mstar_lowmass = 10**np.random.normal(loc=new_median_logsm, scale=synthetic_scatter)

    reassign_mstar_prob = np.interp(np.log10(composite_mpeak),
        [mpeak_mstar_fit_low_mpeak, mpeak_mstar_fit_high_mpeak], [1, 0])
    reassign_mstar_mask = np.random.rand(len(composite_mpeak)) < reassign_mstar_prob

    new_mstar = np.zeros_like(composite_mpeak)
    new_mstar[:len(obs_sm_orig)] = obs_sm_orig
    new_mstar[reassign_mstar_mask] = new_mstar_lowmass[reassign_mstar_mask]

    new_mstar_real = new_mstar[:len(obs_sm_orig)]
    new_mstar_synthetic = new_mstar[len(obs_sm_orig):]
    return new_mstar_real, new_mstar_synthetic


def create_synthetic_mock(mpeak_synthetic, mstar_synthetic, Lbox):
    """
    """
    gals = Table()
    gals['mpeak'] = mpeak_synthetic
    gals['obs_sm'] = mstar_synthetic
    gals['_obs_sm_orig_um_snap'] = mstar_synthetic

    ngals = len(gals)
    gals['x'] = np.random.uniform(0, Lbox, ngals)
    gals['y'] = np.random.uniform(0, Lbox, ngals)
    gals['z'] = np.random.uniform(0, Lbox, ngals)
    gals['target_halo_x'] = gals['x']
    gals['target_halo_y'] = gals['y']
    gals['target_halo_z'] = gals['z']

    gals['vx'] = np.random.uniform(-200, 200, ngals)
    gals['vy'] = np.random.uniform(-200, 200, ngals)
    gals['vz'] = np.random.uniform(-200, 200, ngals)
    gals['target_halo_vx'] = gals['vx']
    gals['target_halo_vy'] = gals['vy']
    gals['target_halo_vz'] = gals['vz']

    gals['target_halo_mass'] = gals['mpeak']
    gals['host_halo_mvir'] = gals['mpeak']

    gals['upid'] = -1

    gals['host_centric_x'] = 0.
    gals['host_centric_y'] = 0.
    gals['host_centric_z'] = 0.
    gals['host_centric_vx'] = 0.
    gals['host_centric_vy'] = 0.
    gals['host_centric_vz'] = 0.

    gals['sfr_percentile'] = np.random.uniform(0, 1, ngals)
    ssfr = 10**norm.isf(1 - gals['sfr_percentile'], loc=-10, scale=0.5)
    gals['obs_sfr'] = ssfr*gals['obs_sm']

    gals['halo_id'] = -1
    gals['lightcone_id'] = -1

    gals['restframe_extincted_sdss_abs_magr'] = 0.
    gals['restframe_extincted_sdss_gr'] = 0.
    gals['restframe_extincted_sdss_ri'] = 0.
    gals['is_on_red_sequence_gr'] = False
    gals['is_on_red_sequence_ri'] = False

    return gals

