"""
"""
import numpy as np
from halotools.utils import unsorting_indices
from astropy.table import Table
from scipy.stats import norm
from halotools.empirical_models import NFWPhaseSpace
from astropy.cosmology import FlatLambdaCDM


default_mpeak_mstar_fit_low_mpeak, default_mpeak_mstar_fit_high_mpeak = 11, 11.5
default_desired_logm_completeness = 9.75

__all__ = ('model_extended_mpeak', 'map_mstar_onto_lowmass_extension',
           'create_synthetic_lowmass_mock_with_satellites', 'create_synthetic_lowmass_mock_with_centrals')


def model_extended_mpeak(mpeak, num_galsampled_gals, desired_logm_completeness=default_desired_logm_completeness,
            logm_min_fit=11.75, logm_max_fit=12.25, Lbox=256.):
    """ Given an input set of subhalo mpeak values, and a desired completeness limit,
    fit the input distribution with a power law at the low mass end,
    extrapolate subhalo abundance to lower masses, and return a set of subhalos
    whose abundance obeys the best-fit power law down to the desired completeness limit.

    Parameters
    ----------
    mpeak : ndarray
        Numpy array of shape (nsubs_orig, )

    num_galsampled_gals : int
        Number of galaxies in the Mpeak array that were selected by GalSampler

    desired_logm_completeness : float, optional
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
    frac_galsampled = num_galsampled_gals/float(len(mpeak))
    new_ngals_max = int((10**lognd_extension_max)*Vbox)
    num_synthetic = int(frac_galsampled*new_ngals_max)
    _nd_new = np.sort(np.random.choice(
        np.arange(1 + npts_total, new_ngals_max), num_synthetic, replace=False))[::-1]
    logndarr_extension = np.log10(_nd_new/Vbox)
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


def create_synthetic_lowmass_mock_with_satellites(
        mock, healpix_mock, synthetic_dict,
        halo_id_offset=0, halo_unique_id=0):
    """
    """
    mstar_max = min(10**8., 10.**(np.log10(np.max(synthetic_dict['mpeak']))+1))
    mock_sample_mask = mock['obs_sm'] < mstar_max
    num_sample = np.count_nonzero(mock_sample_mask)
    selection_indices = np.random.randint(0, num_sample, len(synthetic_dict['mpeak']))

    gals = Table()
    #  populate gals table with selected galaxies
    for key in mock.keys():
        if key in list(synthetic_dict.keys()):
            gals[key] = synthetic_dict[key]
        else:
            gals[key] = mock[key][mock_sample_mask][selection_indices]
    ngals = len(gals)

    gals['_obs_sm_orig_um_snap'] = gals['obs_sm']

    host_mask = healpix_mock['upid'] == -1
    host_indices = np.arange(len(healpix_mock))[host_mask]
    selected_host_indices = np.random.choice(host_indices, ngals, replace=True)
    gals['target_halo_x'] = healpix_mock['target_halo_x'][selected_host_indices]
    gals['target_halo_y'] = healpix_mock['target_halo_y'][selected_host_indices]
    gals['target_halo_z'] = healpix_mock['target_halo_z'][selected_host_indices]
    gals['target_halo_mass'] = healpix_mock['target_halo_mass'][selected_host_indices]
    gals['upid'] = healpix_mock['target_halo_id'][selected_host_indices]
    gals['target_halo_redshift'] = healpix_mock['target_halo_redshift'][selected_host_indices]
    gals['target_halo_vx'] = healpix_mock['target_halo_vx'][selected_host_indices]
    gals['target_halo_vy'] = healpix_mock['target_halo_vy'][selected_host_indices]
    gals['target_halo_vz'] = healpix_mock['target_halo_vz'][selected_host_indices]

    gals['host_halo_mvir'] = gals['target_halo_mass']

    nfw = NFWPhaseSpace()
    nfw_sats = nfw.mc_generate_nfw_phase_space_points(
        mass=gals['target_halo_mass'])
    gals['host_centric_x'] = nfw_sats['x']
    gals['host_centric_y'] = nfw_sats['y']
    gals['host_centric_z'] = nfw_sats['z']
    gals['x'] = gals['host_centric_x'] + gals['target_halo_x']
    gals['y'] = gals['host_centric_y'] + gals['target_halo_y']
    gals['z'] = gals['host_centric_z'] + gals['target_halo_z']

    gals['vx'] = np.random.uniform(-100, 100, ngals) + gals['target_halo_vx']
    gals['vy'] = np.random.uniform(-100, 100, ngals) + gals['target_halo_vy']
    gals['vz'] = np.random.uniform(-100, 100, ngals) + gals['target_halo_vz']

    gals['sfr_percentile'] = np.random.uniform(0, 1, ngals)
    ssfr = 10**norm.isf(1 - gals['sfr_percentile'], loc=-10, scale=0.5)
    gals['obs_sfr'] = ssfr*gals['obs_sm']

    gals['halo_id'] = -(np.arange(ngals)*halo_id_offset + halo_unique_id).astype(int)
    
    print('...Max and min synthetic halo_id = {} -> {}'.format(np.min(gals['halo_id']), np.max(gals['halo_id'])))
    gals['lightcone_id'] = -1

    return gals


def get_redshifts_from_comoving_distances(comoving_distances, zmin, zmax, H0=71.0, OmegaM=0.2648):
    """
    """
    zgrid = np.logspace(np.log10(zmin), np.log10(zmax), 50)
    cosmology = FlatLambdaCDM(H0=H0, Om0=OmegaM)
    CDgrid = cosmology.comoving_distance(zgrid)*H0/100.
    redshifts = np.interp(comoving_distances, CDgrid, zgrid)

    return redshifts


def create_synthetic_lowmass_mock_with_centrals(mock, healpix_mock, synthetic_dict,
                                                cutout_id=None, Nside=8, H0=71.0, OmegaM=0.2648,
                                                halo_id_offset=0, halo_unique_id=0):
    """
    """
    import healpy as hp
    if cutout_id is None:
        print('...missing cutout_id')
        return Table()

    ngals = len(synthetic_dict['mpeak'])
    mstar_max = min(10**8., 10.**(np.log10(np.max(synthetic_dict['mpeak']))+1))
    mock_sample_mask = mock['obs_sm'] < mstar_max
    num_sample = np.count_nonzero(mock_sample_mask)
    selection_indices = np.random.randint(0, num_sample, ngals)

    #  select positions inside box defined by halos in healpix mock and remove any locations outside the healpixel
    halo_healpixels = hp.pixelfunc.vec2pix(Nside, healpix_mock['target_halo_x'],
        healpix_mock['target_halo_y'], healpix_mock['target_halo_z'], nest=False)
    halo_healpix_mask = (halo_healpixels == cutout_id)
    if np.sum(~halo_healpix_mask) > 0:
        print('{} halo(s) detected outide healpixel'.format(np.sum(~halo_healpix_mask)))
    gals_x = np.random.uniform(np.min(healpix_mock['target_halo_x']), np.max(healpix_mock['target_halo_x']), ngals)
    gals_y = np.random.uniform(np.min(healpix_mock['target_halo_y']), np.max(healpix_mock['target_halo_y']), ngals)
    gals_z = np.random.uniform(np.min(healpix_mock['target_halo_z']), np.max(healpix_mock['target_halo_z']), ngals)
    healpixels = hp.pixelfunc.vec2pix(Nside, gals_x, gals_y, gals_z, nest=False)
    healpix_number_mask = (healpixels == cutout_id)
    print('...removing {} fakes falling outside healpixel'.format(np.sum(~healpix_number_mask)))
    r_halo = np.sqrt(healpix_mock['target_halo_x']**2 + healpix_mock['target_halo_y']**2 + healpix_mock['target_halo_z']**2)
    r_gals = np.sqrt(gals_x**2 + gals_y**2 + gals_z**2)
    r_mask = (r_gals >= np.min(r_halo)) & (r_gals <= np.max(r_halo))
    print('...removing {} fakes falling outside comoving distance bounds {:.2f}-{:.2f}'
          .format(np.sum(~r_mask),np.min(r_halo), np.max(r_halo)))
    healpix_mask = healpix_number_mask & r_mask
    if np.sum(healpix_mask) == 0:
        return Table()

    #  compute redshifts from comoving distance
    redshifts = get_redshifts_from_comoving_distances(r_gals[healpix_mask], 
                    np.min(healpix_mock['target_halo_redshift'][healpix_mask]),
                    np.max(healpix_mock['target_halo_redshift'][healpix_mask]), H0=H0, OmegaM=OmegaM)
    print('...Min and max synthetic redshifts = {} -> {}'.format(np.min(redshifts), np.max(redshifts)))

    gals = Table()
    #  populate gals table with selected galaxies
    for key in mock.keys():
        if key in list(synthetic_dict.keys()):
            gals[key] = synthetic_dict[key]
        else:
            gals[key] = mock[key][mock_sample_mask][selection_indices]
    gals = gals[healpix_mask]

    #  overwrite positions with new random positions from healpix selection
    gals['x'] = gals_x[healpix_mask]
    gals['y'] = gals_y[healpix_mask]
    gals['z'] = gals_z[healpix_mask]

    #  overwrite redshifts with new redshifts
    gals['target_halo_redshift'] = redshifts
    gals['_obs_sm_orig_um_snap'] = gals['obs_sm']

    ngals = len(gals)

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

    gals['halo_id'] = -(np.arange(ngals)*halo_id_offset + halo_unique_id).astype(int)
    print('...Max and min synthetic halo_id = {} -> {}'.format(np.min(gals['halo_id']), np.max(gals['halo_id'])))
    gals['lightcone_id'] = -1

    return gals
