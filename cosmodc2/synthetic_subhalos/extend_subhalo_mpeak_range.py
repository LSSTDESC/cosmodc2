"""
"""
import numpy as np
import healpy as hp
from halotools.utils import unsorting_indices
from astropy.table import Table
from scipy.stats import norm
from halotools.empirical_models import NFWPhaseSpace
from astropy.cosmology import FlatLambdaCDM
import warnings
warnings.filterwarnings("error")

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
    new_mstar[len(obs_sm_orig):] = new_mstar_lowmass[len(obs_sm_orig):]
    new_mstar[reassign_mstar_mask] = new_mstar_lowmass[reassign_mstar_mask]

    new_mstar_real = new_mstar[:len(obs_sm_orig)]
    new_mstar_synthetic = new_mstar[len(obs_sm_orig):]
    return new_mstar_real, new_mstar_synthetic


def create_synthetic_lowmass_mock_with_satellites(
        mock, healpix_mock, synthetic_dict):
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
    gals['target_halo_id'] = healpix_mock['target_halo_id'][selected_host_indices]
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

    gals['lightcone_id'] = -10
    gals['halo_id'] = -10

    return gals


def get_comoving_distances(zmin, zmax, cosmology, H0=71.0):
    rmin = (cosmology.comoving_distance(zmin)*H0/100.).value   #Mpc/h
    rmax = (cosmology.comoving_distance(zmax)*H0/100.).value

    return rmin, rmax


def get_box_boundaries(Nside, cutout_id, rmin, rmax):

    corners = hp.boundaries(Nside, cutout_id, nest=False)
    center = np.asarray(hp.pixelfunc.pix2vec(Nside, cutout_id, nest=False)) # acccount for spherical cap
    box_corners = np.vstack((corners.T*rmin, corners.T*rmax, center*rmax))
    box_mins = np.asarray([np.min(box_corners[:, i]) for i in range(len(box_corners.T))])
    box_maxs = np.asarray([np.max(box_corners[:, i]) for i in range(len(box_corners.T))])

    return box_mins, box_maxs

box_zero = 1.e-6  #minimum value of position coordinate in box

def get_volume_factor(box_mins, box_maxs, Nside, cutout_id, r_min, r_max, 
                      volume_minx=box_zero, volume_miny=box_zero, volume_maxz=-box_zero, Nsample=100000):
    volume_factor = 1.0
    volume_box = (box_maxs[0] - box_mins[0])*(box_maxs[1] - box_mins[1])*(box_maxs[2] - box_mins[2])
    
    #check boundaries against edges of octant                                          
    x_min = max(box_mins[0], volume_minx)
    y_min = max(box_mins[1], volume_miny)
    z_max = min(box_maxs[2], volume_maxz)
    volume_in_octant = (box_maxs[0] - x_min)*(box_maxs[1] - y_min)*(z_max - box_mins[2])
    vol_frac = volume_in_octant/volume_box
    if vol_frac < 1.:  # edge pixel needs adjustment
        if vol_frac > 1./Nsample:  #check if overlap with octant is big enough for estimate of reduction factor 
            # Monte Carlo the area to find the reduced number of synthetics needed
            gals_x, gals_y , gals_z = generate_trial_sample(box_mins, box_maxs, Nsample=Nsample)
            healpix_mask = mask_galaxies_outside_healpix(gals_x, gals_y, gals_z, cutout_id, Nside, r_min, r_max)
            N_inhpx = np.count_nonzero(healpix_mask) 
            octant_mask = (gals_x >= volume_minx) & (gals_y >= volume_miny) & (gals_z <= volume_maxz)
            mask = healpix_mask & octant_mask
            N_inoctant = np.count_nonzero(mask)
            print('...edge-healpix measure: {} in octant out of {} in healpix'.format(N_inoctant, N_inhpx))
            volume_factor = float(N_inoctant)/float(N_inhpx)
            print('...adjusting xyz box boundaries for octant edges: {:.3g}, {:.3g}, {:.3g}'.format(x_min,
                                                                                                    y_min,
                                                                                                    z_max))
        else:
            volume_factor = 0.0
            print('...fraction of box volume in octant ({:.3g}) too small for Monte Carlo measure'. format(vol_frac))

    #adjust box boundaries 
    box_mins[0] = x_min
    box_mins[1] = y_min
    box_maxs[2] = z_max

    return volume_factor, box_mins, box_maxs
            
def generate_trial_sample(box_mins, box_maxs, Nsample=100000):
        
    gals_x = np.random.uniform(box_mins[0], box_maxs[0], Nsample)
    gals_y = np.random.uniform(box_mins[1], box_maxs[1], Nsample)
    gals_z = np.random.uniform(box_mins[2], box_maxs[2], Nsample)
    
    return gals_x, gals_y , gals_z

z_zero = 1.e-10  #minimum redshift for interpolation grid

def get_redshifts_from_comoving_distances(comoving_distances, zmin, zmax,
                                          cosmology, H0=71.0, zgrid_min=z_zero):
    """
    """
    zgrid = np.logspace(np.log10(max(zmin, zgrid_min)), np.log10(zmax), 50)  # enforce lower limit if zmin=0
    CDgrid = cosmology.comoving_distance(zgrid)*H0/100.
    redshifts = np.interp(comoving_distances, CDgrid, zgrid)
    # check redshifts for z=0 shell
    if zmin < zgrid_min:
        assert all(np.isfinite(redshifts)), "Error: bad redshift values"

    return redshifts


def mask_galaxies_outside_healpix(gals_x, gals_y, gals_z, cutout_id, Nside, r_min, r_max):
    """                                                                                                                                  """
    healpixels = hp.pixelfunc.vec2pix(Nside, gals_x, gals_y, gals_z, nest=False)
    healpix_number_mask = (healpixels == cutout_id)
    #print('.....removing {} fakes falling outside healpixel'.format(np.sum(~healpix_number_mask)))
    r_gals = np.sqrt(gals_x**2 + gals_y**2 + gals_z**2)
    r_mask = (r_gals >= r_min) & (r_gals <= r_max)
    #print('.....removing {} fakes falling outside comoving distance bounds {:.2f}-{:.2f}'.format(np.sum(~r_mask), r_min, r_max))
    healpix_mask = healpix_number_mask & r_mask

    return healpix_mask


def create_synthetic_lowmass_mock_with_centrals(
            mock, healpix_mock, synthetic_dict,
            snapshot_redshift_min, snapshot_redshift_max, cosmology,
            cutout_id=None, Nside=32, H0=71., Ntrial_min=3000,
            volume_minx=box_zero, volume_miny=box_zero, volume_maxz=-box_zero,
            halo_id_offset=0, halo_unique_id=0):
    """ Function generates a data table storing synthetic ultra-faint galaxies
    for purposes of extending the resolution limit of the simulation.
    The generated ultra-faint population will be made up exclusively of central galaxies.

    Parameters
    ----------
    mock : Astropy Table
        Table storing the UniverseMachine galaxy population that we will sample from

    healpix_mock : Astropy Table
        Table storing the galaxies in the healpixel of the Outer Rim simulation being populated

    synthetic_dict : dict
        Dictionary of additional properties being modeled onto synthetic galaxies

    snapshot_redshift_min: float
        Minimum redshift of shell (= redshift or previous snapshot)

    snapshot_redshift_max: float
        Maximum redshift of shell

    Returns
    -------
    ultra_faints : Astropy Table
        Table storing the synthetic population of ultra-faint galaxies

    """
    import healpy as hp
    if cutout_id is None:
        print('...missing cutout_id')
        return Table()

    # setup r_min and r_max for lightcone shell
    r_min = (cosmology.comoving_distance(snapshot_redshift_min)*H0/100.).value
    r_max = (cosmology.comoving_distance(snapshot_redshift_max)*H0/100.).value
    print('...min/max comoving distances for z ({:.4f}-{:.4f}) = ({:.5g}-{:.5g})'.format(snapshot_redshift_min,
                                                                                         snapshot_redshift_max,
                                                                                         r_min, r_max))

    # find coordinates of box enclosing healpixel
    box_mins, box_maxs = get_box_boundaries(Nside, cutout_id, r_min, r_max)
    # adjust for edge cases at boundaries of octant
    volume_factor, box_mins, box_maxs = get_volume_factor(box_mins, box_maxs, Nside, cutout_id, r_min, r_max,
                                                          volume_minx=volume_minx, volume_miny=volume_miny,
                                                          volume_maxz=volume_maxz)
                                   
    nsynthetic = len(synthetic_dict['mpeak'])
    mstar_max = min(10**8., 10.**(np.log10(np.max(synthetic_dict['mpeak']))+1))
    mock_sample_mask = mock['obs_sm'] < mstar_max   #selection mask for low mass UM galaxies
    num_sample = np.count_nonzero(mock_sample_mask)
    if volume_factor < 1.0:
        ngals = int(nsynthetic*volume_factor) #reduce nsynthetic by volume factor
        print('...down-sampling synthetics by {:.3f} to {} for edge pixel {}'.format(volume_factor,
                                                                                     ngals,
                                                                                     cutout_id))
        downsampled_indices =  np.random.randint(0, nsynthetic, ngals)
    else:
        ngals = nsynthetic
        downsampled_indices = np.arange(0, ngals)

    gals = Table()
    if ngals == 0:
        return gals

    selection_indices = np.random.randint(0, num_sample, ngals) # randint(low, high=None, size=None); size=output shape

    #  populate gals table with selected galaxies
    for key in mock.keys():
        if key in list(synthetic_dict.keys()):
            gals[key] = synthetic_dict[key][downsampled_indices]
        else:
            gals[key] = mock[key][mock_sample_mask][selection_indices]  #select properties of low-mass galaxies 
    ngals = len(gals)   #now reset total number to length of table

    #  check that all halos are inside healpixel
    halo_healpixels = hp.pixelfunc.vec2pix(Nside, healpix_mock['target_halo_x'],
        healpix_mock['target_halo_y'], healpix_mock['target_halo_z'], nest=False)
    halo_healpix_mask = (halo_healpixels == cutout_id)
    if np.sum(~halo_healpix_mask) > 0:
        print('...Warning: {} halo(s) detected outside healpixel'.format(np.sum(~halo_healpix_mask)))
        healpix_mock = healpix_mock[halo_healpix_mask]

    #  loop over galaxy-position generator until required number are created
    total_num_created = 0
    nloop = 0
    print('...looping over position generation for {} synthetic centrals'.format(ngals))

    while total_num_created < ngals:
        start_index = total_num_created
        num_needed = int(max(5*(ngals - total_num_created), Ntrial_min)) #  boost by factor of 5 to reduce number of loops
        nloop = nloop + 1
        #  select positions inside box and remove any locations outside the healpixel
        gals_x, gals_y , gals_z = generate_trial_sample(box_mins, box_maxs, Nsample=num_needed)
        healpix_mask = mask_galaxies_outside_healpix(gals_x, gals_y, gals_z, cutout_id, Nside, r_min, r_max)
        num_created = np.sum(healpix_mask)
        total_num_created = min(total_num_created + num_created, ngals)
        print('.....created {} synthetic centrals in loop #{}; {} remaining'.format(num_created, nloop, ngals - total_num_created))
        gals['x'][start_index:total_num_created] = gals_x[healpix_mask][0:total_num_created - start_index]
        gals['y'][start_index:total_num_created] = gals_y[healpix_mask][0:total_num_created - start_index]
        gals['z'][start_index:total_num_created] = gals_z[healpix_mask][0:total_num_created - start_index]

    #  compute redshifts from comoving distance
    r_gals = np.sqrt(gals['x']**2 + gals['y']**2 + gals['z']**2)
    redshifts = get_redshifts_from_comoving_distances(r_gals, snapshot_redshift_min,
                                                      snapshot_redshift_max, cosmology, H0=H0)
    print('...Min and max synthetic redshifts = {:.2f} -> {:.2f}'.format(np.min(redshifts), np.max(redshifts)))

    #  overwrite redshifts with new redshifts
    gals['target_halo_redshift'] = redshifts
    gals['_obs_sm_orig_um_snap'] = gals['obs_sm']

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

    gals['target_halo_id'] = -(np.arange(ngals)*halo_id_offset + halo_unique_id).astype(int)
    print('...Max and min synthetic target halo_id = {} -> {}'.format(np.min(gals['target_halo_id']), np.max(gals['target_halo_id'])))

    #  add other keys
    gals['lightcone_id'] = -20
    gals['halo_id'] = -20
    
    return gals
