import numpy as np
from scipy.spatial import cKDTree
from astropy.table import Table
from halotools.empirical_models import NFWPhaseSpace
from scipy.stats import powerlaw
from ..sdss_colors.sigmoid_magr_model import magr_monte_carlo
from ..sdss_colors.sigmoid_g_minus_r import red_sequence_peak_gr
from ..sdss_colors.sigmoid_r_minus_i import red_sequence_peak_ri


__all__ = ('nearby_hostmass_selection_indices', 'calculate_synthetic_richness',
           'create_synthetic_cluster_satellites', 'model_synthetic_cluster_satellites')


def nearby_hostmass_selection_indices(hostmass, desired_hostmass):
    """
    Parameters
    ----------
    hostmass : ndarray
        Numpy array of shape (ntot_available_sats, ) storing the host halo mass
        of every satellite in the selection sample

    desired_hostmass : ndarray
        Numpy array of shape (ntot_desired_fakesats, ) storing the host halo mass
        of every desired synthetic satellite

    Returns
    -------
    selection_indices : ndarray
        Numpy integer array of shape (ntot_desired_fakesats, ) storing the indices
        that select the satellites from the original sample
    """
    noisy_hostmass = 10**np.random.normal(loc=np.log10(hostmass), scale=0.02)
    X1 = np.vstack((noisy_hostmass, )).T
    tree = cKDTree(X1)

    X2 = np.vstack((desired_hostmass, )).T
    nn_distinces, nn_indices = tree.query(X2)
    nn_indices = np.minimum(len(hostmass)-1, nn_indices)
    nn_indices = np.maximum(0, nn_indices)

    return nn_indices


def calculate_synthetic_richness(halo_richness, logmhalo, logmhalo_source,
        cluster_satboost_logm_table, cluster_satboost_table, logm_outer_rim_effect=14.75):
    """
    Parameters
    ----------
    halo_richness : ndarray
        Numpy array of shape (nhalos, ) storing the richness of each halo

    logmhalo : ndarray
        Numpy array of shape (nhalos, ) storing the log mass of each halo

    Returns
    -------
    synthetic_richness : ndarray
        Numpy integer array of shape (nhalos, ) storing the synthetic richness
    """
    boost_factor = np.interp(logmhalo, cluster_satboost_logm_table, cluster_satboost_table)
    dlogm = logmhalo - logmhalo_source
    outer_rim_boost_factor = 10.**dlogm
    low, high = logm_outer_rim_effect - 0.25, logm_outer_rim_effect + 0.25
    logm_outer_rim_effect = np.random.uniform(low, high, len(boost_factor))
    highmass_mask = logmhalo > logm_outer_rim_effect
    boost_factor = np.where(highmass_mask, outer_rim_boost_factor*boost_factor, boost_factor)
    return np.array(halo_richness*boost_factor, dtype=int)


def model_synthetic_cluster_satellites(mock, Lbox=256.,
        cluster_satboost_logm_table=[13.5, 13.75, 14],
        cluster_satboost_table=[0., 0.15, 0.2], **kwargs):
    """
    """
    #  Calculate the mass and richness of every target halo
    host_halo_id, idx, counts = np.unique(
        mock['target_halo_id'], return_counts=True, return_index=True)
    host_mass = mock['target_halo_mass'][idx]
    host_redshift = mock['target_halo_redshift'][idx]
    host_x = mock['target_halo_x'][idx]
    host_y = mock['target_halo_y'][idx]
    host_z = mock['target_halo_z'][idx]
    host_vx = mock['target_halo_vx'][idx]
    host_vy = mock['target_halo_vy'][idx]
    host_vz = mock['target_halo_vz'][idx]
    source_halo_mvir = mock['source_halo_mvir'][idx]
    target_halo_id = mock['target_halo_id'][idx]
    host_richness = counts

    #  For each target halo, calculate the number of synthetic satellites we need to add
    synthetic_richness = calculate_synthetic_richness(
        host_richness, np.log10(host_mass), np.log10(source_halo_mvir),
        cluster_satboost_logm_table, cluster_satboost_table)

    if np.sum(synthetic_richness) == 0:
        return Table()
    else:
        #  For every synthetic galaxy, calculate the mass, redshift, position, and velocity of the host halo
        synthetic_hostmass = np.repeat(host_mass, synthetic_richness)
        synthetic_redshift = np.repeat(host_redshift, synthetic_richness)
        synthetic_target_halo_x = np.repeat(host_x, synthetic_richness)
        synthetic_target_halo_y = np.repeat(host_y, synthetic_richness)
        synthetic_target_halo_z = np.repeat(host_z, synthetic_richness)
        synthetic_target_halo_vx = np.repeat(host_vx, synthetic_richness)
        synthetic_target_halo_vy = np.repeat(host_vy, synthetic_richness)
        synthetic_target_halo_vz = np.repeat(host_vz, synthetic_richness)
        synthetic_target_halo_id = np.repeat(target_halo_id, synthetic_richness)

        #  Use Halotools to generate halo-centric positions and velocities according to NFW
        nfw = NFWPhaseSpace()
        nfw_sats = nfw.mc_generate_nfw_phase_space_points(mass=synthetic_hostmass)

        sats = Table()
        sats['host_centric_x'] = nfw_sats['x']
        sats['host_centric_y'] = nfw_sats['y']
        sats['host_centric_z'] = nfw_sats['z']
        sats['host_centric_vx'] = nfw_sats['vx']
        sats['host_centric_vy'] = nfw_sats['vy']
        sats['host_centric_vz'] = nfw_sats['vz']

        #  Add host-centric pos/vel to target halo pos/vel
        sats['x'] = synthetic_target_halo_x + nfw_sats['x']
        sats['y'] = synthetic_target_halo_y + nfw_sats['y']
        sats['z'] = synthetic_target_halo_z + nfw_sats['z']
        sats['vx'] = synthetic_target_halo_vx + nfw_sats['vx']
        sats['vy'] = synthetic_target_halo_vy + nfw_sats['vy']
        sats['vz'] = synthetic_target_halo_vz + nfw_sats['vz']

        if Lbox > 0.:    # enforce periodicity
            sats['x'] = np.mod(sats['x'], Lbox)
            sats['y'] = np.mod(sats['y'], Lbox)
            sats['z'] = np.mod(sats['z'], Lbox)

        sats['host_halo_mvir'] = synthetic_hostmass
        sats['upid'] = synthetic_target_halo_id

        #  Assign synthetic subhalo mass according to a power law
        #  Maximum allowed value of the subhalo mass is the host halo mass
        #  Power law distribution in subhalo mass spans [11, logMhost]
        alpha = np.zeros_like(synthetic_hostmass) + 2.0  #  power-law slope
        sats['mpeak'] = 10**(np.log10(synthetic_hostmass) - 4*powerlaw.rvs(alpha))

        sats['halo_id'] = -1
        sats['lightcone_id'] = -1
        sats['um_target_halo_id'] = -1
        sats['um_target_halo_mass'] = synthetic_hostmass
        sats['target_halo_redshift'] = synthetic_redshift

        #  Rather than modeling the M*-Mpeak relation, for convenience we'll sample it and add noise
        #  Note that we will *NOT* derive restframe flux or color this way, only stellar mass and SFR
        sat_sample_mask = (mock['upid'] != -1) & (mock['is_on_red_sequence_gr'])
        tree = cKDTree(np.vstack((mock['mpeak'][sat_sample_mask], )).T)

        X2 = np.vstack((sats['mpeak'], )).T
        nn_distinces, nn_indices = tree.query(X2)
        nn_indices = np.minimum(len(mock['mpeak'][sat_sample_mask])-1, nn_indices)
        nn_indices = np.maximum(0, nn_indices)
        selected_logsm = np.log10(mock['obs_sm'][sat_sample_mask][nn_indices])
        selected_sfr = mock['obs_sfr'][sat_sample_mask][nn_indices]

        #  Add noise to M* and SFR to prevent repeated values
        #  Probably unnecessary but doesn't hurt
        sats['obs_sm'] = 10**np.random.normal(loc=selected_logsm, scale=0.1)
        sfr_scatter = np.maximum(0.1*selected_sfr, 1.)
        sats['obs_sfr'] = np.maximum(np.random.normal(loc=selected_sfr, scale=sfr_scatter), 0.)

        sats['_obs_sm_orig_um_snap'] = sats['obs_sm']
        sats['sfr_percentile'] = mock['sfr_percentile'][sat_sample_mask][nn_indices]

        #  We will only boost the richness of red sequence cluster members
        sats['is_on_red_sequence_gr'] = True
        sats['is_on_red_sequence_ri'] = True

        #  Model r-band magnitude just like regular galaxies
        sats['restframe_extincted_sdss_abs_magr'] = magr_monte_carlo(
            sats['obs_sm'], sats['upid'], sats['target_halo_redshift'])

        #  Model g-r and r-i color just like regular red sequence galaxies
        red_sequence_loc_gr = red_sequence_peak_gr(
            sats['restframe_extincted_sdss_abs_magr'], sats['target_halo_redshift'])
        red_sequence_loc_ri = red_sequence_peak_ri(
            sats['restframe_extincted_sdss_abs_magr'], sats['target_halo_redshift'])

        sats['restframe_extincted_sdss_gr'] = np.random.normal(loc=red_sequence_loc_gr, scale=0.05)
        sats['restframe_extincted_sdss_ri'] = np.random.normal(loc=red_sequence_loc_ri, scale=0.02)

        #  It is important to insure that `sats` has a column for every column of the `dc2` mock
        #  In case we missed one of the obscure ones, just add it to the table
        #  Note that this is what we previously did for *ALL* columns
        satkeys = list(sats.keys())
        for key in mock.keys():
            if key not in satkeys:
                sats[key] = mock[key][sat_sample_mask][nn_indices]

        return sats


def create_synthetic_cluster_satellites(mock, Lbox=256.,
        cluster_satboost_logm_table=[13.5, 13.75, 14],
        cluster_satboost_table=[0., 0.15, 0.2], **kwargs):
    """
    """
    raise ValueError("This function is deprecated. Use `model_synthetic_cluster_satellites` instead.")
    host_halo_id, idx, counts = np.unique(
        mock['target_halo_id'], return_counts=True, return_index=True)
    host_mass = mock['target_halo_mass'][idx]
    source_halo_mvir = mock['source_halo_mvir'][idx]
    host_richness = counts

    synthetic_richness = calculate_synthetic_richness(
        host_richness, np.log10(host_mass), np.log10(source_halo_mvir),
        cluster_satboost_logm_table, cluster_satboost_table)

    if np.sum(synthetic_richness) > 0:
        synthetic_hostmass = np.repeat(host_mass, synthetic_richness)

        sat_sample_mask = mock['upid'] != -1
        selection_indices = nearby_hostmass_selection_indices(
            mock['target_halo_mass'][sat_sample_mask], synthetic_hostmass)

        if len(selection_indices) > 1:  # ensure more than 1 galaxy is being faked
            sats = Table()
            for key in mock.keys():
                sats[key] = mock[key][sat_sample_mask][selection_indices]

            nfw = NFWPhaseSpace()
            nfw_sats = nfw.mc_generate_nfw_phase_space_points(
                mass=sats['target_halo_mass'])

            sats['host_centric_x'] = nfw_sats['x']
            sats['host_centric_y'] = nfw_sats['y']
            sats['host_centric_z'] = nfw_sats['z']
            sats['host_centric_vx'] = nfw_sats['vx']
            sats['host_centric_vy'] = nfw_sats['vy']
            sats['host_centric_vz'] = nfw_sats['vz']

            sats['x'] = sats['host_centric_x'] + sats['target_halo_x']
            sats['y'] = sats['host_centric_y'] + sats['target_halo_y']
            sats['z'] = sats['host_centric_z'] + sats['target_halo_z']
            sats['vx'] = sats['host_centric_vx'] + sats['target_halo_vx']
            sats['vy'] = sats['host_centric_vy'] + sats['target_halo_vy']
            sats['vz'] = sats['host_centric_vz'] + sats['target_halo_vz']

            if Lbox > 0.:    # enforce periodicity
                sats['x'] = np.mod(sats['x'], Lbox)
                sats['y'] = np.mod(sats['y'], Lbox)
                sats['z'] = np.mod(sats['z'], Lbox)

            sats['halo_id'] = -1
            sats['lightcone_id'] = -1

            return sats
        else:
            return Table()
    else:
        return Table()
