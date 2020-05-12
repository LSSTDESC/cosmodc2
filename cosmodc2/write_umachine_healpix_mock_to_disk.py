""" Module storing the primary driver script used for the v1 release of cosmoDC2.
"""
import os
import psutil
import numpy as np
import h5py
import re
import healpy as hp
from time import time
from astropy.table import Table, vstack
from astropy.cosmology import FlatLambdaCDM, WMAP7
from astropy.utils.misc import NumpyRNGContext
from cosmodc2.sdss_colors import assign_restframe_sdss_gri
from cosmodc2.sdss_colors.sigmoid_magr_model import magr_monte_carlo
from cosmodc2.get_healpix_cutout_info import get_snap_redshift_min

from scipy.spatial import cKDTree
from cosmodc2.stellar_mass_remapping import remap_stellar_mass_in_snapshot
from galsampler import halo_bin_indices, source_halo_index_selection
from galsampler.cython_kernels import galaxy_selection_kernel
from halotools.utils import crossmatch

from cosmodc2.synthetic_subhalos import map_mstar_onto_lowmass_extension
from cosmodc2.synthetic_subhalos import create_synthetic_lowmass_mock_with_centrals
from cosmodc2.synthetic_subhalos import create_synthetic_lowmass_mock_with_satellites
from cosmodc2.synthetic_subhalos import model_synthetic_cluster_satellites
from cosmodc2.synthetic_subhalos import synthetic_logmpeak
from cosmodc2.triaxial_satellite_distributions.axis_ratio_model import monte_carlo_halo_shapes
from halotools.empirical_models import halo_mass_to_halo_radius
from halotools.utils import normalized_vectors
from cosmodc2.triaxial_satellite_distributions.monte_carlo_triaxial_profile import generate_triaxial_satellite_distribution
from cosmodc2.get_fof_halo_shapes import get_halo_shapes
from cosmodc2.get_fof_halo_shapes import get_matched_shapes

fof_halo_mass = 'fof_halo_mass'
# fof halo mass in healpix cutouts
fof_mass = 'fof_mass'
mass = 'mass'
fof_max = 14.5
sod_mass = 'sod_mass'
m_particle_1000 = 1.85e12
H0 = 71.0
OmegaM = 0.2648
OmegaB = 0.0448

Nside = 2048  #  fine pixelization for determining sky area

# halo id offsets
cutout_id_offset_halo = int(1e3)  # offset to generate unique id for cutouts and snapshots
halo_id_offset = int(1e8)  # offset to guarantee unique halo ids across cutout files and snapshots

#galaxy id offsets for non image-sim catalogs (eg. 5000 sq. deg.)
cutout_id_offset = int(1e9)
z_offsets_not_im = {'32':[0, 1e8, 2e8, 3e8]}

# galaxy id offsets for image simulations
cutout_id_offset_galaxy = {'8':1e9, '32': 62500000} #  offset to guarantee unique galaxy ids across cutout files
z_offsets = {'8':[0, 4e7, 2e8, 1e9], '32':[0, 2500000, 12500000, 62500000]} #  offset to guarantee unique galaxy ids across z-ranges
cutout_remap = {'8': {'564':1, '565':2, '566':3, '597':4, '598':5, '628':6, '629':7, '630':8,
                      '533':9, '534':10, '596':11, '599':12, '660':13, '661':14},
                '32': {
 '8276': 0, '8280': 1, '8403': 2, '8404': 3, '8407': 4, '8408': 5, '8531': 6, '8532': 7, '8533': 8, '8535': 9, '8536': 10,
 '8537': 11, '8658': 12, '8659': 13, '8660': 14, '8661': 15, '8662': 16, '8663': 17, '8664': 18, '8665': 19, '8786': 20, '8787': 21,
 '8788': 22, '8789': 23, '8790': 24, '8791': 25, '8792': 26, '8793': 27, '8794': 28, '8913': 29, '8914': 30, '8915': 31, '8916': 32,
 '8917': 33, '8918': 34, '8919': 35, '8920': 36, '8921': 37, '8922': 38, '9041': 39, '9042': 40, '9043': 41, '9044': 42, '9045': 43,
 '9046': 44, '9047': 45, '9048': 46, '9049': 47, '9050': 48, '9051': 49, '9168': 50, '9169': 51, '9170': 52, '9171': 53, '9172': 54,
 '9173': 55, '9174': 56, '9175': 57, '9176': 58, '9177': 59, '9178': 60, '9179': 61, '9296': 62, '9297': 63, '9298': 64, '9299': 65,
 '9300': 66, '9301': 67, '9302': 68, '9303': 69, '9304': 70, '9305': 71, '9306': 72, '9307': 73, '9308': 74, '9423': 75, '9424': 76,
 '9425': 77, '9426': 78, '9427': 79, '9428': 80, '9429': 81, '9430': 82, '9431': 83, '9432': 84, '9433': 85, '9434': 86, '9435': 87,
 '9436': 88, '9551': 89, '9552': 90, '9553': 91, '9554': 92, '9555': 93, '9556': 94, '9557': 95, '9558': 96, '9559': 97, '9560': 98,
 '9561': 99, '9562': 100, '9563': 101, '9564': 102, '9565': 103, '9678': 104, '9679': 105, '9680': 106, '9681': 107, '9682': 108,
 '9683': 109, '9684': 110, '9685': 111, '9686': 112, '9687': 113, '9688': 114, '9689': 115, '9690': 116, '9691': 117, '9692': 118,
 '9693': 119, '9807': 120, '9808': 121, '9809': 122, '9810': 123, '9811': 124, '9812': 125, '9813': 126, '9814': 127, '9815': 128,
 '9816': 129, '9817': 130, '9818': 131, '9819': 132, '9820': 133, '9821': 134, '9935': 135, '9936': 136, '9937': 137, '9938': 138,
 '9939': 139, '9940': 140, '9941': 141, '9942': 142, '9943': 143, '9944': 144, '9945': 145, '9946': 146, '9947': 147, '9948': 148,
 '10064': 149, '10065': 150, '10066': 151, '10067': 152, '10068': 153, '10069': 154, '10070': 155, '10071': 156, '10072': 157,
 '10073': 158, '10074': 159, '10075': 160, '10076': 161, '10192': 162, '10193': 163, '10194': 164, '10195': 165, '10196': 166,
 '10197': 167, '10198': 168, '10199': 169, '10200': 170, '10201': 171, '10202': 172, '10203': 173, '10320': 174, '10321': 175,
 '10322': 176, '10323': 177, '10324': 178, '10325': 179, '10326': 180, '10327': 181, '10328': 182, '10329': 183, '10330': 184,
 '10444': 185, '10445': 186, '10446': 187, '10447': 188, '10448': 189, '10449': 190, '10450': 191, '10451': 192, '10452': 193,
 '10453': 194, '10564': 195, '10565': 196, '10566': 197, '10567': 198, '10568': 199, '10569': 200, '10570': 201, '10571': 202,
 '10572': 203, '10680': 204, '10681': 205, '10682': 206, '10683': 207, '10684': 208, '10685': 209, '10686': 210, '10687': 211,
 '10792': 212, '10793': 213, '10794': 214, '10796': 215, '10797': 216, '10798': 217, '10900': 218, '10901': 219, '10904': 220,
 '10905': 221, '11004': 222, '11008': 223}
                }

# constants to determine synthetic number density
Ntotal_synthetics = 1932058570 # total number of synthetic galaxies in cosmoDC2_image
nhpx_total = float(131)  # number of healpixels in image area
snapshot_min = 121
# specify edges of octant
volume_minx=0.
volume_miny=0.
volume_maxz=0.

def write_umachine_healpix_mock_to_disk(
            umachine_mstar_ssfr_mock_fname_list, umachine_host_halo_fname_list,
            healpix_data, snapshots, output_color_mock_fname,  shape_dir,
            redshift_list, commit_hash, synthetic_halo_minimum_mass=9.8, num_synthetic_gal_ratio=1.,
            use_centrals=True, use_substeps_real=True, use_substeps_synthetic=False, image=False,
            randomize_redshift_real=True, randomize_redshift_synthetic=True, Lbox=3000.,
            gaussian_smearing_real_redshifts=0., nzdivs=6, Nside_cosmoDC2=32, mstar_min=7e6, z2ts={},
            mass_match_noise=0.1):
    """
    Main driver function used to paint SDSS fluxes onto UniverseMachine,
    GalSample the mock into the lightcone healpix cutout, and write the healpix mock to disk.

    Parameters
    ----------
    umachine_mstar_ssfr_mock_fname_list : list
        List of length num_snaps storing the absolute path to the
        value-added UniverseMachine snapshot mock

    umachine_host_halo_fname_list : list
        List of length num_snaps storing the absolute path to the
        value-added host halo catalog hosting the UniverseMachine snapshot mock

    healpix_data : <HDF5 file>
        Pointer to open hdf5 file for the lightcone healpix cutout
        source halos into which UniverseMachine will be GalSampled

    snapshots : list
        List of snapshots in lightcone healpix cutout

    output_color_mock_fname : string
        Absolute path to the output healpix mock

    shape_dir: string
        Directory storing files with halo-shape information

    redshift_list : list
        List of length num_snaps storing the value of the redshifts
        in the target halo lightcone cutout

    commit_hash : string
        Commit hash of the version of the cosmodc2 repo used when
        calling this function.

        After updating the cosmodc2 repo to the desired version,
        the commit_hash can be determined by navigating to the root
        directory and typing ``git log --pretty=format:'%h' -n 1``

    synthetic_halo_minimum_mass: float
        Minimum value of log_10 of synthetic halo mass

    num_synthetic_gal_ratio: float
        Ratio to control number of synthetic galaxies generated

    use_centrals: boolean
        Flag controlling if ultra faint galaxies are added as centrals or satellites

    use_substeps_real: boolean
        Flag controlling use of color substepping for real galaxies

    use_substeps_synthetic: boolean
        Flag controlling use of color substepping for synthetic galaxies

    image: boolean
        Flag specifying if catalog will be used for image simulations (affects ids)

    mstar_min: stellar mass cut for synthetic galaxies (not used in image simulations)

    mass_match_noise: noise added to log of source halo masses to randomize the match to target halos
    """

    output_mock = {}
    gen = zip(
            umachine_mstar_ssfr_mock_fname_list,
            umachine_host_halo_fname_list, redshift_list, snapshots)
    start_time = time()
    process = psutil.Process(os.getpid())

    #  determine number of healpix cutout to use as offset for galaxy ids
    output_mock_basename = os.path.basename(output_color_mock_fname)
    file_ids = [int(d) for d in re.findall(r'\d+', os.path.splitext(output_mock_basename)[0])]

    cutout_number_true = file_ids[-1]
    z_range_id = file_ids[-3]  #  3rd-last digits in filename

    if image:
        cutout_number = cutout_remap[str(Nside_cosmoDC2)].get(str(cutout_number_true),
                                                              cutout_number_true) #  translate for imsims
        galaxy_id_offset = int(cutout_number*cutout_id_offset_galaxy[str(Nside_cosmoDC2)] +\
                               z_offsets[str(Nside_cosmoDC2)][z_range_id])
        halo_id_cutout_offset = int(cutout_number*cutout_id_offset_halo)
    else:
        cutout_number = cutout_number_true # used for output
        galaxy_id_offset = int(cutout_number_true*cutout_id_offset + \
                               z_offsets_not_im[str(Nside_cosmoDC2)][z_range_id])
        halo_id_cutout_offset = int(cutout_number_true*cutout_id_offset_halo)

    #  determine seed from output filename
    seed = get_random_seed(output_mock_basename)

    #  determine maximum redshift and volume covered by catalog
    redshift_max = [float(k) for k,v in z2ts.items() if int(v)==snapshot_min][0]
    cosmology = FlatLambdaCDM(H0=H0, Om0=OmegaM)
    Vtotal = cosmology.comoving_volume(redshift_max).value    

    #  determine total number of synthetic galaxies for arbitrary healpixel for full z range
    synthetic_number = int(Ntotal_synthetics/nhpx_total)
    #  number for healpixels straddling the edge of the octant will be adjusted later
    # initialize previous redshift for computing synthetic galaxy distributions
    previous_redshift = get_snap_redshift_min(z2ts, snapshots)

    #  initialize book-keeping variables
    fof_halo_mass_max = 0.
    Ngals_total = 0

    print('\nStarting snapshot processing')
    print('Using initial seed = {}'.format(seed))
    print('Using nside = {}'.format(Nside_cosmoDC2))
    print('Maximum redshift for catalog = {}'.format(redshift_max))
    print('Minimum redshift for catalog = {}'.format(previous_redshift))
    print('Synthetic-halo minimum mass =  {}'.format(synthetic_halo_minimum_mass))
    print('Number of synthetic ultra-faint galaxies = {}'.format(synthetic_number))
    print('Using {} synthetic low-mass galaxies'.format('central' if use_centrals else 'satellite'))
    galaxy_types = ['real', 'synthetic']
    for flag, t in zip([randomize_redshift_real, randomize_redshift_synthetic], galaxy_types):
        print('Using {} redshifts for {} galaxies'.format('randomized' if flag else 'assigned', t))
    for flag, t in zip([use_substeps_real, use_substeps_synthetic], galaxy_types):
        print('{} for {} galaxies'.format('Using color sub-stepping' if flag else 'NO color sub-stepping', t))
    print('Using halo-id offset = {}'.format(halo_id_offset))
    print('Using galaxy-id offset = {} for cutout number {}'.format(galaxy_id_offset, cutout_number_true))
    for a, b, c, d in gen:
        umachine_mock_fname = a
        umachine_halos_fname = b
        redshift = c
        snapshot = d
        halo_unique_id = int(halo_id_cutout_offset + int(snapshot))
        print('Using halo_unique id = {} for snapshot {}'.format(halo_unique_id, snapshot))

        new_time_stamp = time()

        #  seed should be changed for each new shell
        seed = seed + 2

        #  check for halos in healpixel
        if len(healpix_data[snapshot]['id']) == 0:
            output_mock[snapshot] = {}
            print("\n...skipping empty snapshot {}".format(snapshot))
            continue

        #  Get galaxy properties from UM catalogs and target halo properties
        print("\n...loading z = {0:.2f} galaxy catalog into memory".format(redshift))
        mock = Table.read(umachine_mock_fname, path='data')

        ###  GalSampler
        print("\n...loading z = {0:.2f} halo catalogs into memory".format(redshift))
        source_halos = Table.read(umachine_halos_fname, path='data')
        #
        target_halos = get_astropy_table(healpix_data[snapshot], halo_unique_id=halo_unique_id)
        fof_halo_mass_max = max(np.max(target_halos[fof_halo_mass].quantity.value), fof_halo_mass_max)

        b_to_a, c_to_a, e, p = monte_carlo_halo_shapes(np.log10(target_halos[fof_halo_mass]))
        target_halos['halo_ellipticity'] = e
        target_halos['halo_prolaticity'] = p
        spherical_halo_radius = halo_mass_to_halo_radius(
            target_halos[fof_halo_mass], WMAP7, redshift, 'vir')
        target_halos['axis_A_length'] = 1.5*spherical_halo_radius  #  crude fix for B and C shrinking
        target_halos['axis_B_length'] = b_to_a*target_halos['axis_A_length']
        target_halos['axis_C_length'] = c_to_a*target_halos['axis_A_length']

        nvectors = len(target_halos)
        rng = np.random.RandomState(seed)
        random_vectors = rng.uniform(-1, 1, nvectors*3).reshape((nvectors, 3))
        axis_A = normalized_vectors(random_vectors)*target_halos['axis_A_length'].reshape((-1, 1))
        target_halos['axis_A_x'] = axis_A[:, 0]
        target_halos['axis_A_y'] = axis_A[:, 1]
        target_halos['axis_A_z'] = axis_A[:, 2]
        # now add halo shape information for those halos with matches in shape files 
        shapes = get_halo_shapes(snapshot, target_halos['fof_halo_id'],  target_halos['rep'],
                                 shape_dir)
        if shapes:
            target_halos = get_matched_shapes(shapes, target_halos)

        print("...Finding halo--halo correspondence with GalSampler")
        #  Bin the halos in each simulation by mass
        dlogM = 0.15
        mass_bins = 10.**np.arange(10.5, 14.5+dlogM, dlogM)
        source_halos['mass_bin'] = halo_bin_indices(
            mass=(source_halos['mvir'], mass_bins))
        target_halos['mass_bin'] = halo_bin_indices(
            mass=(target_halos[fof_halo_mass], mass_bins))

        #  For every target halo, find a source halo with closely matching mass 
        #  Add noise to randomize the selections around the closest match
        log_src_mass = np.log10(source_halos['mvir'])
        noisy_log_src_mass = np.random.normal(loc=log_src_mass, scale=mass_match_noise)
        X = np.vstack((noisy_log_src_mass, )).T
        source_halo_tree = cKDTree(X)
        Y = np.vstack((np.log10(target_halos[fof_halo_mass]), )).T
        # original code crashes - don't need 2 KDTrees
        #target_halo_tree = cKDTree(Y)
        #source_halo_dlogm, source_halo_indx = target_halo_tree.query(source_halo_tree)
        #  Find indices of source halos masses that match target halo masses 
        source_halo_dlogm, source_halo_indx = source_halo_tree.query(Y)

        #  Transfer quantities from the source halos to the corresponding target halo
        target_halos['source_halo_id'] = source_halos['halo_id'][source_halo_indx]
        target_halos['matching_mvir'] = source_halos['mvir'][source_halo_indx]
        target_halos['richness'] = source_halos['richness'][source_halo_indx]
        target_halos['first_galaxy_index'] = source_halos['first_galaxy_index'][source_halo_indx]

        ################################################################################
        #  Use GalSampler to calculate the indices of the galaxies that will be selected
        ################################################################################
        print("...GalSampling z={0:.2f} galaxies to OuterRim halos".format(redshift))

        source_galaxy_indx = np.array(galaxy_selection_kernel(
            target_halos['first_galaxy_index'].astype('i8'),
            target_halos['richness'].astype('i4'), target_halos['richness'].sum()))

        ########################################################################
        #  Correct stellar mass for low-mass subhalos and create synthetic mpeak
        ########################################################################
        print("...correcting low mass mpeak and assigning synthetic mpeak values")
        #  First generate the appropriate number of synthetic galaxies for the snapshot
        mpeak_synthetic_snapshot = 10**synthetic_logmpeak(
            mock['mpeak'], seed=seed, desired_logm_completeness=synthetic_halo_minimum_mass)
        print('...assembling {} synthetic galaxies'.format(len(mpeak_synthetic_snapshot)))

        ########################################################################
        #  Assign stellar mass
        ########################################################################
        print("...re-assigning high-mass mstar values")

        #  Map stellar mass onto mock using target halo mass instead of UM Mpeak for cluster BCGs
        new_mstar = remap_stellar_mass_in_snapshot(redshift, mock['mpeak'], mock['obs_sm'])
        mock.rename_column('obs_sm', '_obs_sm_orig_um_snap')
        mock['obs_sm'] = new_mstar

        #  Add call to map_mstar_onto_lowmass_extension function after pre-determining low-mass slope
        print("...re-assigning low-mass mstar values")
        new_mstar_real, mstar_synthetic_snapshot = map_mstar_onto_lowmass_extension(
            mock['mpeak'], mock['obs_sm'], mpeak_synthetic_snapshot)
        mock['obs_sm'] = new_mstar_real
        mstar_mask = np.isclose(mstar_synthetic_snapshot, 0.)
        if np.sum(mstar_mask ) > 0:
            print('...Warning: Number of synthetics with zero mstar = {}'.format(np.sum(mstar_mask )))

        #  Assign colors to synthetic low-mass galaxies
        synthetic_upid = np.zeros_like(mpeak_synthetic_snapshot).astype(int) - 1
        if randomize_redshift_synthetic:
            with NumpyRNGContext(seed):
                synthetic_redshift = np.random.normal(loc=redshift, size=len(synthetic_upid), scale=0.05)
                synthetic_redshift = np.where(synthetic_redshift < 0, 0, synthetic_redshift)
        else:
            synthetic_redshift = np.zeros(len(synthetic_upid)) + redshift

        with NumpyRNGContext(seed):
            synthetic_sfr_percentile = np.random.uniform(0, 1, len(synthetic_upid))
        print("...Calling assign_restframe_sdss_gri "
            "with synthetic_upid array having {0} elements".format(len(synthetic_upid)))
        _result = assign_restframe_sdss_gri(
            synthetic_upid, mstar_synthetic_snapshot, synthetic_sfr_percentile,
            mpeak_synthetic_snapshot, synthetic_redshift, seed=seed, use_substeps=use_substeps_synthetic,
            nzdivs=nzdivs)
        (magr_synthetic_snapshot, gr_synthetic_snapshot, ri_synthetic_snapshot,
            is_red_gr_synthetic_snapshot, is_red_ri_synthetic_snapshot) = _result

        #  check for bad values
        for m_id, m in zip(['magr', 'gr', 'ri'], [magr_synthetic_snapshot, gr_synthetic_snapshot, ri_synthetic_snapshot]):
            num_infinite = np.sum(~np.isfinite(m))
            if num_infinite > 0:
                print('...Warning: {} infinite values in synthetic {}'.format(num_infinite, m_id))

        #  Now downsample the synthetic galaxies to adjust for volume of lightcone shell
        #  desired number = synthetic_number*comoving_vol(snapshot)/comoving_vol(healpixel)
        volume_factor = get_volume_factor(redshift, previous_redshift, Vtotal, cosmology)
        num_selected_synthetic = int(synthetic_number*volume_factor)
        num_synthetic_gals_in_snapshot = len(mpeak_synthetic_snapshot)
        synthetic_indices = np.arange(0, num_synthetic_gals_in_snapshot).astype(int)
        with NumpyRNGContext(seed):
            selected_synthetic_indices = np.random.choice(
                synthetic_indices, size=num_selected_synthetic, replace=False)
        print('...down-sampling synthetic galaxies with volume factor {} to yield {} selected synthetics'.format(volume_factor,
                                                                                                                 num_selected_synthetic))
        mstar_synthetic = mstar_synthetic_snapshot[selected_synthetic_indices]
        #  Apply additional M* cut to reduce number of synthetics for 5000 sq. deg. catalog
        if not image and mstar_min > 0:
            mstar_mask = (mstar_synthetic > mstar_min)
            print('...removing synthetics with M* < {:.1e} to yield {}  selected synthetics'.format(mstar_min,
                                                                                                    np.count_nonzero(mstar_mask)))
        else:
            mstar_mask = np.ones(len(mstar_synthetic), dtype=bool)

        mstar_synthetic = mstar_synthetic[mstar_mask]
        mpeak_synthetic = mpeak_synthetic_snapshot[selected_synthetic_indices][mstar_mask]
        magr_synthetic = magr_synthetic_snapshot[selected_synthetic_indices][mstar_mask]
        gr_synthetic = gr_synthetic_snapshot[selected_synthetic_indices][mstar_mask]
        ri_synthetic = ri_synthetic_snapshot[selected_synthetic_indices][mstar_mask]
        is_red_gr_synthetic = is_red_gr_synthetic_snapshot[selected_synthetic_indices][mstar_mask]
        is_red_ri_synthetic = is_red_ri_synthetic_snapshot[selected_synthetic_indices][mstar_mask]
        synthetic_dict = dict(
            mpeak=mpeak_synthetic, obs_sm=mstar_synthetic, restframe_extincted_sdss_abs_magr=magr_synthetic,
            restframe_extincted_sdss_gr=gr_synthetic, restframe_extincted_sdss_ri=ri_synthetic,
            is_on_red_sequence_gr=is_red_gr_synthetic, is_on_red_sequence_ri=is_red_ri_synthetic)

        ###################################################
        #  Map restframe Mr, g-r, r-i onto mock
        ###################################################
        #  Use the target halo redshift for those galaxies that have been selected;
        #  otherwise use the redshift of the snapshot of the target simulation
        print("...assigning rest-frame Mr and colors")
        check_time = time()
        if randomize_redshift_real:  # randomize unselected galaxy redshifts to ensure sufficient numbers for sub-step binning
            with NumpyRNGContext(seed):
                redshift_mock = np.random.normal(loc=redshift, scale=0.02, size=len(mock))
                redshift_mock = np.where(redshift_mock < 0, 0, redshift_mock)
        else:
            redshift_mock = np.zeros(len(mock)) + redshift
        redshift_mock[source_galaxy_indx] = np.repeat(
            target_halos['halo_redshift'], target_halos['richness'])

        if gaussian_smearing_real_redshifts > 0:
            msg = ("\n...gaussian_smearing_real_redshifts = {0}\n"
                "...The functions involved in the color modeling will be passed\n"
                "...noisy versions of the target halo redshifts")
            print(msg.format(gaussian_smearing_real_redshifts))
            with NumpyRNGContext(seed):
                redshift_mock[source_galaxy_indx] = np.random.normal(
                    loc=redshift_mock[source_galaxy_indx], scale=gaussian_smearing_real_redshifts)
        else:
            msg = ("\n...gaussian_smearing_real_redshifts = 0\n"
                "...Using the exact target halo redshifts "
                "to assign restframe colors to galaxies")
            print(msg)

        magr, gr_mock, ri_mock, is_red_gr, is_red_ri = assign_restframe_sdss_gri(
            mock['upid'], mock['obs_sm'], mock['sfr_percentile'],
            mock['host_halo_mvir'], redshift_mock, seed=seed, use_substeps=use_substeps_real,
            nzdivs=nzdivs)
        #  check for bad values
        for m_id, m in zip(['magr', 'gr', 'ri'], [magr, gr_mock, ri_mock]):
            num_infinite = np.sum(~np.isfinite(m))
            if num_infinite > 0:
                print('...Warning: {} infinite values in mock {}'.format(num_infinite, m_id))

        mock['restframe_extincted_sdss_abs_magr'] = magr
        mock['restframe_extincted_sdss_gr'] = gr_mock
        mock['restframe_extincted_sdss_ri'] = ri_mock
        mock['is_on_red_sequence_gr'] = is_red_gr
        mock['is_on_red_sequence_ri'] = is_red_ri
        print('...time to assign_restframe_sdss_gri = {:.2f} secs'.format(time()-check_time))

        ########################################################################
        #  Assemble the output mock by snapshot
        ########################################################################
        
        print("...building output snapshot mock for snapshot {}".format(snapshot))
        output_mock[snapshot] = build_output_snapshot_mock(float(redshift),
                mock, target_halos, source_galaxy_indx, galaxy_id_offset,
                synthetic_dict, Nside_cosmoDC2, cutout_number_true, float(previous_redshift),
                cosmology, H0=H0, volume_minx=volume_minx,
                volume_miny=volume_miny, volume_maxz=volume_maxz,
                halo_unique_id=halo_unique_id, redshift_method='halo', use_centrals=use_centrals)
        galaxy_id_offset = galaxy_id_offset + len(output_mock[snapshot]['galaxy_id'])  #increment offset
        #check that offset is within index bounds for imsim pixels
        if image and str(cutout_number_true) in cutout_remap.keys():
            galaxy_id_bound = cutout_number*cutout_id_offset_galaxy + z_offsets[z_range_id+1]
        else:
            galaxy_id_bound = cutout_number*cutout_id_offset_galaxy + z_offsets_not_im[z_range_id+1]
        if galaxy_id_offset > galaxy_id_bound:
            print('...Warning: galaxy_id bound of {} exceeded for snapshot {}'.format(galaxy_id_bound, snapshot))


        Ngals_total += len(output_mock[snapshot]['galaxy_id'])
        print('...saved {} galaxies to dict'.format(len(output_mock[snapshot]['galaxy_id'])))
        previous_redshift = redshift # update for next snap

        time_stamp = time()
        msg = "\nLightcone-shell runtime = {0:.2f} minutes"
        print(msg.format((time_stamp-new_time_stamp)/60.))

        mem = "Memory usage =  {0:.2f} GB"
        print(mem.format(process.memory_info().rss/1.e9))

    ########################################################################
    #  Write the output mock to disk
    ########################################################################
    if len(output_mock) > 0:
        check_time = time()
        write_output_mock_to_disk(output_color_mock_fname, output_mock, commit_hash, seed,
                                  synthetic_halo_minimum_mass, cutout_number_true, Nside_cosmoDC2)
        print('...time to write mock to disk = {:.2f} minutes'.format((time()-check_time)/60.))

    print('Maximum halo mass for {} ={}\n'.format(output_mock_basename, fof_halo_mass_max))
    print('Number of galaxies for {} ={}\n'.format(output_mock_basename, Ngals_total))

    time_stamp = time()
    msg = "\nEnd-to-end runtime = {0:.2f} minutes\n"
    print(msg.format((time_stamp-start_time)/60.))


def get_random_seed(filename, seed_max=4294967095):  #reduce max seed by 200 to allow for 60 light-cone shells
    import hashlib
    s = hashlib.md5(filename).hexdigest()
    seed = int(s, 16)

    #  enforce seed is below seed_max and odd
    seed = seed%seed_max
    if seed%2 == 0:
        seed = seed + 1
    return seed


def get_volume_factor(redshift, previous_redshift, Vtotal, cosmology):

    Vshell = cosmology.comoving_volume(float(redshift)).value - cosmology.comoving_volume(float(previous_redshift)).value
    return Vshell/Vtotal


def get_astropy_table(table_data, halo_unique_id=0, check=False, cosmology=None):
    """
    """
    t = Table()
    for k in table_data.keys():
        t[k] = table_data[k]

    t.rename_column('id', 'fof_halo_id')
    t['halo_redshift'] = 1/t['a'] - 1.
    t['halo_id'] = (np.arange(len(table_data['id']))*halo_id_offset + halo_unique_id).astype(int)

    #  rename column mass if found
    if mass in t.colnames:
        t.rename_column(mass, fof_halo_mass)
    elif fof_mass in t.colnames:
        t.rename_column(fof_mass, fof_halo_mass)
    else:
        print('  Warning; halo mass or fof_mass not found')

    #  check sod information and clean bad values
    if sod_mass in t.colnames:
        mask_valid = (t[sod_mass] > 0)
        mask = mask_valid & (t[sod_mass] < m_particle_1000)
        # overwrite
        for cn in ['sod_cdelta', 'sod_cdelta_error', sod_mass, 'sod_radius']:
            t[cn][mask] = -1

        print('...Overwrote {}/{} SOD quantities failing {:.2g} mass cut'.format(np.count_nonzero(mask),
                                                                                 np.count_nonzero(mask_valid),
                                                                                 m_particle_1000))

    if check and cosmology is not None:
        #  compute comoving distance from z and from position
        r = np.sqrt(t['x']*t['x'] + t['y']*t['y'] + t['z']*t['z'])
        comoving_distance = cosmology.comoving_distance(t['halo_redshift'])*H0/100.
        print('r == comoving_distance(z) is {}', np.isclose(r, comoving_distance))

    return t


def build_output_snapshot_mock(
            snapshot_redshift, umachine, target_halos, galaxy_indices, galaxy_id_offset,
            synthetic_dict, Nside, cutout_number_true, previous_redshift,
            cosmology, H0=H0, volume_minx=0., volume_miny=0., volume_maxz=0.,
            halo_unique_id=0, redshift_method='galaxy', use_centrals=True):
    """
    Collect the GalSampled snapshot mock into an astropy table

    Parameters
    ----------
    snapshot_redshift : float
        Float of the snapshot redshift

    umachine : astropy.table.Table
        Astropy Table of shape (num_source_gals, )
        storing the UniverseMachine snapshot mock

    target_halos : astropy.table.Table
        Astropy Table of shape (num_target_halos, )
        storing the target halo catalog

    galaxy_indices: ndarray
        Numpy indexing array of shape (num_target_gals, )
        storing integers valued between [0, num_source_gals)

    commit_hash : string
        Commit hash of the version of the cosmodc2 repo used when
        calling this function.

        After updating the cosmodc2 repo to the desired version,
        the commit_hash can be determined by navigating to the root
        directory and typing ``git log --pretty=format:'%h' -n 1``

    Returns
    -------
    dc2 : astropy.table.Table
        Astropy Table of shape (num_target_gals, )
        storing the GalSampled galaxy catalog
    """
    dc2 = Table()
    dc2['source_halo_id'] = umachine['hostid'][galaxy_indices]
    dc2['target_halo_id'] = np.repeat(
        target_halos['halo_id'], target_halos['richness'])

    #  copy lightcone information
    dc2['target_halo_redshift'] = np.repeat(
        target_halos['halo_redshift'], target_halos['richness'])
    dc2['target_halo_fof_halo_id'] = np.repeat(
        target_halos['fof_halo_id'], target_halos['richness'])
    dc2['lightcone_rotation'] = np.repeat(
        target_halos['rot'], target_halos['richness'])
    dc2['lightcone_replication'] = np.repeat(
        target_halos['rep'], target_halos['richness'])
    dc2['source_halo_mvir'] = np.repeat(
        target_halos['matching_mvir'], target_halos['richness'])

    idxA, idxB = crossmatch(dc2['target_halo_id'], target_halos['halo_id'])

    msg = "target IDs do not match!"
    assert np.all(dc2['source_halo_id'][idxA] == target_halos['source_halo_id'][idxB]), msg

    dc2['target_halo_x'] = 0.
    dc2['target_halo_y'] = 0.
    dc2['target_halo_z'] = 0.
    dc2['target_halo_vx'] = 0.
    dc2['target_halo_vy'] = 0.
    dc2['target_halo_vz'] = 0.

    dc2['target_halo_x'][idxA] = target_halos['x'][idxB]
    dc2['target_halo_y'][idxA] = target_halos['y'][idxB]
    dc2['target_halo_z'][idxA] = target_halos['z'][idxB]

    dc2['target_halo_vx'][idxA] = target_halos['vx'][idxB]
    dc2['target_halo_vy'][idxA] = target_halos['vy'][idxB]
    dc2['target_halo_vz'][idxA] = target_halos['vz'][idxB]

    dc2['target_halo_mass'] = 0.
    dc2['target_halo_mass'][idxA] = target_halos['fof_halo_mass'][idxB]

    dc2['target_halo_ellipticity'] = 0.
    dc2['target_halo_ellipticity'][idxA] = target_halos['halo_ellipticity'][idxB]

    dc2['target_halo_prolaticity'] = 0.
    dc2['target_halo_prolaticity'][idxA] = target_halos['halo_prolaticity'][idxB]

    dc2['target_halo_axis_A_length'] = 0.
    dc2['target_halo_axis_B_length'] = 0.
    dc2['target_halo_axis_C_length'] = 0.
    dc2['target_halo_axis_A_length'][idxA] = target_halos['axis_A_length'][idxB]
    dc2['target_halo_axis_B_length'][idxA] = target_halos['axis_B_length'][idxB]
    dc2['target_halo_axis_C_length'][idxA] = target_halos['axis_C_length'][idxB]

    dc2['target_halo_axis_A_x'] = 0.
    dc2['target_halo_axis_A_y'] = 0.
    dc2['target_halo_axis_A_z'] = 0.
    dc2['target_halo_axis_A_x'][idxA] = target_halos['axis_A_x'][idxB]
    dc2['target_halo_axis_A_y'][idxA] = target_halos['axis_A_y'][idxB]
    dc2['target_halo_axis_A_z'][idxA] = target_halos['axis_A_z'][idxB]

    # add SOD information from target_halo table
    dc2['sod_halo_cdelta'] = 0.
    dc2['sod_halo_cdelta_error'] = 0.
    dc2['sod_halo_mass'] = 0.
    dc2['sod_halo_radius'] = 0.
    dc2['sod_halo_cdelta'][idxA] = target_halos['sod_cdelta'][idxB]
    dc2['sod_halo_cdelta_error'][idxA] = target_halos['sod_cdelta_error'][idxB]
    dc2['sod_halo_mass'][idxA] = target_halos['sod_mass'][idxB]
    dc2['sod_halo_radius'][idxA] = target_halos['sod_radius'][idxB]

    #  Here the host_centric_xyz_vxvyvz in umachine should be overwritten
    #  Then we can associate x <--> A, y <--> B, z <--> C and then apply a random rotation
    #  It will be important to record the true direction of the major axis as a stored column
    source_galaxy_keys = ('host_halo_mvir', 'upid', 'mpeak',
            'host_centric_x', 'host_centric_y', 'host_centric_z',
            'host_centric_vx', 'host_centric_vy', 'host_centric_vz',
            'obs_sm', 'obs_sfr', 'sfr_percentile',
            'restframe_extincted_sdss_abs_magr',
            'restframe_extincted_sdss_gr', 'restframe_extincted_sdss_ri',
            'is_on_red_sequence_gr', 'is_on_red_sequence_ri',
            '_obs_sm_orig_um_snap', 'halo_id')
    for key in source_galaxy_keys:
        try:
            dc2[key] = umachine[key][galaxy_indices]
        except KeyError:
            msg = ("The build_output_snapshot_mock function was passed a umachine mock\n"
                "that does not contain the ``{0}`` key")
            raise KeyError(msg.format(key))

    max_umachine_halo_mass = np.max(umachine['mpeak'])
    ultra_high_mvir_halo_mask = (dc2['upid'] == -1) & (dc2['target_halo_mass'] > max_umachine_halo_mass)
    num_to_remap = np.count_nonzero(ultra_high_mvir_halo_mask)
    if num_to_remap > 0:
        print("...remapping stellar mass of {0} BCGs in ultra-massive halos".format(num_to_remap))

        halo_mass_array = dc2['target_halo_mass'][ultra_high_mvir_halo_mask]
        mpeak_array = dc2['mpeak'][ultra_high_mvir_halo_mask]
        mhalo_ratio = halo_mass_array/mpeak_array
        mstar_array = dc2['obs_sm'][ultra_high_mvir_halo_mask]
        redshift_array = dc2['target_halo_redshift'][ultra_high_mvir_halo_mask]
        upid_array = dc2['upid'][ultra_high_mvir_halo_mask]

        assert np.shape(halo_mass_array) == (num_to_remap, ), "halo_mass_array has shape = {0}".format(np.shape(halo_mass_array))
        assert np.shape(mstar_array) == (num_to_remap, ), "mstar_array has shape = {0}".format(np.shape(mstar_array))
        assert np.shape(redshift_array) == (num_to_remap, ), "redshift_array has shape = {0}".format(np.shape(redshift_array))
        assert np.shape(upid_array) == (num_to_remap, ), "upid_array has shape = {0}".format(np.shape(upid_array))
        assert np.all(mhalo_ratio >= 1), "Bookkeeping error: all values of mhalo_ratio ={0} should be >= 1".format(mhalo_ratio)

        dc2['obs_sm'][ultra_high_mvir_halo_mask] = mstar_array*(mhalo_ratio**0.5)
        dc2['restframe_extincted_sdss_abs_magr'][ultra_high_mvir_halo_mask] = magr_monte_carlo(
            dc2['obs_sm'][ultra_high_mvir_halo_mask], upid_array, redshift_array)
        idx = np.argmax(dc2['obs_sm'])
        halo_id_most_massive = dc2['halo_id'][idx]
        assert dc2['obs_sm'][idx] < 10**13.5, "halo_id = {0} has stellar mass {1:.3e}".format(
            halo_id_most_massive, dc2['obs_sm'][idx])

    satmask = dc2['upid'] != -1
    nsats = np.count_nonzero(satmask)
    host_conc = 5.
    if nsats > 0:
        host_Ax = dc2['target_halo_axis_A_x'][satmask]
        host_Ay = dc2['target_halo_axis_A_y'][satmask]
        host_Az = dc2['target_halo_axis_A_z'][satmask]
        b_to_a = dc2['target_halo_axis_B_length'][satmask]/dc2['target_halo_axis_A_length'][satmask]
        c_to_a = dc2['target_halo_axis_C_length'][satmask]/dc2['target_halo_axis_A_length'][satmask]
        host_centric_x, host_centric_y, host_centric_z = generate_triaxial_satellite_distribution(
            host_conc, host_Ax, host_Ay, host_Az, b_to_a, c_to_a)
        dc2['host_centric_x'][satmask] = host_centric_x
        dc2['host_centric_y'][satmask] = host_centric_y
        dc2['host_centric_z'][satmask] = host_centric_z

    dc2['x'] = dc2['target_halo_x'] + dc2['host_centric_x']
    dc2['vx'] = dc2['target_halo_vx'] + dc2['host_centric_vx']

    dc2['y'] = dc2['target_halo_y'] + dc2['host_centric_y']
    dc2['vy'] = dc2['target_halo_vy'] + dc2['host_centric_vy']

    dc2['z'] = dc2['target_halo_z'] + dc2['host_centric_z']
    dc2['vz'] = dc2['target_halo_vz'] + dc2['host_centric_vz']

    print('...number of galaxies before adding synthetic satellites = {}'.format(len(dc2['halo_id'])))
    print("...generating and stacking any synthetic cluster satellites")
    fake_cluster_sats = model_synthetic_cluster_satellites(dc2, Lbox=0., host_conc=host_conc,
                                                           tri_axial_positions=True) # turn off periodicity
    if len(fake_cluster_sats) > 0:
        check_time = time()
        dc2 = vstack((dc2, fake_cluster_sats))
        print('...time to create {} galaxies in fake_cluster_sats = {:.2f} secs'.format(len(fake_cluster_sats['target_halo_id']), time()-check_time))

    if len(synthetic_dict['mpeak']) > 0:
        check_time = time()
        if use_centrals:
            lowmass_mock = create_synthetic_lowmass_mock_with_centrals(
                umachine, dc2, synthetic_dict, previous_redshift, snapshot_redshift,
                cosmology, Nside=Nside, cutout_id=cutout_number_true, H0=H0,
                volume_minx=volume_minx, volume_miny=volume_miny, volume_maxz=volume_maxz,
                halo_id_offset=halo_id_offset, halo_unique_id=halo_unique_id)
        else:
            lowmass_mock = create_synthetic_lowmass_mock_with_satellites(
                umachine, dc2, synthetic_dict)
        if len(lowmass_mock) > 0:
            dc2 = vstack((dc2, lowmass_mock)) # astropy vstack pads missing values with zeros in lowmass_mock
            print('...time to create {} galaxies in synthetic_lowmass_mock = {:.2f} secs'.format(len(lowmass_mock['target_halo_id']), time()-check_time))

    dc2['galaxy_id'] = np.arange(galaxy_id_offset, galaxy_id_offset + len(dc2['target_halo_id'])).astype(int)
    print('...Min and max galaxy_id = {} -> {}'.format(np.min(dc2['galaxy_id']), np.max(dc2['galaxy_id'])))

    #  Use gr and ri color to compute gi flux
    dc2['restframe_extincted_sdss_abs_magg'] = (
        dc2['restframe_extincted_sdss_gr'] +
        dc2['restframe_extincted_sdss_abs_magr'])
    dc2['restframe_extincted_sdss_abs_magi'] = (
        -dc2['restframe_extincted_sdss_ri'] +
        dc2['restframe_extincted_sdss_abs_magr'])

    #  compute galaxy redshift, ra and dec
    if redshift_method is not None:
        r = np.sqrt(dc2['x']*dc2['x'] + dc2['y']*dc2['y'] + dc2['z']*dc2['z'])
        mask = (r > 5000.)
        if np.sum(mask) > 0:
            print('WARNING: Found {} co-moving distances > 5000'.format(np.sum(mask)))

        dc2['redshift'] = dc2['target_halo_redshift']  #  copy halo redshifts to galaxies
        if redshift_method == 'galaxy':
            #  generate distance estimates for values between min and max redshifts
            zmin = np.min(dc2['redshift'])
            zmax = np.max(dc2['redshift'])
            zgrid = np.logspace(np.log10(zmin), np.log10(zmax), 50)
            CDgrid = cosmology.comoving_distance(zgrid)*H0/100.
            #  use interpolation to get redshifts for satellites only
            sat_mask = (dc2['upid'] != -1)
            dc2['redshift'][sat_mask] = np.interp(r[sat_mask], CDgrid, zgrid)
            if redshift_method == 'pec':
                #  TBD add pecliar velocity correction
                print('Not implemented')

        dc2['dec'] = 90. - np.arccos(dc2['z']/r)*180.0/np.pi  #  co-latitude
        dc2['ra'] = np.arctan2(dc2['y'], dc2['x'])*180.0/np.pi
        dc2['ra'][(dc2['ra'] < 0)] += 360.   #  force value 0->360

    #convert table to dict
    check_time = time()
    output_dc2 = {}
    for k in dc2.keys():
        output_dc2[k] = dc2[k].quantity.value

    print('...time to new dict = {:.4f} secs'.format(time()-check_time))

    return output_dc2


def get_skyarea(output_mock, Nside):
    """
    """
    import healpy as hp
    #  compute sky area from ra and dec ranges of galaxies
    nominal_skyarea = np.rad2deg(np.rad2deg(4.0*np.pi/hp.nside2npix(Nside)))
    if Nside > 8:
        skyarea = nominal_skyarea
    else:
        pixels = set()
        for k in output_mock.keys():
            if output_mock[k].has_key('ra') and output_mock[k].has_key('dec'):
                for ra, dec in zip(output_mock[k]['ra'], output_mock[k]['dec']):
                    pixels.add(hp.ang2pix(Nside, ra, dec, lonlat=True))
        frac = len(pixels)/float(hp.nside2npix(Nside))
        skyarea = frac*np.rad2deg(np.rad2deg(4.0*np.pi))
        if np.isclose(skyarea, nominal_skyarea, rtol=.02):  #  agreement to about 1 sq. deg.
            print(' Replacing calculated sky area {} with nominal_area'.format(skyarea))
            skyarea = nominal_skyarea
        if np.isclose(skyarea, nominal_skyarea/2., rtol=.01):  #  check for half-filled pixels
            print(' Replacing calculated sky area {} with (nominal_area)/2'.format(skyarea))
            skyarea = nominal_skyarea/2.

    return skyarea


def write_output_mock_to_disk(output_color_mock_fname, output_mock, commit_hash, seed,
                              synthetic_halo_minimum_mass, cutout_number, Nside,
                              versionMajor=1, versionMinor=1, versionMinorMinor=1):
    """
    """

    print("...writing to file {} using commit hash {}".format(output_color_mock_fname, commit_hash))
    hdfFile = h5py.File(output_color_mock_fname, 'w')
    hdfFile.create_group('metaData')
    hdfFile['metaData']['commit_hash'] = commit_hash
    hdfFile['metaData']['seed'] = seed
    hdfFile['metaData']['versionMajor'] = versionMajor
    hdfFile['metaData']['versionMinor'] = versionMinor
    hdfFile['metaData']['versionMinorMinor'] = versionMinorMinor
    hdfFile['metaData']['H_0'] = H0
    hdfFile['metaData']['Omega_matter'] = OmegaM
    hdfFile['metaData']['Omega_b'] = OmegaB
    hdfFile['metaData']['skyArea'] = get_skyarea(output_mock, Nside)
    hdfFile['metaData']['synthetic_halo_minimum_mass'] = synthetic_halo_minimum_mass
    hdfFile['metaData']['healpix_cutout_number'] = cutout_number

    for k, v in output_mock.items():
        gGroup = hdfFile.create_group(k)
        check_time = time()
        for tk in v.keys():
            check_atime = time()
            #gGroup[tk] = v[tk].quantity.value
            gGroup[tk] = v[tk]

        print('.....time to write group {} = {:.4f} secs'.format(k, time()-check_time))

    check_time = time()
    hdfFile.close()
    print('.....time to close file {:.4f} secs'.format(time()-check_time))
