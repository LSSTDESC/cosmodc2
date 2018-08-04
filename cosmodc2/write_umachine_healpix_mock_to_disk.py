""" Module storing the primary driver script used for the v4 release of DC2.
"""
import os
import psutil
import numpy as np
import h5py
import re
from time import time
from astropy.table import Table, vstack
from astropy.cosmology import FlatLambdaCDM
from cosmodc2.sdss_colors import assign_restframe_sdss_gri
from cosmodc2.stellar_mass_remapping import remap_stellar_mass_in_snapshot
from galsampler import halo_bin_indices, source_halo_index_selection
from galsampler.cython_kernels import galaxy_selection_kernel
from halotools.utils import crossmatch

from cosmodc2.synthetic_subhalos import model_extended_mpeak, map_mstar_onto_lowmass_extension
from cosmodc2.synthetic_subhalos import create_synthetic_lowmass_mock_with_centrals
from cosmodc2.synthetic_subhalos import create_synthetic_lowmass_mock_with_satellites
from cosmodc2.synthetic_subhalos import create_synthetic_cluster_satellites

fof_halo_mass = 'fof_halo_mass'
mass = 'mass'
fof_max = 14.5
H0 = 71.0
OmegaM = 0.2648
OmegaB = 0.0448
cutoff_id_offset = 1e8  #  offset to guarantee unique galaxy ids across cutout files
z_offsets = [0, 2e7, 6e7]  #  offset to guarantee unique galaxy ids across z-ranges

Nside = 2048  #  fine pixelization for determining sky area
Nside_cosmoDC2 = 8


def write_umachine_healpix_mock_to_disk(
            umachine_mstar_ssfr_mock_fname_list, umachine_host_halo_fname_list,
            healpix_data, snapshots, output_color_mock_fname,
            redshift_list, commit_hash,
            synthetic_halo_minimum_mass=9.8, num_synthetic_gal_ratio=1., use_centrals=False, Lbox=3000.):
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

    cutout_number = file_ids[-1]
    z_range_id = file_ids[0]
    galaxy_id_offset = int(cutout_number*cutoff_id_offset + z_offsets[z_range_id])

    #  determine seed from output filename
    seed = get_random_seed(output_mock_basename)

    #  initialize book-keeping variables
    fof_halo_mass_max = 0.
    Ngals_total = 0

    print('\nStarting snapshot processing')
    print('Synthetic-halo minimum mass =  {}'.format(synthetic_halo_minimum_mass))
    print('Using galaxy-id offset = {}'.format(galaxy_id_offset))
    print('Using {} synthetic low-mass galaxies'.format('central' if use_centrals else 'satellite'))
    for a, b, c, d in gen:
        umachine_mock_fname = a
        umachine_halos_fname = b
        redshift = c
        snapshot = d

        new_time_stamp = time()

        #  seed should be changed for each new shell
        seed = seed + 2

        #  Get galaxy properties from UM catalogs and target halo properties
        print("\n...loading z = {0:.2f} galaxy catalog into memory".format(redshift))
        mock = Table.read(umachine_mock_fname, path='data')

        ###  GalSampler
        print("\n...loading z = {0:.2f} halo catalogs into memory".format(redshift))
        source_halos = Table.read(umachine_halos_fname, path='data')
        #
        target_halos = get_astropy_table(healpix_data[snapshot])
        fof_halo_mass_max = max(np.max(target_halos[fof_halo_mass].quantity.value), fof_halo_mass_max)

        print("...Finding halo--halo correspondence with GalSampler")
        #  Bin the halos in each simulation by mass
        dlogM = 0.15
        mass_bins = 10.**np.arange(10.5, 14.5+dlogM, dlogM)
        source_halos['mass_bin'] = halo_bin_indices(
            mass=(source_halos['mvir'], mass_bins))
        target_halos['mass_bin'] = halo_bin_indices(
            mass=(target_halos[fof_halo_mass], mass_bins))

        #  Randomly draw halos from corresponding mass bins
        nhalo_min = 10
        source_halo_bin_numbers = source_halos['mass_bin']
        target_halo_bin_numbers = target_halos['mass_bin']
        target_halo_ids = target_halos['halo_id']
        _result = source_halo_index_selection(source_halo_bin_numbers,
                      target_halo_bin_numbers, target_halo_ids, nhalo_min, mass_bins, seed=seed)
        source_halo_indx, matching_target_halo_ids = _result

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
        num_selected_galaxies = len(source_galaxy_indx)
        corrected_mpeak, mpeak_synthetic = model_extended_mpeak(
                mock['mpeak'], num_selected_galaxies, synthetic_halo_minimum_mass, Lbox=Lbox)
        mock['mpeak'] = corrected_mpeak

        #  Select (num_synthetic_gal_ratio*num_selected_galaxies) synthetic galaxies to keep file size down
        num_synthetic_galaxies = int(num_synthetic_gal_ratio*num_selected_galaxies)
        mpeak_synthetic = np.random.choice(mpeak_synthetic, size=num_synthetic_galaxies, replace=False)
        print('...assembling {} synthetic galaxies'.format(num_synthetic_galaxies))

        ########################################################################
        #  Assign stellar mass, using Outer Rim halo mass for very massive halos
        ########################################################################
        print("...re-assigning high-mass mstar values")
        #  For mock central galaxies that have been assigned to a very massive target halo,
        #  we use the target halo mass instead of the source halo mpeak to assign M*
        #  Allocate an array storing the target halo mass for galaxies selected by GalSampler,
        #  with -1 in all other entries pertaining to unselected galaxies

        mock_target_halo_mass = np.zeros(len(mock)) - 1.
        mock_target_halo_mass[source_galaxy_indx] = np.repeat(
            target_halos['fof_halo_mass'], target_halos['richness'])

        #  Calculate a boolean mask for those centrals that get mapped to very massive target halos
        cenmask = mock['upid'] == -1
        massive_target_halo_mask = mock_target_halo_mass > np.max(mock['mpeak'])
        remap_mpeak_mask = cenmask & massive_target_halo_mask
        mpeak_mock = np.where(remap_mpeak_mask, mock_target_halo_mass, mock['mpeak'])
        assert np.all(mpeak_mock > 0), "Bookkeeping error in remapping target halo mass onto cluster BCGs"

        #  Map stellar mass onto mock using target halo mass instead of UM Mpeak for cluster BCGs
        new_mstar = remap_stellar_mass_in_snapshot(redshift, mpeak_mock, mock['obs_sm'])
        mock.rename_column('obs_sm', '_obs_sm_orig_um_snap')
        mock['obs_sm'] = new_mstar

        #  Add call to map_mstar_onto_lowmass_extension function after pre-determining low-mass slope
        print("...re-assigning low-mass mstar values")
        new_mstar_real, mstar_synthetic = map_mstar_onto_lowmass_extension(
            mock['mpeak'], mock['obs_sm'], mpeak_synthetic)
        mock['obs_sm'] = new_mstar_real

        #  Assign target halo id and target halo mass to selected galaxies in mock
        mock_target_halo_id = np.zeros(len(mock)) - 1.
        mock_target_halo_id[source_galaxy_indx] = np.repeat(
            target_halos['halo_id'], target_halos['richness'])
        mock['target_halo_id'] = mock_target_halo_id
        mock['target_halo_mass'] = mock_target_halo_mass

        ###################################################
        #  Map restframe Mr, g-r, r-i onto mock
        ###################################################
        #  Use the target halo redshift for those galaxies that have been selected;
        #  otherwise use the redshift of the snapshot of the target simulation
        print("...assigning rest-frame Mr and colors")
        check_time = time()
        redshift_mock = np.zeros(len(mock)) + redshift
        redshift_mock[source_galaxy_indx] = np.repeat(
            target_halos['halo_redshift'], target_halos['richness'])
        mock['target_halo_redshift'] = redshift_mock  #  used later for synthetic galaxies

        #  Allocate an array storing the target halo mass for galaxies selected by GalSampler,
        #  with mock['host_halo_mvir'] in all other entries pertaining to unselected galaxies
        mock_remapped_halo_mass = mock['host_halo_mvir']
        mock_remapped_halo_mass[source_galaxy_indx] = np.repeat(
            target_halos['fof_halo_mass'], target_halos['richness'])

        magr, gr_mock, ri_mock, is_red_gr, is_red_ri = assign_restframe_sdss_gri(
            mock['upid'], mock['obs_sm'], mock['sfr_percentile'],
            mock_remapped_halo_mass, redshift_mock, seed=seed)
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
        output_mock[snapshot] = build_output_snapshot_mock(
                mock, target_halos, source_galaxy_indx, galaxy_id_offset,
                mpeak_synthetic, mstar_synthetic, Nside_cosmoDC2, cutout_number,
                redshift_method='halo', use_centrals=use_centrals)
        galaxy_id_offset = galaxy_id_offset + len(output_mock[snapshot]['halo_id'])  #increment offset

        Ngals_total += len(output_mock[snapshot]['galaxy_id'])

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
                                  synthetic_halo_minimum_mass)
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


def get_astropy_table(table_data, check=False):
    """
    """
    t = Table()
    for k in table_data.keys():
        t[k] = table_data[k]

    t.rename_column('id', 'fof_halo_id')
    t['halo_redshift'] = 1/t['a'] - 1.
    t['halo_id'] = np.arange(len(table_data['id'])).astype(int)

    #  rename column mass if found
    if mass in t.colnames:
        t.rename_column(mass, fof_halo_mass)

    if check:
        #  compute comoving distance from z and from position
        cosmology = FlatLambdaCDM(H0=H0, Om0=OmegaM)
        r = np.sqrt(t['x']*t['x'] + t['y']*t['y'] + t['z']*t['z'])
        comoving_distance = cosmology.comoving_distance(t['halo_redshift'])*H0/100.
        print('r == comoving_distance(z) is {}', np.isclose(r, comoving_distance))

    return t


def build_output_snapshot_mock(
            umachine, target_halos, galaxy_indices, galaxy_id_offset,
            mpeak_synthetic, mstar_synthetic, Nside, cutout_number,
            redshift_method='galaxy', use_centrals=True):
    """
    Collect the GalSampled snapshot mock into an astropy table

    Parameters
    ----------
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
    umachine.rename_column('target_halo_id', 'um_target_halo_id')

    #  copy lightcone information
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
    umachine.rename_column('target_halo_mass', 'um_target_halo_mass')

    source_galaxy_keys = ('host_halo_mvir', 'upid', 'mpeak',
            'host_centric_x', 'host_centric_y', 'host_centric_z',
            'host_centric_vx', 'host_centric_vy', 'host_centric_vz',
            'obs_sm', 'obs_sfr', 'sfr_percentile',
            'restframe_extincted_sdss_abs_magr',
            'restframe_extincted_sdss_gr', 'restframe_extincted_sdss_ri',
            'is_on_red_sequence_gr', 'is_on_red_sequence_ri',
            'um_target_halo_id', 'um_target_halo_mass', 'target_halo_redshift',
            '_obs_sm_orig_um_snap', 'halo_id')
    for key in source_galaxy_keys:
        try:
            dc2[key] = umachine[key][galaxy_indices]
        except KeyError:
            msg = ("The build_output_snapshot_mock function was passed a umachine mock\n"
                "that does not contain the ``{0}`` key")
            raise KeyError(msg.format(key))

    dc2['x'] = dc2['target_halo_x'] + dc2['host_centric_x']
    dc2['vx'] = dc2['target_halo_vx'] + dc2['host_centric_vx']

    dc2['y'] = dc2['target_halo_y'] + dc2['host_centric_y']
    dc2['vy'] = dc2['target_halo_vy'] + dc2['host_centric_vy']

    dc2['z'] = dc2['target_halo_z'] + dc2['host_centric_z']
    dc2['vz'] = dc2['target_halo_vz'] + dc2['host_centric_vz']

    print('...number of galaxies before adding synthetic satellites = {}'.format(len(dc2['halo_id'])))
    print("...generating and stacking any synthetic cluster satellites")
    fake_cluster_sats = create_synthetic_cluster_satellites(dc2, Lbox=0.) # turn off periodicity
    if len(fake_cluster_sats) > 0:
        check_time = time()
        dc2 = vstack((dc2, fake_cluster_sats))
        print('...time to create {} galaxies in fake_cluster_sats = {:.2f} secs'.format(len(fake_cluster_sats['halo_id']), time()-check_time))

    if len(mpeak_synthetic) > 0:
        check_time = time()
        if use_centrals:
            lowmass_mock = create_synthetic_lowmass_mock_with_centrals(
                umachine, dc2, mpeak_synthetic, mstar_synthetic, Nside=Nside, cutout_id=cutout_number,
                H0=H0, OmegaM=OmegaM)
        else:
            lowmass_mock = create_synthetic_lowmass_mock_with_satellites(
                umachine, dc2, mpeak_synthetic, mstar_synthetic)
        if len(lowmass_mock) > 0:
            dc2 = vstack((dc2, lowmass_mock))
            print('...time to create {} galaxies in synthetic_lowmass_mock = {:.2f} secs'.format(len(lowmass_mock['halo_id']), time()-check_time))

    dc2['galaxy_id'] = np.arange(galaxy_id_offset, galaxy_id_offset + len(dc2['halo_id'])).astype(int)

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
            zmin = np.min(dc2['redshift_halo_only'])
            zmax = np.max(dc2['redshift_halo_only'])
            zgrid = np.logspace(np.log10(zmin), np.log10(zmax), 50)
            cosmology = FlatLambdaCDM(H0=H0, Om0=OmegaM)
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


def get_skyarea(output_mock):
    """
    """
    import healpy as hp
    #  compute sky area from ra and dec ranges of galaxies
    nominal_skyarea = np.rad2deg(np.rad2deg(4.0*np.pi/hp.nside2npix(Nside_cosmoDC2)))
    pixels = set()
    for k in output_mock.keys():
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
                              synthetic_halo_minimum_mass):
    """
    """

    print("...writing to file {} using commit hash {}".format(output_color_mock_fname, commit_hash))
    hdfFile = h5py.File(output_color_mock_fname, 'w')
    hdfFile.create_group('metaData')
    hdfFile['metaData']['commit_hash'] = commit_hash
    hdfFile['metaData']['seed'] = seed
    hdfFile['metaData']['versionMajor'] = 0
    hdfFile['metaData']['versionMinor'] = 0
    hdfFile['metaData']['versionMinorMinor'] = 0
    hdfFile['metaData']['H_0'] = H0
    hdfFile['metaData']['Omega_matter'] = OmegaM
    hdfFile['metaData']['Omega_b'] = OmegaB
    hdfFile['metaData']['skyArea'] = get_skyarea(output_mock)
    hdfFile['metaData']['synthetic_halo_minimum_mass'] = synthetic_halo_minimum_mass

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
