""" Module storing the primary driver script used for the v4 release of DC2.
"""
import os
import psutil
import numpy as np
import h5py
from time import time
from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM
from cosmodc2.sdss_colors import assign_restframe_sdss_gri
from cosmodc2.stellar_mass_remapping import lift_high_mass_mstar
from galsampler import halo_bin_indices, source_halo_index_selection
from galsampler.cython_kernels import galaxy_selection_kernel
#from cosmodc2.load_gio_halos import load_gio_halo_snapshot
from halotools.empirical_models import enforce_periodicity_of_box
from halotools.utils import crossmatch
#from cosmodc2.lightcone_id import append_lightcone_id, astropy_table_to_lightcone_hdf5

fof_halo_mass ='fof_halo_mass'
fof_max = 14.5
H0 = 71.0
OmegaM = 0.2648

def write_umachine_healpix_mock_to_disk(
            umachine_mstar_ssfr_mock_fname_list, umachine_host_halo_fname_list,
            healpix_data, snapshots, output_color_mock_fname,
            redshift_list, commit_hash):
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

    output_color_mock_fname_list : string
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


    """

    output_mock = {}
    gen = zip(umachine_mstar_ssfr_mock_fname_list, umachine_host_halo_fname_list, redshift_list, snapshots)
    start_time = time()
    process = psutil.Process(os.getpid())
    
    #determine seed from output filename
    seed = get_random_seed(os.path.basename(output_color_mock_fname))

    for a, b, c, d in gen:
        umachine_mock_fname = a
        umachine_halos_fname = b
        redshift = c
        snapshot = d

        new_time_stamp = time()

        print("\n...loading z = {0:.2f} galaxy catalog into memory".format(redshift))

        #could check if this needs to be done again (sometimes same snap is used several times)
        mock = Table.read(umachine_mock_fname, path='data')

        upid_mock = mock['upid']
        mstar_mock = mock['obs_sm']
        sfr_percentile_mock = mock['sfr_percentile']
        host_halo_mvir_mock = mock['host_halo_mvir']
        redshift_mock = np.random.uniform(redshift-0.15, redshift+0.15, len(mock))
        redshift_mock = np.where(redshift_mock < 0, 0, redshift_mock)
        mpeak_mock = mock['mpeak']

        print("\n...re-assigning high-mass mstar values")
        new_mstar = lift_high_mass_mstar(mpeak_mock, mstar_mock, upid_mock, redshift_mock)
        mock['obs_sm'] = new_mstar

        print("\n...assigning SDSS restframe colors")

        #seed should be changed for each new shell
        seed = seed + 2
        magr, gr_mock, ri_mock, is_red_gr, is_red_ri = assign_restframe_sdss_gri(
            upid_mock, new_mstar, sfr_percentile_mock, host_halo_mvir_mock, redshift_mock, seed=seed)
        mock['restframe_extincted_sdss_abs_magr'] = magr
        mock['restframe_extincted_sdss_gr'] = gr_mock
        mock['restframe_extincted_sdss_ri'] = ri_mock
        mock['is_on_red_sequence_gr'] = is_red_gr
        mock['is_on_red_sequence_ri'] = is_red_ri

        ###  GalSampler
        print("\n...loading z = {0:.2f} halo catalogs into memory".format(redshift))
        source_halos = Table.read(umachine_halos_fname, path='data')
        #
        target_halos = get_astropy_table(healpix_data[snapshot])

        print("...Finding halo--halo correspondence with GalSampler")
        #  Bin the halos in each simulation by mass
        dlogM = 0.15
        mass_bins = 10.**np.arange(10.5, fof_max+dlogM, dlogM)
        source_halos['mass_bin'] = halo_bin_indices(
            mass=(source_halos['mvir'], mass_bins))
        target_halos['mass_bin'] = halo_bin_indices(
            mass=(target_halos['fof_halo_mass'], mass_bins))

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
        print("...GalSampling z={0:.2f} galaxies to AlphaQ halos".format(redshift))

        source_galaxy_indx = np.array(galaxy_selection_kernel(
            target_halos['first_galaxy_index'].astype('i8'),
            target_halos['richness'].astype('i4'), target_halos['richness'].sum()))

        ########################################################################
        #  Assemble the output mock by snapshot
        ########################################################################

        print("...building output snapshot mock for snapshot {}".format(snapshot))
        output_mock[snapshot] = build_output_snapshot_mock(
                mock, target_halos, source_galaxy_indx)

        time_stamp = time()
        msg = "Lightcone-shell runtime = {0:.2f} minutes"
        print(msg.format((time_stamp-new_time_stamp)/60.))  

        mem = "Memory usage =  {0:.2f} GB"
        print(mem.format(process.memory_info().rss/1.e9))

    ########################################################################
    #  Write the output mock to disk
    ########################################################################
    if len(output_mock) > 0:
        write_output_mock_to_disk(output_color_mock_fname, output_mock, commit_hash, seed)

    time_stamp = time()
    msg = "End-to-end runtime = {0:.2f} minutes"
    print(msg.format((time_stamp-start_time)/60.))


def get_random_seed(filename, seed_max=4294967095): #reduce max seed by 200 to allow for 60 light-cone shells
    import hashlib
    s = hashlib.md5(filename).hexdigest()
    seed = int(s, 16)

    #enforce seed is below seed_max and odd
    seed = seed%seed_max
    if seed%2 == 0:
        seed = seed +1
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

    if check:
        #compute comoving distance from z and from position  
        cosmology = FlatLambdaCDM(H0=H0, Om0=OmegaM)
        r = np.sqrt(t['x']*t['x'] + t['y']*t['y'] + t['z']*t['z'])
        comoving_distance = cosmology.comoving_distance(t['halo_redshift'])*H0/100.
        print('r == comoving_distance(z) is {}',np.isclose(r, comoving_distance))
              
    return t


def build_output_snapshot_mock(
            umachine, target_halos, galaxy_indices, redshift_method='halo'):
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
    #dc2 = Table(meta={'commit_hash': commit_hash, 'seed': seed})
    dc2 = Table()
    dc2['source_halo_id'] = umachine['hostid'][galaxy_indices]
    dc2['target_halo_id'] = np.repeat(
        target_halos['halo_id'], target_halos['richness'])

    #copy lightcone information 
    dc2['target_halo_fof_halo_id'] = np.repeat(
        target_halos['fof_halo_id'], target_halos['richness'])
    dc2['lightcone_rotation'] = np.repeat(
        target_halos['rot'], target_halos['richness'])
    dc2['lightcone_replication'] = np.repeat(
        target_halos['rep'], target_halos['richness'])

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
    
    source_galaxy_keys = ('host_halo_mvir', 'upid', 'mpeak',
            'host_centric_x', 'host_centric_y', 'host_centric_z',
            'host_centric_vx', 'host_centric_vy', 'host_centric_vz',
            'obs_sm', 'obs_sfr', 'sfr_percentile',
            'restframe_extincted_sdss_abs_magr',
            'restframe_extincted_sdss_gr', 'restframe_extincted_sdss_ri',
            'is_on_red_sequence_gr', 'is_on_red_sequence_ri')
    for key in source_galaxy_keys:
        dc2[key] = umachine[key][galaxy_indices]

    dc2['x'] = dc2['target_halo_x'] + dc2['host_centric_x']
    dc2['vx'] = dc2['target_halo_vx'] + dc2['host_centric_vx']

    dc2['y'] = dc2['target_halo_y'] + dc2['host_centric_y']
    dc2['vy'] = dc2['target_halo_vy'] + dc2['host_centric_vy']

    dc2['z'] = dc2['target_halo_z'] + dc2['host_centric_z']
    dc2['vz'] = dc2['target_halo_vz'] + dc2['host_centric_vz']

    #compute galaxy redshift, ra and dec
    if redshift_method is not None:
        r = np.sqrt(dc2['x']*dc2['x'] + dc2['y']*dc2['y'] + dc2['z']*dc2['z'])
        if redshift_method=='halo':
            #set galaxy redshifts to halo redshifts
            dc2['redshift'] = np.repeat(target_halos['halo_redshift'], target_halos['richness'])
        else:              #compute galaxy redshift from position 
            #generate distance estimates for values between min and max redshifts
            zmin = np.min(target_halos['halo_redshift'])
            zmax = np.max(target_halos['halo_redshift'])
            zgrid = np.logspace(np.log10(zmin), np.log10(zmax), 50)
            cosmology = FlatLambdaCDM(H0=H0, Om0=OmegaM)
            CDgrid = cosmology.comoving_distance(zgrid)*H0/100.
            #use interpolation to get redshifts
            dc2['redshift'] = np.interp(r, zgrid, CDgrid)
            if redshift_method=='pec':
                #TBD add pecliar velocity correction
                print('Not implemented')
    
        dc2['dec'] = np.arccos(dc2['z'] /r)*180.0/np.pi
        dc2['ra'] = np.arctan(dc2['y']/dc2['x'])*180.0/np.pi

    return dc2


def write_output_mock_to_disk(output_color_mock_fname, output_mock, commit_hash, seed):
    """
    """
    print("...writing to file {} using commit hash {}".format(output_color_mock_fname, commit_hash))
    hdfFile=h5py.File(output_color_mock_fname, 'w')
    hdfFile.create_group('metaData')
    hdfFile['metaData']['commit_hash'] = commit_hash
    hdfFile['metaData']['seed'] = seed

    for k, v in output_mock.items():
        gGroup=hdfFile.create_group(k)
        for tk in v.keys():
            gGroup[tk] = v[tk].quantity.value

    hdfFile.close()

