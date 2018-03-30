"""
"""
from astropy.table import Table
import numpy as np
import os
import fnmatch


__all__ = ('load_um_binary_sfr_catalog', 'reformat_umachine_binary_output')


default_sfr_catalog_dtype = np.dtype([('id', '<i8'), ('descid', '<i8'), ('upid', '<i8'),
    ('flags', '<i4'), ('uparent_dist', '<f4'), ('pos', '<f4', (6,)),
    ('vmp', '<f4'), ('lvmp', '<f4'), ('mp', '<f4'), ('m', '<f4'), ('v', '<f4'),
    ('r', '<f4'), ('rank1', '<f4'), ('rank2', '<f4'), ('ra', '<f4'),
    ('rarank', '<f4'), ('t_tdyn', '<f4'), ('sm', '<f4'), ('icl', '<f4'),
    ('sfr', '<f4'), ('obs_sm', '<f4'), ('obs_sfr', '<f4'), ('obs_uv', '<f4'), ('foo', '<f4')])


def reformat_umachine_binary_output(fname,
        keys_to_keep=['id', 'upid', 'vmp', 'mp', 'm', 'v', 'sm', 'sfr', 'obs_sm', 'obs_sfr']):
    """
    """
    t = Table(load_um_binary_sfr_catalog(fname))

    keys_to_keep.append('pos')
    for key in t.keys():
        if key not in keys_to_keep:
            t.remove_column(key)

    t['x'] = t['pos'][:, 0]
    t['y'] = t['pos'][:, 1]
    t['z'] = t['pos'][:, 2]
    t['vx'] = t['pos'][:, 3]
    t['vy'] = t['pos'][:, 4]
    t['vz'] = t['pos'][:, 5]
    t.remove_column('pos')

    t.rename_column('vmp', 'vpeak')
    t.rename_column('mp', 'mpeak')
    t.rename_column('m', 'mvir')
    t.rename_column('v', 'vmax')
    t.rename_column('id', 'halo_id')

    return t


def load_um_binary_sfr_catalog(fname, dtype=default_sfr_catalog_dtype):
    """ Read the binary UniverseMachine outputs sfr_catalog_XXX.bin into
    a Numpy structured array.

    The returned data structure contains every UniverseMachine galaxy at the
    redshift of the snapshot.

    Parameters
    ----------
    fname : string
        Absolute path to the binary file

    dtype : Numpy dtype, optional
        Numpy dtype defining the format of the returned structured array.

        The default option is compatible with the particular
        outputs on Edison used to generate protoDC2, but in general this argument
        must be compatible with the catalog_halo struct declared in
        the make_sf_catalog.h UniverseMachine file.

    Returns
    -------
    arr : Numpy structured array
        UniverseMachine mock galaxy catalog at a single snapshot
    """
    return np.fromfile(fname, dtype=dtype)


def fname_generator(root_dirname, basename_filepat):
    """ Yield the absolute path of all files in the directory tree of ``root_dirname``
    with a basename matching the input pattern
    """

    for path, dirlist, filelist in os.walk(root_dirname):
        for filename in fnmatch.filter(filelist, basename_filepat):
            yield os.path.join(path, filename)


def _parse_scale_factor_from_umachine_sfr_catalog_fname(fname):
    """
    """
    basename = os.path.basename(fname)
    ifirst = len(basename) - basename[::-1].find('_')
    ilast = len(basename) - basename[::-1].find('.') - 1
    return float(basename[ifirst:ilast])


def find_closest_available_umachine_snapshot(z, dirname):
    """
    """
    available_fnames = list(fname_generator(dirname, 'sfr_catalog*.hdf5'))

    f = _parse_scale_factor_from_umachine_sfr_catalog_fname
    available_snaps = np.array([1./f(fname) - 1. for fname in available_fnames])

    return available_fnames[np.argmin(np.abs(z - available_snaps))]


def find_closest_available_bpl_halocat(z, dirname):
    """
    """
    available_fnames = list(fname_generator(dirname, 'hlist*.hdf5'))

    f = _parse_scale_factor_from_umachine_sfr_catalog_fname
    available_snaps = np.array([1./f(fname) - 1. for fname in available_fnames])

    return available_fnames[np.argmin(np.abs(z - available_snaps))]


def retrieve_list_of_filenames(redshift_list, halocat_dirname, um_dirname):
    """
    """
    halocat_fname_list = list(find_closest_available_bpl_halocat(z, halocat_dirname)
        for z in redshift_list)
    um_fname_list = list(find_closest_available_umachine_snapshot(z, um_dirname)
        for z in redshift_list)
    return um_fname_list, halocat_fname_list




