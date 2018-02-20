"""
"""
import os


__author__ = ('Eve Kovacs', 'Dan Korytov')


def _get_galacticus_basename_from_redshift(redshift):
    raise NotImplementedError()


def _load_galacticus_snapshot(fname):
    raise NotImplementedError()


def galacticus_mock_from_umachine_mock(galsampled_lightcone_fname, galacticus_snapshot_dirname,
            output_lightcone_fname):
    """
    """
    galacticus_basename = _get_galacticus_basename_from_redshift(redshift)
    galacticus_fname = os.path.join(galacticus_snapshot_dirname, galacticus_basename)
    galacticus_snapshot = _load_galacticus_snapshot(galacticus_fname)
    raise NotImplementedError()
