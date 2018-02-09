"""
"""
import numpy as np
from astropy.table import Table

sed_fname = "/Users/aphearin/Dropbox/protoDC2/catsim_galaxy_seds/sdss_mags_of_galaxy_seds.txt"

__all__ = ('read_bruzual_charlot_library', )


def _read_fnames(fname):
    with open(fname, 'rb') as f:
        header = next(f)
        while True:
            try:
                raw_line = next(f)
                line = raw_line.strip().split()
                yield line[0]
            except StopIteration:
                break


def _read_ugriz(fname):
    with open(fname, 'rb') as f:
        header = next(f)
        while True:
            try:
                raw_line = next(f)
                line = raw_line.strip().split()
                yield tuple(s for s in line[1:])
            except StopIteration:
                break


def read_bruzual_charlot_library(fname=sed_fname):
    ugriz = np.array(list(_read_ugriz(fname)), dtype='f4')
    ngals = ugriz.shape[0]

    spectra = Table()
    spectra['u'] = ugriz[:, 0]
    spectra['g'] = ugriz[:, 1]
    spectra['r'] = ugriz[:, 2]
    spectra['i'] = ugriz[:, 3]
    spectra['z'] = ugriz[:, 4]
    spectra['specID'] = np.arange(ngals).astype('i4')
    spectra['fname'] = list(_read_fnames(fname))

    spectra['metallicity'] = [parse_metallicity(name) for name in spectra['fname']]
    spectra['age'] = [parse_age(name) for name in spectra['fname']]
    spectra['sfh_model'] = [parse_model(name) for name in spectra['fname']]

    return spectra


def split_fname(fname):
    suffix = '.spec.gz'
    end = len(suffix)
    s = fname[:-end]
    return s.split('.')


def parse_model(fname):
    model_string = split_fname(fname)[0]
    return model_string


def parse_age(fname):
    age_string = split_fname(fname)[1]
    return float(age_string)/10.


def parse_metallicity(fname):
    z_string = split_fname(fname)[2]
    s = z_string[:-1]
    if s[0] == '0':
        return float('.'+s)
    else:
        return float(s[0]+'.'+s[1:])
