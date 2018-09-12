import os
import glob
import h5py
import numpy as np
from mask_DC2 import read_selections
from mask_DC2 import mask_cat

lowz_lib = '/gpfs/mira-fs0/projects/DarkUniverse_esp/dkorytov/data/Galacticus/low_z/galaxy_library/*.hdf5'
hiz_lib = '/gpfs/mira-fs0/projects/DarkUniverse_esp/dkorytov/data/Galacticus/high_z/galaxy_library/*.hdf5'
galaxyProperties = 'galaxyProperties'

#Rv variables
Lum_v = 'otherLuminosities/totalLuminositiesStellar:V:rest'
Lum_v_dust = 'otherLuminosities/totalLuminositiesStellar:V:rest:dustAtlas'
Lum_b = 'otherLuminosities/totalLuminositiesStellar:B:rest'
Lum_b_dust = 'otherLuminosities/totalLuminositiesStellar:B:rest:dustAtlas'

def _calc_Av(lum_v, lum_v_dust):
    with np.errstate(divide='ignore', invalid='ignore'):
        Av = -2.5*(np.log10(lum_v_dust/lum_v))
        Av[lum_v_dust == 0] = np.nan
        return Av

def _calc_Rv(lum_v, lum_v_dust, lum_b, lum_b_dust):
    with np.errstate(divide='ignore', invalid='ignore'):
        v = lum_v_dust/lum_v
        b = lum_b_dust/lum_b
        bv = b/v
        Rv = np.log10(v) / np.log10(bv)
        Rv[v == b] = np.nan
        Rv[(v == 1) & (b == 1)] = 1.0
        return Rv

def _calc_Rv_2(lum_v, lum_v_dust, lum_b, lum_b_dust):
    with np.errstate(divide='ignore', invalid='ignore'):
        v = np.log10(lum_v_dust) - np.log10(lum_v)
        b = np.log10(lum_b_dust) - np.log10(lum_b)
        bv = b - v
        Rv = v / bv
        return Rv

def _calc_Rv_3(lum_v, lum_v_dust, lum_b, lum_b_dust):
    with np.errstate(divide='ignore', invalid='ignore'): #makes no numerical  difference
        Av = -2.5*np.log10(lum_v_dust) + 2.5*np.log10(lum_v)
        Ebv = -2.5*np.log10(lum_b_dust) + 2.5*np.log10(lum_b) - 2.5*np.log10(lum_v_dust) + 2.5*np.log10(lum_v)
        Rv = Av / Ebv
        return Rv

def _calc_Rv_4(lum_v, lum_v_dust, lum_b, lum_b_dust):
    with np.errstate(divide='ignore', invalid='ignore'):
        M_v = _flux_to_mag(lum_v)
        M_v_dust = _flux_to_mag(lum_v_dust)
        M_b = _flux_to_mag(lum_b)
        M_b_dust = _flux_to_mag(lum_b_dust)
        Av = M_v_dust - M_v
        Ebv = M_b_dust - M_b - M_v_dust + M_v
        Rv = Av / Ebv
        return Rv

def _calc_Rv_mag(M_v, M_v_dust, M_b, M_b_dust):
    with np.errstate(divide='ignore', invalid='ignore'): #makes no numerical  difference
        Av = M_v_dust - M_v
        Ebv = M_b_dust - M_b - M_v_dust + M_v
        Rv = Av / Ebv
        return Rv

def _flux_to_mag(lum):
    mag = -2.5*np.log10(lum)
    return mag

def check_Rv_masking(Rv): 
    mask_good = np.isfinite(Rv)
    print('...Calculated {} non-finite--Rv values (total={})'.format(np.sum(~mask_good), len(Rv)))
    mask_zero = np.isclose(Rv, 0.)
    print('...Found {} zero-Rv values'.format(np.sum(mask_zero)))
    mask_ok = mask_good & ~mask_zero
    print('...Accepting {} Rv values'.format(np.sum(mask_ok)))
    return mask_ok

def test_Rv(nfiles=None, yaml='Rv'):
    lib_files = sorted(glob.glob(lowz_lib) + glob.glob(hiz_lib))

    yamlfile = os.path.join('yaml',yaml+'.yaml')
    selections = read_selections(yamlfile=yamlfile)

    R_v = {}
    for lib in lib_files[0:nfiles]:
        key = os.path.splitext(os.path.basename(lib))[0]
        print('Checking {}'.format(os.path.basename(lib)))
        catalog = h5py.File(lib, 'r')
        mask = mask_cat(catalog, selections)
        n_ok = np.sum(mask)
        lum_v = catalog[galaxyProperties][Lum_v].value
        lum_v_dust = catalog[galaxyProperties][Lum_v_dust].value
        lum_b = catalog[galaxyProperties][Lum_b].value
        lum_b_dust = catalog[galaxyProperties][Lum_b_dust].value
        with np.errstate(divide='ignore', invalid='ignore'):        
            M_v = _flux_to_mag(lum_v)
            M_v_dust = _flux_to_mag(lum_v_dust)
            M_b = _flux_to_mag(lum_b)
            M_b_dust = _flux_to_mag(lum_b_dust)

        R_v[key] = {}
        for func in [_calc_Rv, _calc_Rv_2, _calc_Rv_3, _calc_Rv_4]:
            R_v[key][str(func)] = {}
            print('Checking {}'.format(func))
            if 'mag' in str(func):
                Rv = func(M_v, M_v_dust, M_b, M_b_dust)
            else:
                Rv = func(lum_v, lum_v_dust, lum_b, lum_b_dust)
            mask_ok = check_Rv_masking(Rv)
            agree = (np.sum(mask_ok) == n_ok)
            print('Mask and function evaluation {} AGREE'.format('' if agree else 'DO NOT'))
            Rv = Rv[mask_ok]
            R_v[key][str(func)]['Rv'] = Rv
            R_v[key][str(func)]['mask'] = mask_ok
            print('Min |Rv| = {:.3e};  Min/Max Rv = {:.3e}/{:.3e}\n'.format(np.min(np.abs(Rv)), np.min(Rv), np.max(Rv)))
                     

        #calculate diagreement using different function of Rv
        for func in [_calc_Rv, _calc_Rv_2, _calc_Rv_3]:
            print('Comparing recalculation of Rv with {} after applying original mask'.format(func))
            lum_b_masked = lum_b[mask]
            lum_v_masked = lum_v[mask]
            lum_b_dust_masked = lum_b_dust[mask]
            lum_v_dust_masked = lum_v_dust[mask]
            Rv_masked = func(lum_v_masked, lum_v_dust_masked, lum_b_masked, lum_b_dust_masked)
            mask_ok = check_Rv_masking(Rv_masked)            
            
        print('Finished check\n')

    return R_v

#R_v = test_Rv(nfiles=1)
