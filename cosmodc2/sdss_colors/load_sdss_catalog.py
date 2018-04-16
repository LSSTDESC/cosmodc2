"""
"""
from astropy.table import Table


default_fname = "/Users/aphearin/Dropbox/protoDC2/SDSS/dr10_mgs_colors_processed.txt"


__all__ = ('load_umachine_processed_sdss_catalog', )


def load_umachine_processed_sdss_catalog(fname=default_fname):
    """ Return the baseline DR10 SDSS catalog used by UniverseMachine.

    Parameters
    ----------
    fname : string, optional
        Absolute path to the file.

    Returns
    -------
    catalog : Astropy Table
        Table storing the SDSS galaxy sample. See Notes for details.

    Notes
    -----
    This SDSS sample is defined by DR10 SkyServer queries written by Peter Behroozi
    with the perl scripts get_mgs_colors.pl and process_mgs_colors.pl.
    The most important columns for DC2 purposes are as follows:

    * `Rmag` - Absolute r-band Petrosian magnitude k-corrected to z=0.1
    * `GRcolor` and `RIColor` - Petrosian-based magnitudes k-corrected to z=0.1
    * `SM` - Log10 of stellar mass, taken from MPA-JHU
    * `SSFR` - Log10 of M*/SFR, taken from MPA-JHU
    * `DR7PhotoObjID` - Integer that can be used to cross-match into the DR7 database

    Absolute magnitudes, stellar mass and star-formation rate are all quoted
    assuming h=0.7. Explicit notes explaining the little-h convention appear below:

    Let us denote the stellar mass evaluated/observed assuming h=1
    with the notation M*[h=1], and the stellar mass evaluated/observed
    assuming h=0.7 as M*[h=0.7]. Then we have: M*[h=1] = h*h*M*[h=0.7].
    The stellar masses appearing in the returned catalog are M*[h=0.7].
    """
    from halotools.utils import sliding_conditional_percentile

    t = Table.read(fname, format='ascii.commented_header')
    for key in t.keys():
        t.rename_column(key, key.lower())
    t.rename_column('ricolor', 'restframe_extincted_sdss_ri')
    t.rename_column('grcolor', 'restframe_extincted_sdss_gr')
    t.rename_column('rmag', 'restframe_extincted_sdss_abs_magr')
    t.rename_column('size[kpc]', 'size_kpc')
    t.rename_column('dr7photoobjid', 'dr7_PhotoObj_Table_ID')

    t['sfr_percentile_fixed_sm'] = sliding_conditional_percentile(
            t['sm'], t['ssfr'], 301)
    t['gr_percentile_fixed_magr'] = sliding_conditional_percentile(
            t['restframe_extincted_sdss_abs_magr'], t['restframe_extincted_sdss_gr'], 301)

    t = t[t['restframe_extincted_sdss_abs_magr'] > -50]
    t = t[t['restframe_extincted_sdss_gr'] > -0.5]
    t = t[t['restframe_extincted_sdss_gr'] < 1.25]
    t = t[t['restframe_extincted_sdss_ri'] > -0.5]
    t = t[t['restframe_extincted_sdss_ri'] < 1.25]

    return t



