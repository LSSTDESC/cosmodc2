import sys
import os
import glob
import numpy as np
from astropy.table import Table
import argparse
home = '/gpfs/mira-home/ekovacs'
sys.path.insert(0, '/gpfs/mira-home/ekovacs/.local/lib/python2.7/site-packages')
sys.path.insert(0, os.path.join(home, 'cosmology/cosmodc2'))
sys.path.insert(0, os.path.join(home, 'cosmology/galsampler/build/lib.linux-x86_64-2.7'))
sys.path.insert(0, os.path.join(home, 'cosmology/halotools/build/lib.linux-x86_64-2.7'))

from cosmodc2.load_gio_halos import load_gio_halo_snapshot
from cosmodc2.get_fof_info  import get_fof_info

parser = argparse.ArgumentParser()
parser.add_argument("-halo_catalog_dirname",
    help="Directory name storing halo catalogs",
    default='/projects/DarkUniverse_esp/heitmann/OuterRim/M000/L360/HACC001/analysis/Halos/b0168/fofp_new')
parser.add_argument("-snapshot",
    help="number of snapshot",
    default='499')
parser.add_argument("-output_dirname",
    help="Directory name storing output hdf5 halo catalogs",
    default='/gpfs/mira-fs0/projects/DarkUniverse_esp/kovacs/AlphaQ/alphaq_halo_catalogs_hdf5')

args = parser.parse_args()

halo_file_template = 'm000-{}.fofproperties'
halo_filename = os.path.join(args.halo_catalog_dirname, halo_file_template.format(args.snapshot))

halo_table = load_gio_halo_snapshot(halo_filename)

output_filename = os.path.join(args.output_dirname,  halo_file_template.format(args.snapshot)+'.hdf5')
halo_table.write(output_filename, path='data', overwrite=True)

print('Writing output file {}'.format(output_filename))
