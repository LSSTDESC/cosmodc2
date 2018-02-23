import os
import sys
sys.path.append("/gpfs/mira-home/ekovacs/cosmology/genericio/python")
import genericio as gio
from astropy.table import Table

property_template ='fof_halo_{}'
#property_list = ['center_{}', 'mean_{}', 'mean_v{}']
property_list = ['center_{}', 'mean_v{}']
property_modifiers =['x', 'y', 'z']
other_properties = ['tag', 'mass']

def load_gio_halo_snapshot(filename):
    #gio.gio_inspect(filename) #list all properties
    properties_list = [p.format(m) for p in property_list for m in property_modifiers]
    halo_properties_list = sorted([property_template.format(p) for p in other_properties + properties_list])
    print('Reading halo file {}'.format(os.path.split(filename)[-1]))

    halo_table = Table()
    for halo_prop in halo_properties_list:
        print('Reading column {}'.format(halo_prop)) 
        halo_table[halo_prop] = gio.gio_read(filename, halo_prop)

    return halo_table

