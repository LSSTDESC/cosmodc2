import numpy as np
import sys
sys.path.append("/gpfs/mira-home/ekovacs/cosmology/genericio/python")
import genericio as gio
import os
from astropy.table import Table


halo_dir = {'alphaQ': '/projects/DarkUniverse_esp/heitmann/OuterRim/M000/L360/HACC001/analysis/Halos/b0168/fofp_new',
           }
halo_file_template = {'alphaQ': 'm000-{}.fofproperties',
                     } 

property_template ='fof_halo_{}'
property_list = ['center_{}', 'mean_{}', 'mean_v{}']
property_modifiers =['x', 'y', 'z']
other_properties = ['tag', 'mass']

def load_gio_halo_snapshot(snapshot_number, sim_name='alphaQ'):
    filename = os.path.join(halo_dir[sim_name], halo_file_template[sim_name].format(str(snapshot_number)))
    #gio.gio_inspect(filename) #list all properties
    properties_list = [p.format(m) for p in property_list for m in property_modifiers]
    halo_properties_list = sorted([property_template.format(p) for p in other_properties + properties_list])

    halo_table = Table()
    for halo_prop in halo_properties_list:
        print('Reading column {} in snapshot {}'.format(halo_prop, snapshot_number)) 
        halo_table[halo_prop] = gio.gio_read(filename, halo_prop)

    return halo_table

