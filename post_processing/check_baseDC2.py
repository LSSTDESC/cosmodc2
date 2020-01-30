import h5py
import numpy as np
from astropy.table import Table
import os
import re
from matplotlib import pyplot as plt
from matplotlib import rc
from matplotlib.backends.backend_pdf import PdfPages
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)


def get_table(f): 
    t = Table()
    for k in f.keys():
        t[k] = f[k]
    
    return t
def get_data(fh):
    data = {}
    keys = list(k for k in fh.keys() if k.isdigit())
    for s in keys:
        data[s] = get_table(fh[s])
        
    return data


def save_fig(fig, plotdir, figname, title='', hspace=.05):
    fig.suptitle(title, size=16)
    fig.tight_layout()
    fig.subplots_adjust(hspace=hspace)
    fig.subplots_adjust(top=0.94)
    file_type = '.pdf' if 'pdf' in plotdir else '.png'
    figname = os.path.join(plotdir, re.sub('_+', '_', figname) + file_type)
    print('  Saving {}\n'.format(figname))
    fig.savefig(figname, bbox_inches='tight')
    #plt.close(fig)

def get_subsample(npoints, ntotal):
    index=np.asarray([])
    if ntotal > 0:
        sample=np.random.sample(ntotal)
        fraction=min(float(npoints)/float(ntotal),1.0)
        index=(sample<fraction)
    return index
        
def plot_scatter(data, x='target_halo_mass', y='obs_sm', plotdir='cosmology/DC2/validation',
                 xlabel='$M_{halo}$', ylabel='$M^{*}$', figname='Mstar_vs_Mhalo',
                 snaps=['247', '331', '401', '487'], nsample=5000):
    fig, ax_all = plt.subplots(2, 2, figsize=(10, 10))
    
    for s, ax in zip(snaps, ax_all.flatten()):
        index = get_subsample(nsample, len(data[s][x]))
        print(np.min(np.log10(data[s][x][index])), np.max(np.log10(data[s][x][index])))
        ax.scatter(np.log10(data[s][x][index]), np.log10(data[s][y][index]),
                   label='baseDC2: snap={}'.format(s))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(loc='best', numpoints=1)
        
    save_fig(fig, plotdir, figname)
    return fig


# In[6]:


plotdir = '/gpfs/mira-home/ekovacs/cosmology/DC2/validation'
figname = 'Mstar_vs_Mhalo'
#fig = plot_scatter(data, nsample=5000, plotdir=plotdir)
#plt.close(fig)
figname = 'Mhalo_vs_Mhalo'
#fig = plot_scatter(data, nsample=5000, y='source_halo_mvir', xlabel='$M^{target}_{halo}$', ylabel='$M^{source}_{halo}$', plotdir=plotdir, figname = 'Mhalo_vs_Mhalo')


# In[7]:


# compute angles
def get_angles(data, logMcut=13.5):
    for s in snaps:
        data[s]['A_phi'] = np.arctan2(data[s]['target_halo_axis_A_y'], data[s]['target_halo_axis_A_x'])
        data[s]['A_theta'] = np.arccos(data[s]['target_halo_axis_A_z']/data[s]['target_halo_axis_A_length'])
        #mask = (data[s]['target_halo_axis_A_length'] > 0)
        mask = (np.log10(data[s]['target_halo_mass']) > logMcut)
        print('Number in {} = {}/{}'.format(s, np.count_nonzero(mask), len(data[s])))
        if np.count_nonzero(mask) > 0:
            print('phi', np.min(data[s]['A_phi'][mask]), np.max(data[s]['A_phi'][mask]))
            print('Az', np.min(data[s]['target_halo_axis_A_z'][mask]), np.max(data[s]['target_halo_axis_A_z'][mask]))
            print('A', np.min(data[s]['target_halo_axis_A_length'][mask]), np.max(data[s]['target_halo_axis_A_length'][mask]))
            print('theta', np.min(data[s]['A_theta'][mask]), np.max(data[s]['A_theta'][mask]))
    return data

def plot_dist(data, phi='A_phi', theta='A_theta', plotdir='cosmology/DC2/validation',
              xlabel='$A_{\alpha}$', ylabel='$N$', figname='A_alpha', nbins=40, logMcut=13.5,
              snaps=['247', '307', '365', '401']):
    fig, ax_all = plt.subplots(2, 2, figsize=(10, 10))
    Nbins = np.arange(-np.pi, np.pi, nbins)
    
    for s, ax in zip(snaps, ax_all.flatten()):
        for q, c, l in zip([phi, theta], ['r', 'blue'], ['$\phi$', '$\theta$']):
            print(np.min(data[s][q]), np.max(data[s][q]))
            ax.hist(data[s][q], bins=Nbins, color=c, label=l)
            
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title('baseDC2: snap={}, $M_{{halo}} > {}'.format(s, logMcut))
        ax.legend(loc='best', numpoints=1)
        
    save_fig(fig, plotdir, figname)
    return fig


# In[33]:


def plot_coords(data, plotdir='cosmology/DC2/validation', x='target_halo_axis_A_{}',
                coords=['x', 'y', 'z', 'length'],
                xlabel='$A_{{{}}}$', ylabel='$N$', figname='A', nbins=40, logMcuts=[11, 12, 13, 13.5],
                snaps=['247']):
    for s in snaps:
        fig, ax_all = plt.subplots(2, 2, figsize=(10, 10))
        for co, ax in zip(coords, ax_all.flatten()):
            q = x.format(co)
            for cut, c in zip(logMcuts, ['c', 'blue', 'g', 'r']):
                if cut is not None:
                    mask = (np.log10(data[s]['target_halo_mass']) > cut)
                else:
                    mask = np.ones(len(data[s][q]), dtype=bool)
                l = '$\log_{{10}}(M_{{halo}}) > {}$'.format(cut)
                if np.count_nonzero(mask) > 0:
                    print(co, cut, np.count_nonzero(mask), np.min(data[s][q][mask]), np.max(data[s][q][mask]))
                    ax.hist(data[s][q][mask], bins=nbins, color=c, label=l, alpha=0.6)

            ax.set_xlabel(xlabel.format(co))
            ax.set_ylabel(ylabel)
            ax.legend(loc='best', numpoints=1)
        
    save_fig(fig, plotdir, figname, title='baseDC2: snap={}'.format(s), hspace=0.1)
    return fig


# In[34]:


#fig = plot_coords(data)


# In[ ]:


#data = get_angles(data)


# In[ ]:


#fig = plot_dist(data)


# In[ ]:


halo_shapes_dir = '/gpfs/mira-fs0/projects/DarkUniverse_esp/kovacs/OR_5000/OR_haloshapes'
shape_file_template = 'shapes_{}_l.hdf5'
def get_shapes(snaps):
    shapes = {}
    for s in snaps:
        # fix missing steps
        if s == '347':
            sn = '338'
        else:
            sn = '253' if s == '259' else s
        fn = os.path.join(halo_shapes_dir, shape_file_template.format(sn))
        if os.path.isfile(fn):
            fh = h5py.File(fn)
            shapes[s] = get_table(fh)
        
    return shapes
        

fn = '/gpfs/mira-fs0/projects/DarkUniverse_esp/kovacs/OR_5000/baseDC2_v0.1/baseDC2_z_0_1_cutout_9554.hdf5'
hpx = '/gpfs/mira-fs0/projects/DarkUniverse_esp/kovacs/OR_5000/healpix_cutouts/z_0_1/cutout_9554.hdf5'
evalues = 'eigenvalues_SIT_COM'
evectors = 'eigenvectors_SIT_COM'



def run_check(fn, hpx, Mcut=2e13):
    fh = h5py.File(fn)
    snaps = list(k for k in fh.keys() if k.isdigit())
    print(snaps, fh[snaps[0]].keys())
    data = get_data(fh)
    shapes = get_shapes(snaps)
    print(shapes.keys(), shapes['487'].keys())
    hp = h5py.File(hpx)
    snaps.reverse()
    for s in snaps:
        print('Checking {}'.format(s))
        fof_orig = np.sort(hp[s]['id'].value)
        #get real halos
        fof_mask = (data[s]['target_halo_fof_halo_id'] > 0)
        fof_cat = data[s]['target_halo_fof_halo_id'][fof_mask]
        fof_unq = np.unique(fof_cat)
        print('Lengths: cutout {}; cat(unique) {}: equal? {}'.format(len(fof_orig),
                                                                     len(fof_unq),
                                                                     np.array_equal(fof_orig, fof_unq)))
        fof_shape = shapes[s]['fof_halo_tag']
        id_mask = np.in1d(fof_cat, fof_shape) #locations in fof_cat if in fof_shape
        unq_mask = np.in1d(fof_unq, fof_shape)
        msg = ''
        if np.count_nonzero(id_mask) > 0:
            msg = 'Min mass = {:.4e}'.format(np.min(data[s]['target_halo_mass'][fof_mask][id_mask]))
        print('{} halos in shapes; {}'.format(np.count_nonzero(unq_mask), msg))

        #find positions in shapes
        matched_ids = fof_unq[unq_mask]
        shape_mask = np.in1d(fof_shape, matched_ids)
        #reorder evalues
        reorder = shapes[s][evalues].argsort()
        nvals = len(shapes[s][evalues])
        ordered_evals = np.asarray([shapes[s][evalues][i][reorder[i]] for i in range(nvals)])
        # get major axis length for matched halos
        a = np.sort(np.sqrt(ordered_evals[:, 2][shape_mask]))
        # get cat values
        fof_match = np.in1d(data[s]['target_halo_fof_halo_id'], matched_ids)
        A = np.unique(data[s]['target_halo_axis_A_length'][fof_match])
        print('Check: catalog A_length = sqrt(max(shape(evalues))? {}'.format(all(np.isclose(A, a))))


    return data, shapes

