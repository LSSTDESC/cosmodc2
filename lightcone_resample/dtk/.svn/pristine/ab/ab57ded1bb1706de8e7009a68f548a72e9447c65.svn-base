
import numpy as np
import scipy.optimize as opt
from scipy.stats import norm
from util import *

def conf_interval_helper(x, pdf, conf_level):
    return np.sum(pdf[pdf > x])-conf_level


def conf_interval(H,clvls):
    clvls = np.array(clvls)
    result = np.zeros(clvls.size)
    clvls = clvls*np.sum(H)
    for i in range(0,clvls.size):
        result[i] = opt.brentq(conf_interval_helper,0.,np.sum(H),args=(H,clvls[i]))
    return result


def contour_labels(lvls,c_lvls):
    result = {}
    for i in range(0,len(c_lvls)):
        a = "%2.f%%"%(100.0*lvls[i])
        result[c_lvls[i]] = a
    return result

def smoothen_H(data):
    xsize = data[:,0].size
    ysize = data[0,:].size
    result = np.zeros(data.shape)
    for i in range(-1,1):
        for j in range(-1,1):
            result[0+max(i,0):xsize+min(i,0),0+max(j,0):ysize+min(j,0)] += data[0-min(i,0):xsize-max(i,0),0-min(j,0):ysize-max(j,0)]
    result[0,:] += 3.0*data[0,:]
    result[-1,:] += 3.0*data[-1,:]
    result[1:,0] += 3.0*data[1:,0]
    result[1:,-1] += 3.0*data[1:,-1]
    return result/9.0

def get_sigmas(vals):
    return norm.cdf(vals)


def quick_contour(x_bins, y_bins, H, cmap=None, levels=None,
                  label=True, smoothen=True, colors=None, bins_edges = True):
    if bins_edges:
        x_bins_avg = bins_avg(x_bins)
        y_bins_avg = bins_avg(y_bins)
    else:
        x_bins_avg = x_bins
        y_bins_avg = y_bins
    if(smoothen):
        Hs = smoothen_H(H)
    else:
        Hs = H
    if(levels is None):
        levels = np.array( [0.5,0.67,.75,0.86,0.95,0.99])
    else:
        levels = np.array(levels)
    contour_lvls = conf_interval(Hs,levels)
    print('contour_lvls', contour_lvls)
    if(cmap is None):
        cmap = plt.cm.gist_heat_r
    if(colors is None):
        cm_lvls = np.linspace(0,1,levels.size)
        colors = cmap(cm_lvls)
    cs = plt.contour(x_bins_avg,y_bins_avg,Hs.T,levels=contour_lvls,colors=colors)
    if(label):
        labels = contour_labels(levels,contour_lvls)
        plt.clabel(cs,fmt=labels)
    return cs
