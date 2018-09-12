import healpy as hp
from healpy.pixelfunc import pix2ang
from healpy.pixelfunc import ang2pix
from healpy.pixelfunc import get_all_neighbours
import glob
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from healpy.pixelfunc import pix2ang
from healpy.pixelfunc import ang2pix
from healpy.pixelfunc import get_all_neighbours
filedir='/global/projecta/projectdirs/lsst/groups/CS/cosmoDC2/cosmoDC2_v1.0.0_full_highres/'
filelist=glob.glob(filedir+'z_0_1*')

healpixels = np.asarray(sorted([int(re.findall(r'\d+', re.split('healpix', f)[-1])[0]) for f in filelist]))

dec_min = -44.33
dec_max = -27.25
ra_min_lo = 49.92
ra_max_lo = 73.79
ra_min_hi = 52.25
ra_max_hi = 71.46
ppr = 8 #pixels per row

Nside = 32
ra, dec = pix2ang(32, healpixels, lonlat=True)

ra_mask = (ra >= ra_min_lo) & (ra <= ra_max_lo)
dec_mask = (dec >= dec_min) & (dec <= dec_max)
mask = ra_mask & dec_mask
images = healpixels[mask]
image_pixels = images.tolist()

#check corners
ne_corner = ang2pix(32, ra_max_hi, dec_max, lonlat=True)
nw_corner = ang2pix(32, ra_min_hi, dec_max, lonlat=True)
se_corner = ang2pix(32, ra_max_lo, dec_min, lonlat=True)
sw_corner = ang2pix(32, ra_min_lo, dec_min, lonlat=True)
corners =[nw_corner, ne_corner, sw_corner, se_corner]
#add rows of pixels between corners
n_row = [h for h in range(nw_corner, ne_corner+1)]
s_row = [h for h in range(sw_corner, se_corner+1)]

top_edge = [get_all_neighbours(32, h,lonlat=True)[4] for h in image_pixels[0:ppr]]
bottom_edge = [get_all_neighbours(32, h,lonlat=True)[6] for h in image_pixels[-ppr:]]
bottom_edge=[]
east_pixels = [image_pixels[n] for n in range(len(image_pixels)-1) if image_pixels[n]!= image_pixels[n+1]-1]
west_pixels = [image_pixels[n] for n in range(len(image_pixels)) if image_pixels[n]!= image_pixels[n-1]+1]
right_edge = [get_all_neighbours(32, h,lonlat=True)[6] for h in east_pixels]
left_edge = [get_all_neighbours(32, h,lonlat=True)[2] for h in west_pixels]

_all = sorted(list(set(sorted(image_pixels + top_edge + bottom_edge + right_edge + left_edge + n_row + s_row))))
print('number of healpixels = {}'.format(len(_all)))

hp_map = np.empty(hp.nside2npix(32))
hp_map.fill(hp.UNSEEN)
hp_map[_all] = 0
hp_map[corners]=1
hp.mollview(hp_map, title='cosmoDC2_v1.0_image', coord='C', cbar=None)
plt.savefig('./cosmoDC2_v1.0_image.png')
hp.cartview(hp_map, title='cosmoDC2_v1.0_image', coord='C', cbar=None)
plt.savefig('./cosmoDC2_v1.0_image_flat.png')

sublist = [_all[n*26:n*26+26]  for n in range(5)]
print(', '.join([str(d) for d in sublist[0]]))
print(', '.join([str(d) for d in sublist[1]]))
print(', '.join([str(d) for d in sublist[2]]))
print(', '.join([str(d) for d in sublist[3]]))
print(', '.join([str(d) for d in sublist[4]] + [str(_all[-1])]))


