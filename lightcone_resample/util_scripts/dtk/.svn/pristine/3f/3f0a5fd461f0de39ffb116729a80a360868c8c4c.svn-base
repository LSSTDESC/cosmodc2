import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import random
import scipy.misc
import scipy.spatial



    
def autocorr3D_N2(datax,datay,dataz,box_length=None,bins=None):
    max_dist = box_length/2.0
    size = len(datax)
    bins2 = bins**2
    result = np.zeros(bins.size-1)
    for i in range(0,size):
        for j in range(0,i):
            dist_x = np.abs(datax[i]-datax[j])
            dist_y = np.abs(datay[i]-datay[j])
            dist_z = np.abs(dataz[i]-dataz[j])
            if(dist_x > box_length/2.0):
                dist_x -= box_length
            if(dist_y > box_length/2.0):
                dist_y -= box_length
            if(dist_z > box_length/2.0):
                dist_z -= box_length
            dist = np.sqrt(dist_x**2+dist_y**2+dist_z**2)
            dist = dist_x**2+dist_y**2+dist_z**2
            indx = np.searchsorted(bins2,dist)
            if(indx > 1 and  indx < (bins.size-1)):
                result[indx-1]+=2.0
    density = float(datax.size-1)/(box_length**3)            
    bin_vol = 4.0/3.0*np.pi*(bins[1:]**3-bins[:-1]**3)
    expected = bin_vol*density*datax.size
    return result/expected-1.0, result, expected

def autocorr3D_KDTree(datax,datay,dataz,box_length=None,bins=None):
    data = np.zeros((datax.size,3))
    data[:,0]=datax;
    data[:,1]=datay;
    data[:,2]=dataz;
    kdtree = scipy.spatial.KDTree(data,10)
    result = kdtree.count_neighbors(kdtree,bins)
    result2=result[1:]-result[:-1]
    bin_vol = 4.0/3.0*np.pi*(bins[1:]**3-bins[:-1]**3)
    density = float(datax.size-1)/(box_length**3)
    expected = bin_vol*density*float(datax.size)
    final_result = result2/expected -1.0
    return final_result , result2,expected

def power_spectrum3D(datax,datay,dataz,n,domain):
    """Takes x,y,z coordinates, the grid size, the size of the box and
    calculates the linear power spectrum"""
    assert(len(datax) == len(datay) and len(datay) == len(dataz))
    grid = np.zeros((n,n,n),dtype=np.float32)
    for i in range(0,len(datax)):
        grid[int(datax[i]/domain*n),int(datay[i]/domain*n),int(dataz[i]/domain*n)] +=1
    power_spectrum3D_grid(grid,n,domain)

def power_spectrum3D_grid(grid,n,domain):
    space_ps = np.abs(np.fft.fftn(grid))
    space_ps *= space_ps
    fft_x = np.fft.fftfreq(n,domain/n)
    ps_1d = from_3d_to_1d(space_ps,n)
    return [ps_1d,fft_x[0:n/2]]

def autocorr3D(datax,datay,dataz,bins_count,size):
    """Takes x,y,z coordinates, the grid size, the size of the box and
    calculates the autocorrelation"""
    assert(len(datax) == len(datay) and len(datay) == len(dataz))
    datax= datax%size;
    datay= datay%size;
    dataz= dataz%size;
    grid = np.zeros((bins_count,bins_count,bins_count),dtype=np.float32)
    for i in range(0,len(datax)):
        grid[int(datax[i]/size*bins_count),int(datay[i]/size*bins_count),int(dataz[i]/size*bins_count)] +=1
    #grid = remove_avg(grid)
    return autocorr3D_grid(grid,bins_count,size)

def remove_avg(grid):
    grid = grid.astype("<f4")
    grid_avg = np.average(grid)
    return grid-grid_avg

def from_3d_to_1d(grid,n):
    values = np.zeros(n/2,dtype=np.float32)
    weight = np.zeros(n/2,dtype=np.float32)
    max_dist = n/2
    for i in range(0,n/2):
        for j in range(0,n/2):
            for k in range(0,n/2):
                power = grid[i,j,k]
                index_f = np.sqrt(i*i + j*j + k*k)
                index = int (index_f) #bin the power linear into int bins
                fraction = index_f - index
                if(index< n/2):
                    values[index] += power*(1-fraction) 
                    weight[index]+=(1-fraction) #keep track of how many times power is added to a bin
                    if(index+1 < n/2): #don't go out of bounds
                        values[index+1] += power*(fraction)
                        weight[index+1] += fraction
    return values/weight

def autocorr3D_grid(grid,n,domain):
    points_sum = np.sum(grid)
    space_ps = np.abs(np.fft.fftn(grid))
    space_ps *= space_ps
    space_ac = np.fft.ifftn(space_ps).real
    space_dist=np.fft.rfftfreq(n,domain/n)
    print space_ac[0,0,0]
    print points_sum
#    space_ac /= space_ac[0, 0, 0]
    space_ac /=points_sum*points_sum
    space_ac *= n*n*n
    autocorr = from_3d_to_1d(space_ac,n)
    dist = np.arange(0,n/2.0)*domain/float(n)
    return [autocorr-1.0,dist]

def power_spectrum2D(datax,datay,n,domain):
    assert(len(datax) == len(datay))
    grid = np.zeros((n,n),dtype=np.float32)
    for i in range(0,len(datax)):
        grid[int(n*datax[i]/domain),int(n*datay[i]/domain)] +=1
    plt.figure()
    plt.imshow(grid)
    space_ps = np.abs(np.fft.fftn(grid))
    space_ps *= space_ps
    fft_x = np.fft.fftfreq(n,domain/n)
    max_k = max(fft_x)
    plt.figure()
    plt.imshow(space_ps)
    values = np.zeros(n/2,dtype=np.float32)
    weight = np.zeros(n/2,dtype=np.float32)
    k_dist = fft_x*fft_x
    for i in range(0,n/2):
        for j in range(0,n/2):
            power = space_ps[i,j]
            dist2 = k_dist[i] + k_dist[j]
            dist = np.sqrt(dist2)
            index_f = (n/2)*dist/max_k #find the fractional distance
            index = int (index_f) #bin the power linear into int bins
            fraction = index_f - index
            if(index< n/2):
                values[index] += power*(1-fraction) 
                weight[index]+=(1-fraction) #keep track of how many times power is added to a bin
                if(index+1 < n/2): #don't go out of bounds
                    values[index+1] += power*(fraction)
                    weight[index]+= fraction
    return[values/weight,fft_x[:n/2]]
    
def power_spectrum2D(grid,n,domain):
    plt.figure()
    plt.imshow(grid)
    space_ps = np.abs(np.fft.fftn(grid))
    space_ps *= space_ps
    fft_x = np.fft.fftfreq(n,domain/n)
    max_k = max(fft_x)
    plt.figure()
    plt.imshow(space_ps)
    values = np.zeros(n/2,dtype=np.float32)
    weight = np.zeros(n/2,dtype=np.float32)
    k_dist = fft_x*fft_x
    for i in range(0,n/2):
        for j in range(0,n/2):
            power = space_ps[i,j]
            dist2 = k_dist[i] + k_dist[j]
            dist = np.sqrt(dist2)
            index_f = (n/2)*dist/max_k #find the fractional distance
            index = int (index_f) #bin the power linear into int bins
            fraction = index_f - index
            if(index< n/2):
                values[index] += power*(1-fraction) 
                weight[index]+=(1-fraction) #keep track of how many times power is added to a bin
                if(index+1 < n/2): #don't go out of bounds
                    values[index+1] += power*(fraction)
                    weight[index]+= fraction
    return[values/weight,fft_x[:n/2]]
    


##old code from PS_3D
    
#    max_k = max(fft_x)
#    values = np.zeros(n/2,dtype=np.float32)
 #   weight = np.zeros(n/2,dtype=np.float32)
    #k_dist = fft_x*fft_x
 #   for i in range(0,n/2):
 #       for j in range(0,n/2):
  #          for k in range(0,n/2):
   #             power = space_ps[i,j,k]
    #            dist2 = k_dist[i] + k_dist[j] + k_dist[k]
     #           dist = np.sqrt(dist2)
      #          index_f = (n/2)*dist/max_k #find the fractional distance
       #         index = int (index_f) #bin the power linear into int bins
        #        fraction = index_f - index
         #       if(index< n/2):
          #          values[index] += power*(1-fraction) 
           #         weight[index]+=(1-fraction) #keep track of how many times power is added to a bin
            #        if(index+1 < n/2): #don't go out of bounds
             #           values[index+1] += power*(fraction)
              #          weight[index]+= fraction
    #return[values/weight,fft_x[:n/2]]
#def copy_3d(grid,n,domain):
#    space = grid
#    space_ps = np.abs(np.fft.fftn(space))
#    space_ps *= space_ps
#    space_ac = np.fft.ifftn(space_ps).real.round()
