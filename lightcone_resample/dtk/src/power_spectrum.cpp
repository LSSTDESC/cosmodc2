


#include <math.h>
#include <iostream>

#include "dtk/power_spectrum.hpp"

namespace dtk{

  inline int get_index(int i, int j, int k, int n){
    return i*n*n+j*n+k;
  }
  inline float complex_abs(fftwf_complex& c){
    return (float) c[0]*c[0]+c[1]*c[1];
  }
  inline void grid3d_to_1d(fftwf_complex* grid, int n, float* out,bool for_corr=false,float domain= 1){
    float norm_factor =1.0/((float) n*(float)n*(float)n);
    float weight[n/2+1];
    float spatial_per_k = 1.0/domain;
    for(int i =0;i<n/2+1;++i){
      weight[i] =0;
      out[i]=0;
    }
    for(int i =0;i<n/2+1;++i){
      for(int j=0;j<n/2+1;++j){
	for(int k=0;k<n/2+1;++k){
	  float r = sqrt(static_cast<float>(i*i+j*j+k*k)); //radial k index
	  int r_index = static_cast<int>(r);
	  float power = complex_abs(grid[get_index(i,j,k,n)])*norm_factor*norm_factor*4; //Should be 8, but 4 gives correct result
	  float fract = r-r_index;
	  if(for_corr){
	    power/=(r*spatial_per_k);
	  }
	  if(r_index < n/2+1){
	    out[r_index] +=(1.0-fract)*power;
	     weight[r_index]+=(1.0-fract);
	    if(r_index+1 < n/2+1 && fract !=0){
	      out[r_index+1] += (fract)*power;
	      weight[r_index+1]+=(fract);
	      //	      if(r<33 && r>31)
	      //std::cout<<"r: "<<r<<"\tr_index: "<<r_index<<"\tfract: "<<fract<<std::endl;
	      
	    }
	  }
	}
      }
    }
    for(int i =0;i<n/2+1;++i){
      if(weight[i] != 0)
	out[i]/=weight[i];
      else
	out[i] =0;
    }
  }
  
  inline void add_k_power(fftwf_complex* grid, int n,float domain){ //grows each mode by a factor of k
    float k_per_r = ((float)n)/domain;
    for(int i =0;i<n/2+1;++i)
      for(int j=0;j<n/2+1;++j)
	for(int k=0;k<n/2+1;++k){
	  float r = sqrt(static_cast<float>(i*i+j*j+k*k)); //radial k index
	  int index= get_index(i,j,k,n);
	  grid[index][0]*=r*k_per_r;
	  grid[index][1]*=r*k_per_r;
	}
  }
  inline void complex_squared(fftwf_complex* a, fftwf_complex* b, int n){
    float norm = 1.0/((float)(n*n*n));
    for(int i =0;i<n*n*n;++i){
      float real = a[i][0]*b[i][0]+a[i][1]*b[i][1];
      float img  = a[i][0]*b[i][1]+a[i][1]*b[i][0];
      a[i][0] = real*norm;
      a[i][1] = img*norm;
    }
  }

  fftwf_complex* gridify(float* x, float* y, float* z, int count, int n, float domain){
    fftwf_complex* grid = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * n*n*n);
    int i,j,k, index;
    for(int m =0;m<count;++m){
      i = x[m]/domain*n;
      j = y[m]/domain*n;
      k = z[m]/domain*n;
      index = get_index(i,j,k,n);
      grid[index][0]+=1;
    }
    return grid;
  }
  float* gridify_real(float* x, float* y, float* z, int count, int n, float domain){
    float* grid = new float[n*n*n];
    int i,j,k, index;
    for(int m =0;m<count;++m){
      i = x[m]/domain*n;
      j = y[m]/domain*n;
      k = z[m]/domain*n;
      index = get_index(i,j,k,n);
      grid[index]+=1;
    }
    return grid;
  }
  
  PowerSpectrum calc_power_spectrum(float* x, float* y, float* z,int count,int n, float domain){
    //    fftwf_complex* grid = gridify(x,y,z,count,n,domain);
    float* grid = gridify_real(x,y,z,count,n,domain);
    PowerSpectrum result = calc_power_spectrum(grid,n,domain); 
    delete [] grid;
    return result;
  }
  
  
  PowerSpectrum calc_power_spectrum(fftwf_complex* grid, int n, float domain){
    PowerSpectrum ps;
    int m = n/2+1;
    fftwf_complex* out = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * n*n*n);
    fftwf_plan p = fftwf_plan_dft_3d( n,  n,  n,
				    grid,  out,
    				    1, FFTW_ESTIMATE|FFTW_DESTROY_INPUT );
    fftwf_execute(p);
    ps.size = m;
    ps.ps   = new float[m];
    ps.k    = new float[m];

    for(int i =0;i<m;++i)
      ps.k[i]= ((float)(i)/domain);
    grid3d_to_1d(out,n,ps.ps);
    fftwf_destroy_plan(p);
    fftwf_free(out);
    return ps;
  }
  PowerSpectrum calc_power_spectrum(float* grid, int n, float domain){
    std::cout<<"we are trying to plan with pthreads"<<std::endl;
    //    fftw_init_threads();
    //fftw_plan_with_nthreads(8);
    PowerSpectrum ps;
    int m = n/2+1;
    fftwf_complex* out = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * n*n*n);
    fftwf_plan p = fftwf_plan_dft_r2c_3d( n, n, n,
					grid,  out,
					FFTW_ESTIMATE|FFTW_DESTROY_INPUT );
    fftwf_execute(p);
    ps.size = m;
    ps.ps   = new float[m];
    ps.k    = new float[m];

    for(int i =0;i<m;++i)
      ps.k[i]= ((float)(i)/domain);
    grid3d_to_1d(out,n,ps.ps);
    fftwf_destroy_plan(p);
    fftwf_free(out);
    //fftw_cleanup_threads();
    return ps;
  }

  Correlation    calc_correlatio(float* x, float* y, float* z, int count, int n, float domain){
    fftwf_complex* grid = gridify(x,y,z,count,n,domain);
    Correlation result = calc_correlation(grid,n,domain);
    delete [] grid;
    return result;
  }
  Correlation    calc_cross_correlation(float* x1, float* y1, float* z1, int count1, 
					float* x2, float* y2, float* z2, int count2, 
					int n, float domain){
    fftwf_complex* grid1 = gridify(x1,y1,z1,count1,n,domain);
    fftwf_complex* grid2 = gridify(x2,y2,z2,count2,n,domain);
    Correlation result = calc_cross_correlation(grid1,grid2,n,domain);
    delete [] grid1; 
    delete [] grid2;
    return result;
  }
  Correlation    calc_correlation(fftwf_complex* grid,                     int n, float domain){
    Correlation corr;
    int m = n/2+1;
    std::cout<<"calcing corr n: "<<n<<" domain: "<<domain<<" ..."<<std::endl;
    fftwf_complex* out = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * n*n*n);
    fftwf_plan p1 = fftwf_plan_dft_3d( n,  n,  n,
				    grid,  out,
    				    1, FFTW_ESTIMATE|FFTW_DESTROY_INPUT );
    fftwf_execute(p1);
    complex_squared(out,out,n);
    //add_k_power(out,n,domain);
    fftwf_plan p2 = fftwf_plan_dft_3d( n,  n,  n,
				      out,  grid, //note: grid is overwritten and used as the output
    				    -1, FFTW_ESTIMATE|FFTW_DESTROY_INPUT );
    fftwf_execute(p2);
    
    corr.size = m;
    corr.corr = new float[m];
    corr.r    = new float[m];
    for(int i =0;i<m;++i)
      corr.r[i]= domain*((float)i)/((float)n);
    grid3d_to_1d(grid,n,corr.corr,true,domain);
    fftwf_destroy_plan(p1);
    fftwf_destroy_plan(p2);
    fftwf_free(out);
    return corr;

  }
  Correlation calc_cross_correlation(fftwf_complex* grid1, fftwf_complex* grid2, int n, float domain){
    
    Correlation corr;
    int m = n/2+1;
    fftwf_complex* out1= (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * n*n*n);
    fftwf_complex* out2= (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * n*n*n);
    fftwf_plan p1 = fftwf_plan_dft_3d( n,  n,  n,
				       grid1, out1,
				       1, FFTW_ESTIMATE|FFTW_DESTROY_INPUT );

    fftwf_plan p2 = fftwf_plan_dft_3d( n,  n,  n,
				       grid2, out2,
				       1, FFTW_ESTIMATE|FFTW_DESTROY_INPUT );
    
    fftwf_execute(p1);
    fftwf_execute(p2);
    complex_squared(out1,out2,n);
    fftwf_plan p3 = fftwf_plan_dft_3d( n,  n,  n,
				      out1,  grid1, //note: grid is overwritten and used as the output
    				    -1, FFTW_ESTIMATE|FFTW_DESTROY_INPUT );
    fftwf_execute(p3);
    
    corr.size = m;
    corr.corr = new float[m];
    corr.r    = new float[m];
    for(int i =0;i<m;++i)
      corr.r[i]= domain*((float)i)/((float)n);    
    grid3d_to_1d(grid1,n,corr.corr);
    fftwf_destroy_plan(p1);
    fftwf_destroy_plan(p2);
    fftwf_destroy_plan(p3);
    fftwf_free(out1);
    fftwf_free(out2);
    return corr;


  }

  
  int bin_length(float r2,float min2, float max2){


  }
  Correlation calc_correlation_n2(float* x, float* y, float* z,
				  int count, int n, float domain, float min_limit){
    Correlation corr_result;
    float min2 = min_limit*min_limit;
    float max  = domain*domain;
    float* corr = new float[n];
    float x1,y1,z1;
    float x2,y2,z2;
    for(int n1 =0;n1<n;++n1)
      for(int n2=0;n2<n;++n){
	x1=x[n1];
	y1=y[n1];
	z1=z[n1];
	for(int ii=-1;ii<2;++ii)
	  for(int jj=-1;jj<2;++jj)
	    for(int kk=-1;kk<2;++kk){
	      x2=x[n2]+ii*domain-x1;
	      y2=y[n2]+jj*domain-y1;
	      z2=z[n2]+kk*domain-z1;
	      float r2 = x2*x2+y2*y2+z2*z2;
	    }
      }
  
    return corr_result;
  }
}
