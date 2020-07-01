#ifndef DTK_POWER_SPECTRUM_HPP
#define DTK_POWER_SPECTRUM_HPP
#include <fftw3.h>
namespace dtk{

  struct PowerSpectrum{
    int size;
    float* ps;
    float* k;

  };
  struct Correlation{
    int size;
    float* corr;
    float* r;

  };
  fftwf_complex* gridify     (float* x, float* y, float* z, int count, int n, float domain);
  float*         gridify_real(float* x, float* y, float* z, int count, int n, float domain);
  PowerSpectrum  calc_power_spectrum(float* x, float* y, float* z,int count,int n, float domain);
  PowerSpectrum  calc_power_spectrum(fftwf_complex* grid,                   int n, float domain);
  PowerSpectrum  calc_power_spectrum(float* grid,                           int n, float domain);

  Correlation    calc_correlation(float* x, float* y, float* z, int count, int n, float domain);
  Correlation    calc_correlation(fftwf_complex* grid,                     int n, float domain);
  Correlation    calc_correlation(float* grid,                     int n, float domain);
  Correlation    calc_cross_correlation(float* x1, float* y1, float* z1, int count1,
					float* x2, float* y2, float* z2, int count2,
					int n, float domain);
  Correlation    calc_cross_correlation(fftwf_complex* grid1, 
					fftwf_complex* grid2,
					int n, float domain);
  Correlation    calc_cross_correlation(float* grid1, 
					float* grid2,
					int n, float domain);
  
  Correlation    calc_correlation_n2(float* x, float* y, float* z, int count, float domain,float min_limit);

};

#endif //DTK_POWER_SPECTRUM_HPP
