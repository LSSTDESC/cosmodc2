#ifndef DTK_COSMO_HPP
#define DTK_COSMO_HPP

#include <iostream>
namespace dtk{
  template <typename T> inline
  T z_from_a(T a){
    return 1.0/a -1.0;
  }
  
  template <typename T> inline
  T a_from_z(T z){
    return 1.0/(z+1.0);
  }
  
  class StepRedshift{
    float z_in,z_out,a_in,a_out;
    int num_steps;
    float a_del;

  public:
    StepRedshift(float z_start,float z_end,int num_steps);
    float step_from_z(float z);
    float z_from_step(float step);
  };
}
#endif // DTK_COSMO_HPP
