
#include "dtk/timer.hpp"
#include <sstream>

namespace dtk{
  
  Timer::Timer(){}
  Timer::Timer(bool autostart){
    if(autostart)
      start();
  }
  Timer::~Timer(){}
  Timer& Timer::start(){
    gettimeofday(&start_,0);
    return *this;
  }
  Timer& Timer::stop(){
    gettimeofday(&end_,0);
    return *this;
  }
  double const Timer::get_seconds(){
    double result = end_.tv_sec-start_.tv_sec;
    result += (end_.tv_usec-start_.tv_usec)/1e6;
    return result;
  }
  double const Timer::get_mseconds(){    
    double result = (end_.tv_sec-start_.tv_sec)*1e3;
    result += (end_.tv_usec-start_.tv_usec)/1e3;
    return result;    
  }
  double const Timer::get_useconds(){
    double result = (end_.tv_sec-start_.tv_sec)*1e6;
    result += (end_.tv_usec-start_.tv_usec);
    return result;
  }
  
  std::string  Timer::timef() const{
    double sec = end_.tv_sec-start_.tv_sec;
    sec+= (end_.tv_usec-start_.tv_usec)/1e6;
    std::stringstream ss;
    if(sec>3600)
      ss<<sec/3600<<" hr";
    else if(sec>60)
      ss<<sec/60.0<<" min";
    else if(sec>1.0)
      ss<<sec<<" s";
    else if(sec>1e-3)
      ss<<sec*1e3<<" ms";
    else
      ss<<sec*1e6<<" us";
    return ss.str();
  }
  
  AutoTimer::AutoTimer(){
    start();
  }




}
std::ostream& operator<<(std::ostream& os, dtk::Timer const &t){
  os<<t.timef();
  return os;
}
std::ostream& operator<<(std::ostream& os, dtk::AutoTimer &t){
  t.stop();
  os<<t.timef();
  return os;
}
