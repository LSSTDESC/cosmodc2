#ifndef DTK_MATH_HPP
#define DTK_MATH_HPP
#include <cmath>
#include <cstdlib>
namespace dtk{
  
  template<class T,class U>
  T max(T* data, U size){
    T max_val = data[0];
    for(U i =1;i<size;++i){
      if(data[i]>max_val)
	max_val = data[i];
    }
    return max_val;
  }
  template<class T,class U>
  T min(T* data, U size){
    T min_val = data[0];
    for(U i =1;i<size;++i){
      if(data[i]<min_val)
	min_val = data[i];
    }
    return min_val;
  }
  template<class T>
  T min(std::vector<T> data){
    return min(&data[0],data.size());
  }
  template<class T>
  T max(std::vector<T> data){
    return max(&data[0],data.size());
  }

  template<class T,class U>
  T average(T* data, U size){
    T result = 0;
    for(U i =0;i<size;++i){
      result += data[i];
    }
    return result/static_cast<T>(size);
  }
  template<class T>
  T average(std::vector<T> data){
    size_t size = data.size();
    T result = 0;
    for(size_t i =0;i<size;++i){
      result += data[i];
    }
    return result/static_cast<T>(size);

  }
  template<class T> 
  std::vector<T> linspace(T start,T end, int num){
    std::vector<T> result(num);
    T del=0;
    if(num != 0)
     del = (end-start)/num;
    for(int i =0;i<num-1;++i){
      result[i]=start+del*i;
    }
    result[num-1]=end; //to avoid round off errors.
    return result;
  }
  template<class T>
  std::vector<T> logspace(T start,T end, int num){
    std::vector<T> result = linspace(start,end,num);
    for(int i =0;i<result.size();++i){
      result[i] = std::pow(10,result[i]);
    }
    return result;
  }
  
  template<class T>
  T root_mean_squared(std::vector<T> data){
    T result =0;
    for(size_t i =0;i<data.size();++i){
      result+=data[i]*data[i];
    }
    return sqrt(result);
  }
  
  template<class T>
  T rand(){
    return (T) std::rand()/((T)RAND_MAX);
  }
}






#endif
