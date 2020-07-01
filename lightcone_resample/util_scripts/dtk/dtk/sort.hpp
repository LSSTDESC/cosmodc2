#ifndef DTK_SORT_HPP
#define DTK_SORT_HPP

#include <vector>
#include <cstddef>
#include <stdint.h>
#include <algorithm>
#include <stdio.h>
#include <string.h>

namespace dtk{

  template <class T>
  class Sorter; // With std::sort, it is able to sort an array on indexes based
  // on value array. Using the sorted indexes, the value array will be in
  // increasing order.
  
  template <class T,class U>
  class SubSorter; // Similar to the Sorter, but it uses two value array to
  // sort. If first array has equal values, it will look at the second array
  // to figureout the order.

  template <class T>
  class SortedIndex; // This class wraps a sorted index so that you don't have
  // go through two arrays when using a sorted index (a[srt[i]] -> a[SortedIndex]).
  // The sorted index acts like a integer and when incremented/decremented it's
  // value changes depending on the sorted index given.

  template <class T>
  class MultiSort; //This class is able to sort multiple sets of associated 
  // arrays (x,y,z, etc) by a specific array. The sorting is done in place. 

  template <class T, class U>
  void reorder(T* data, int size, const U* srt);
  template <class T, class U>
  void reorder(T* data, int64_t size, const U* srt);
  template <class T, class U>
  void reorder(T* data, size_t size, const U* srt);

  template <class T, class U>
  void reorder(std::vector<T>& data,const U* srt); // Takes an array/vector and reorders 
  // it out of place using a sorted index. Template U must be an int or int64_t.

  template <class T>
  T* new_sort_array(int64_t size); // Returns an array of size 'size' with values
  // increasing from 0 to size-1. It's meant to be used to quickly make an index
  // array for sorted. Template T should be int/int64_t for index arrays.

  template <class T,class U>
  T* arg_sort(U* data, T size);

  template <class T>
  class Sorter{
    T* src;
  public:
    Sorter( T* given): src(given){}
    Sorter(std::vector<T> &given):src(&given[0]){}
    bool operator()(int i, int j){
      return src[i]<src[j];
    }
  };
  
  //not tested yet
  template <class T, class U>
  class SubSorter{
    T* src1; //main sort array
    U* src2; //secondary sort array
  public:
    SubSorter(T* src1, U* src2): src1(src1),src2(src2){}
    SubSorter(std::vector<T> &given1, std::vector<U> &given2){
      src1=&given1[0];
      src2=&given2[0];
    }
    bool operator()(int i, int j){
      if(src1[i]<src1[j])
	return true;
      else if(src1[i] == src1[j])
	return src2[i] < src2[j];
      else
	return false;
    }
  };

  template <class T = int64_t>
  class SortedIndex{
    const T* srt;
    T size;
    T i;
  public:
    SortedIndex(): srt(NULL),size(0),i(0){}
    SortedIndex(const T* srt,T size): srt(srt),size(size),i(0){}
    SortedIndex(const SortedIndex<T>& ref):srt(ref.srt),size(ref.size),i(ref.i){}
    SortedIndex(const std::vector<T> srt):srt(&srt[0]),size(srt.size()),i(0){}
    bool good(){
      return i<size && i>=0;
    }
    bool last(){
      return i+1 == size;
    }
    T remaining(){
      return size-i;
    }
    T get_index(){
      return i;
    }
    T get_size(){
      return size;
    }
    void reset(){
      i=0;
    }
    void set(T* srt,T size){
      this->srt = srt;
      this->size = size;
      i = 0;
    }
    operator T(){
      return srt[i];
    }
    T operator [](T j){
      return srt[j];
    }
    SortedIndex& operator++(){
      ++i;
      return *this;
    }
    SortedIndex operator++(int){
      SortedIndex<T> tmp(*this);
      ++*this;
      return tmp;
    }
    SortedIndex& operator--(){
      --i;
      return *this;
    }
    SortedIndex operator--(int){
      SortedIndex<T> tmp(*this);
      --*this;
      return tmp;
    }
    SortedIndex& operator=(const SortedIndex& ref){
      srt = ref.srt;
      size = ref.size;
      i = ref.i;
      return *this;
    }
  };


  
  template<class T>
  class MultiSort{
    struct Data{
      char* data_;
      int   element_size_;
    };
    T* data_;
    int64_t data_size_;
    std::vector<Data> extra_data_; //to be sorted in the same order
    // as data_ will be.
  public:

    MultiSort():data_(0),data_size_(0){}

    ~MultiSort(){}

    //    void set_size(int64_t size){ data_size_ = size;}

    void add_sort_data(T* data,int64_t size){
      data_ = data;
      data_size_ = size;
    }
    template <class U>
    void add_data(U* data){
      Data ex_data;
      ex_data.data_ = (char*) (data);
      ex_data.element_size_ = sizeof(U)/sizeof(char);
      extra_data_.push_back(ex_data);
      std::cout<<"element size: "<<ex_data.element_size_<<std::endl;
    }

  private:
  

    inline void swap_individual(int64_t i, int64_t j, Data& data){
      int elem_size = data.element_size_;
      char tmp[elem_size];
      memcpy(tmp,data.data_+(i*elem_size),elem_size);
      memcpy(data.data_+(i*elem_size),data.data_+(j*elem_size),elem_size);
      memcpy(data.data_+(j*elem_size),tmp,elem_size);
      //      std::cout<<"\tswapping "<<i<<" "<<j<<std::endl;
    }
    inline void swap(int64_t i, int64_t j){
      if(i==j)
	return;
      // std::cout<<"swapping "<<i<<" "<<j<<std::endl;
      T tmp = data_[i];
      data_[i] = data_[j];
      data_[j] = tmp;
      for(uint k =0;k<extra_data_.size();++k)
	swap_individual(i,j,extra_data_[k]);
    }
    void sort(int64_t i, int64_t size){
       if(size<=2){
	int other = size/2;
	if(data_[i]>data_[i+other])
	  swap(i,i+other);
      }
      else{
	int k = 1;
	for(int j =1;j<size;++j){
	  if(data_[i+j]< data_[i+0]){
	    swap(i+j,i+k);
	    ++k;
	  }
	}
	swap(i,i+k-1);
	sort(i,k);
	sort(i+k,size-k);
      }  
    }
  public:
    void sort(){
      sort(0,data_size_);
    }
  };

 template<class T,class U>
  void reorder(T* data, int64_t size, const U* srt){
     T* tmp = new T[size];
    for(U i=0;i<size;++i){
      tmp[i]=data[srt[i]];
    }
    memcpy(data,tmp,size*sizeof(T));
    delete [] tmp;
  }
  template<class T,class U>
  void reorder(std::vector<T>& data,const U* srt){
    reorder<T,U>(&data[0],(int64_t)data.size(),srt);
  }
 
  template<class T,class U>
  void reorder(T* data, int size, const U* srt){
     reorder(data,(int64_t)(size),srt);
  }
  template <class T, class U>
  void reorder(T* data, size_t size, const U* srt){
    reorder(data,(int64_t)(size),srt);
  }

  template<class T,class U>
  void reorder(T* data, const U* srt, int64_t size){
     reorder(data,size,srt);
  }
  template<class T>
  T* new_sort_array(int64_t size){
    T* result = new T[size];
    for(int64_t i=0;i<size;++i)
      result[i]=i;
    return result;

  }  
  template <class T>
  void group_by_id(T* groups_id,
		   size_t size,
		   std::vector<size_t>& group_start,
		   std::vector<size_t>& group_size){
    if(size==0)
      return;
      T current= groups_id[0];
      size_t start=0;
    for(size_t i=0;i<size;++i){
        if(groups_id[i] == current){
	continue;
      }
      else{
	group_start.push_back(start);
	group_size.push_back(i-start);
	start=i;
	current=groups_id[i];
      }
    }
    if(size-start != 0){
      group_start.push_back(start);
      group_size.push_back(size-start);
    }
  }

  template <class T>
  void group_by_id(std::vector<T> groups_id,
		   std::vector<size_t>& group_start,
		   std::vector<size_t>& group_size){
    group_by_id<T>(&groups_id[0],groups_id.size(),group_start,group_size);
  }

  template <class T,class U>
  T* arg_sort(U* data, T size){
    std::pair<U,T>* pairs = new std::pair<U,T>[size];
    for(T i =0;i<size;++i){
      pairs[i].first = data[i];
      pairs[i].second = i;
    }
    std::sort(pairs,pairs+size);
    T* srt = new T[size];
    for(T i =0;i<size;++i)
      srt[i] = pairs[i].second;
    delete [] pairs;
    return srt;
  }
  template <class T, class U>
  dtk::SortedIndex<U> sort_by(T* data, U*& srt_array, U size){
    srt_array = dtk::new_sort_array<U>(size);
    std::sort(srt_array,srt_array+size,dtk::Sorter<T>(data));
    dtk::SortedIndex<U> si;
    si.set(srt_array,size);
    return si;
  }
  
  template<class T>
  int64_t find(T* data,T target,int64_t size){
    T* pos = std::find(data,data+size,target);
    if(pos == data+size)
      return -1;
    else
      return pos-data;
  }
  template<class T>
  size_t max_index(T* data, size_t size){
    T max = data[0];
    size_t max_indx = 0;
    for(size_t i=1;i<size;++i){
      if(data[i]>max){
	max = data[i];
	max_indx = i;
      }
    }
    return max_indx;
  }
  template<class T>
  size_t min_index(T* data, size_t size){
    T min = data[0];
    size_t min_indx = 0;
    for(size_t i=1;i<size;++i){
      if(data[i]<min){
	min = data[i];
	min_indx = i;
      }
    }
    return min_indx;
  }

  template<typename T>
  int64_t  find_bin(const std::vector<T> bins, T data){
    int64_t indx = std::lower_bound(bins.begin(),bins.end(),data)-bins.begin()-1;
    if(indx==bins.size()-1 || bins[0]>data) //if the indx is greater than the allowed amount, then
      indx = -1;//return -1 to indicate it's not in the range. 
    return indx;
  }
  template<typename T,typename U>
  int64_t  find_bin(const T* bins, U size, T data){
    int64_t indx = std::lower_bound(bins,bins+size,data)-bins.begin()-1;
    if(indx==bins.size()-1 || bins[0]>data) //if the indx is greater than the allowed amount, then
      indx = -1;//return -1 to indicate it's not in the range. 
    return indx;
  }
  
  template<typename T>
  int64_t search_sorted(T* data,T target,size_t size){
    T* result = std::lower_bound(data,data+size,target);
    if(result != data+size && result[0]==target)
      return result-data;
    else
      return -1;
  }
  template<typename T>
  int64_t search_sorted2(T* data,T target, int64_t* srt, size_t size){
    int64_t* result = std::lower_bound(srt,srt+size,target,Sorter<T>(data));
    std::cout<<"result i "<<result-data<<std::endl;
    std::cout<<"target: "<<target<<std::endl;
    std::cout<<"we got: "<<data[result[0]]<<std::endl;
    if(result != srt+size && data[srt[result[0]]] == target){
      return result[0];
    }
    else
      return -1;
  }
  template<typename T>
  int64_t search_sorted(T* data,T target, int64_t* srt, size_t size){
    int start,mid,end;
    start = 0;
    end = size;
    mid = size/2;
    //    std::cout<<"\n\n"<<std::endl;
    do{
      // std::cout<<"start: "<<data[srt[start]]<<std::endl;
      // std::cout<<"mid:   "<<data[srt[mid]]<<std::endl;
      // std::cout<<"target:"<<target<<std::endl;
      // std::cout<<"end:   "<<data[srt[end-1]]<<std::endl;
      if(data[srt[mid]]<target){
	start = mid;
      }
      else{
	end = mid+1;
      }
      mid = (start+end)/2;
      //std::cout<<"len: "<<end-start<<std::endl;

    }while(end-start > 10);

    for(int i =start;i<end;++i){
      //std::cout<<data[srt[i]]<<std::endl;
      if(data[srt[i]]==target){
	return srt[i];
      }
    }
    return -1;
  }
  
}


#endif //SORT_HPP 
