

#include <stdint.h>
#define GENERICIO_NO_MPI
#include "GenericIO.h"
#include <sstream>

template <class T>
void read_gio(char* file_name, std::string var_name, T*& data,int only_rank=-1){
  //  std::cout<<"reading gio template"<<std::endl;
  gio::GenericIO reader(file_name);
  reader.openAndReadHeader(gio::GenericIO::MismatchAllowed);
  int num_ranks = reader.readNRanks();
  uint64_t max_size = 0;
  uint64_t rank_size[num_ranks];
  int start =0;
  // std::cout<<"num_ranks "<<num_ranks<<std::endl;
  // std::cout<<"only_rank "<<only_rank<<std::endl;
  for(int i =0;i<num_ranks;++i){
    if(only_rank==-1){
      // std::cout<<"Reading all data"<<std::endl;
      rank_size[i] = reader.readNumElems(i);
    }
    else{
      if(only_rank == i)
	rank_size[i] = reader.readNumElems(i);
      else
	rank_size[i] = 0;
    }
    if(max_size < rank_size[i])
      max_size = rank_size[i];
  }
  T* rank_data = new T[max_size+reader.requestedExtraSpace()/sizeof(T)];
  int64_t offset =0;
  reader.addVariable(var_name,rank_data,true);
  for(int i=0;i<num_ranks;++i){
    // std::cout<<"reading rank"<<i<<std::endl;
    if(only_rank == -1 || only_rank == i){
      // std::cout<<"read"<<std::endl;
      reader.readData(i,false);
      std::copy(rank_data,rank_data+rank_size[i],data+offset);
      offset +=rank_size[i];
    }
  }
  delete [] rank_data;
  reader.close();
}
extern "C" int64_t get_elem_num(char* file_name);

extern "C" void read_gio_float (char* file_name, char* var_name, float*   data, int only_rank);
extern "C" void read_gio_double(char* file_name, char* var_name, double*  data, int only_rank);
extern "C" void read_gio_int8  (char* file_name, char* var_name, int8_t*  data, int only_rank); 
extern "C" void read_gio_int32 (char* file_name, char* var_name, int*     data, int only_rank); 
extern "C" void read_gio_int64 (char* file_name, char* var_name, int64_t* data, int only_rank);
extern "C" void read_gio_uint8 (char* file_name, char* var_name, uint8_t* data, int only_rank); 
enum var_type{
  float_type=0,
  double_type=1,
  int32_type=2,
  int64_type=3,
  int8_type=4,
  uint8_type=5,
  uint32_type=6,
  uint64_type=7,
  type_not_found=9,
  var_not_found=10
};
extern "C" var_type get_variable_type(char* file_name,char* var_name);
extern "C" void inspect_gio(char* file_name);
