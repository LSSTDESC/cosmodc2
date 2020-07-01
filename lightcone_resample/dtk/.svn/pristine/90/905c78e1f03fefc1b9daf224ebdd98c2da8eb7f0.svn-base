
#ifndef DTK_MPI_UTIL_HPP
#define DTK_MPI_UTIL_HPP

#include <mpi.h>

#include "param.hpp"

namespace dtk{
  
  //Note: MPI_Comm ends up as an 'int' in main.o, but as an 
  //ompi_communicator_t* in the dkt/obj/mpi_util.o. So I've 
  // defined this function fully in the head as a template to
  // make my code compile. 
  template<typename mpi_comm>
  void broadcast_param(dtk::Param& param,int root,mpi_comm comm){
    int myrank;
    MPI_Comm_rank(comm,&myrank);
    std::string param_str;
    char* param_cstr=0;
    int   param_cstr_size;
    if(myrank == root){
      param_str = param.stringify();
      param_cstr_size = param_str.size()+1; //+1 for null char
      param_cstr = new char[param_cstr_size];
      strncpy(param_cstr,param_str.c_str(),param_cstr_size);
    }
    MPI_Bcast(&param_cstr_size,1,MPI_INT,root,comm);
    if(myrank != root)
      param_cstr = new char[param_cstr_size];
    MPI_Bcast(param_cstr,param_cstr_size,MPI_CHAR,root,comm);
    if(myrank != root){
      param_str.assign(param_cstr,param_cstr+param_cstr_size);
      param.parse_string(param_str);
    }
    delete [] param_cstr;
  }

  //for some reason, in the header file these MPI_Datatypes
  //are ints when compiled into main.o. But mpi_util.o has these
  //as ompi_datatype_t* which is an interal struct representing 
  //datatypes. I'm changing the MPI_Datatypes return value to int
  //so the I can compile. 
  //Error messages I got when compiling: 
  ///home/dkorytov/proj/core_tracking/core_fit/dtk/dtk/mpi_util.hpp:41: undefined reference to `int dtk::mpi_type<float>()'
  //What was defined in libdtk.a:
  //0000000000000270 T ompi_datatype_t* dtk::mpi_type<float>()
  //template<typename T> ompi_datatype_t* mpi_type();
  //template<> ompi_datatype_t* mpi_type<float>();
  //template<> ompi_datatype_t* mpi_type<double>();
  //template<> ompi_datatype_t* mpi_type<int>();
  //template<> ompi_datatype_t* mpi_type<int64_t>();
  //Didn't work. I guess I'm not writing a wrapper for mpi...



  template<typename T>
  void broadcast_vector(std::vector<T>& v,int bcast_rank, MPI_Comm comm){
    int myrank;
    size_t size;
    MPI_Comm_rank(comm,&myrank);
    if(myrank==bcast_rank)
      size = v.size();
    //MPI_Bcast(&size,1,mpi_type<T>(),bcast_rank,comm);
    if(myrank!=bcast_rank)
      v.resize(size);
    //MPI_Bcast(&v[0],size,mpi_type<T>(),bcast_rank,comm);

  }
  template<typename T>
  void mpi_bcast(T*& data, int size, int root,MPI_Comm comm){
    /*int myrank;
    MPI_Comm_rank(comm,&myrank);
    MPI_Bcast(&size,1,MPI_INT,root,comm);
    if(myrank != root){
      data = new T[size];
    }
    MPI_Bcast(data,size,dtk::mpi_type<T>(),root,comm);*/
  }
  template<typename T>
  void mpi_bcast(std::vector<T>& data, int root, MPI_Comm comm){
    /*int myrank;
    int size = data.size();
    MPI_Comm_rank(comm,&myrank);
    MPI_Bcast(&size,1,MPI_INT,root,comm);
    if(myrank != root){
      data.resize(size);
    }
    MPI_Bcast(&data[0],size,mpi_type<T>(),root,comm);*/
  }
  
}


#endif //DTK_MPI_UTIL_HPPo
