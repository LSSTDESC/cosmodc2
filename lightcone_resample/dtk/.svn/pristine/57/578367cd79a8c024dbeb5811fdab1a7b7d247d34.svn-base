
#include "dtk/hdf5_util.hpp"


namespace dtk{
  template <> H5::PredType hdf5_type<float>(){
    return H5::PredType::NATIVE_FLOAT;
  }
  template <> H5::PredType hdf5_type<double>(){
    return H5::PredType::NATIVE_DOUBLE;
  }
  template <> H5::PredType hdf5_type<int>(){
    return H5::PredType::NATIVE_INT;
  }
  template <> H5::PredType hdf5_type<int64_t>(){
    return H5::PredType::NATIVE_INT64;
  }
  template <> H5::PredType hdf5_type<uint64_t>(){
    return H5::PredType::NATIVE_UINT64;
  }
  template <> H5::PredType hdf5_type<unsigned int>(){
    return H5::PredType::NATIVE_UINT;
  }
  template <> H5::PredType hdf5_type<bool>(){
    return H5::PredType::NATIVE_HBOOL;
  }

  void hdf5_clear_file(std::string file_loc){
    H5::H5File file(file_loc,H5F_ACC_TRUNC);
    file.close();
  }

  void hdf5_ensure_variable_path(H5::H5File& file, std::string path){
    //std::cout<<"ensuring the variable path to: "<<path<<std::endl;
    size_t substr_size = path.find_last_of("/");
    H5::Group g=file.openGroup("/");
    if(path.find_first_of("/")==0){
      path = path.substr(1,substr_size);
      //std::cout<<"knocked off the starting /"<<std::endl;
    }
    hdf5_ensure_group(g,path.substr(0,substr_size));
  }
  void hdf5_ensure_group(H5::Group& group, std::string path){
    //std::cout<<"working on :"<<path<<std::endl;
    while(path != ""){
      size_t substr_end = path.find_first_of("/");

      std::string current_group;
      if( substr_end != std::string::npos){
	//std::cout<<"can split"<<std::endl;
	current_group= path.substr(0,substr_end);
	path = path.substr(substr_end+1,path.size());
      }
      else{
	//std::cout<<"cannot split"<<std::endl;
	current_group == path;
	path = "";
      }
      //std::cout<<"figuring out: "<<current_group<<std::endl;
      //std::cout<<"saving for later: "<<path<<std::endl;
      hsize_t num = group.getNumObjs();
      //std::cout<<"num: "<<num<<std::endl;
      bool found = false;
      for(int i =0;i<num;++i){
	//std::cout<<"i: "<<group.getObjnameByIdx(i)<<std::endl;
	H5G_obj_t type = group.getObjTypeByIdx(i);
	H5std_string name = group.getObjnameByIdx(i);
	if(name == current_group){
	  if(type == H5G_GROUP){
	    //std::cout<<"\ta group"<<std::endl;
	    group = group.openGroup(current_group);
	    found = true;
	    break;
	  }
	  else if(type == H5G_DATASET){
	    //std::cout<<"\ta dataset this shouldn't be"<<std::endl;
	    std::cout<<"This group (hdf5 internal folder) is actually a dataset. We can't convert"
		     <<" a dataset into a group. Exiting (uncleanly)..."<<std::endl;
	    throw;
	  }
	}
      }
      if(not found){
	//std::cout<<current_group<<" not found"<<"creating group..."<<std::endl;
	group = group.createGroup(current_group);

      }
    }
    return;
    size_t substr_start = path.find_first_of("/");
    std::string current_group = path.substr(0,substr_start);
    std::string remaining_group = path.substr(substr_start,path.size());
    //std::cout<<"trying to see if this group exists:" <<current_group<<std::endl;
    //    std::cout<<"\twill build rest later: "<<remaining_group<<std::endl;
    hsize_t num = group.getNumObjs();
    //std::cout<<"num: "<<num<<std::endl;
    for(int i =0;i<num;++i){
      //std::cout<<"i: "<<group.getObjnameByIdx(i)<<std::endl;
      H5G_obj_t type = group.getObjTypeByIdx(i);
      H5std_string name = group.getObjnameByIdx(i);
      if(name == current_group){
	if(type == H5G_GROUP){
	  // std::cout<<"\ta group"<<std::endl;
	  H5::Group g = group.openGroup(current_group);
	  hdf5_ensure_group(g,remaining_group);
	}
	else if(type == H5G_DATASET){
	  std::cout<<"\ta dataset this shouldn't be"<<std::endl;
	  throw;
	}
      }
    }
    //The group isn't here
    //std::cout<<"the group isn't there -> creating it now"<<std::endl;
    std::cout<<current_group<<std::endl;
    H5::Group g = group.createGroup(current_group);
    hdf5_ensure_group(g,remaining_group);
    
    /*    H5::Exception::dontPrint();
    if(path=="" or path == "/"){
      std::cout<<"\tthe root directory already exists"<<std::endl;
    }
    else{
      try{
	std::cout<<"\ttying to open the path"<<std::endl;
	//file.createGroup(path);
	file.openDataSet(path);
      }catch(H5::Exception e){
	std::cout<<"\t"<<e.getDetailMsg()<<std::endl;
	std::cout<<"\tFailed to open the dataset"<<std::endl;
	std::cout<<"\tthe path does not exist, need to create it & subgroups"<<std::endl;
	size_t substr_size = path.find_last_of("/");
	std::cout<<"\t"<<path.substr(0,substr_size)<<std::endl;
	hdf5_ensure_group(file,path.substr(0,substr_size));
	std::cout<<"creating the group now"<<std::endl;
	file.createGroup(path);
	std::cout<<"all doen"<<std::endl;
      }
    }
    }*/
  }

  size_t hdf5_num_elems(H5::H5File hfile,std::string var_name){
    H5::DataSet dataset = hfile.openDataSet(var_name);
    H5::DataSpace ds = dataset.getSpace();
    hsize_t datasize = ds.getSimpleExtentNpoints();
    return datasize;
  }
  size_t hdf5_num_elems(std::string file_name,std::string var_name){
    hdf5_num_elems(H5::H5File(file_name,H5F_ACC_RDONLY),var_name);
  }

}
