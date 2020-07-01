
#include <sstream>

#include "dtk/util.hpp"


namespace dtk{
  std::string cat_strings(std::vector<std::string> strings){
    std::stringstream ss;
    for(uint i =0;i<strings.size();++i){
      ss<<strings[i];
    }
    return ss.str();
  }


  bool ensure_dir(std::string dir){
#ifdef unix
    std::stringstream ss;
    ss<<"mkdir -p "<<dir;
    int ret = system(ss.str().c_str());
    return ret==0;
#else 
    return false;
#endif //unix
  }

  bool ensure_file_path(std::string file){
    std::cout<<"we are trying to get the path for file: \""<<file<<"\""<<std::endl;
    size_t end = file.find_last_of("\\/");
    std::cout<<end<<std::endl;
    if(end == std::string::npos)
      return false;
    return ensure_dir(file.substr(0,end));
  }
  void pause(){
    std::cout<<"Press enter to continue...";
    std::string t;
    std::getline(std::cin,t);
    //    std::getchar();
  }
  bool ask_continue(){
    std::cout<<"Coninue? [y/yes/enter or n/no]: ";
    return ask_bool();
  }
  bool ask_bool(){
    bool done = true;
    bool result;
    do{
      std::string t;
      std::getline(std::cin,t);
      if(t=="y" || t=="yes" || t==""){
	result = true;
	done = true;
      }
      else if(t=="n" || t=="no"){
	result = false;
	done = true;
      }
      else{
	std::cout<<"Input \""<<t<<"\" not understood. [y/yes/enter or n/no]: ";
	done = false;
      }
    }while(!done);
    return result;
  }
}
