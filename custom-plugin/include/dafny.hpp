#ifndef DAFNY_HPP_
#define DAFNY_HPP_
std::string readFile(const std::string& filename);
void writeFile(const std::string& filename, const std::string& content);
std::string modifyDafnyCode(const std::string& dafnyCode, const std::string &var, const std::string &var_end, const std::string& assertion, 
    bool insert_all);
#endif