#ifndef DAFNY_HPP_
#define DAFNY_HPP_
std::string readFile(const std::string& filename);
void writeFile(const std::string& filename, const std::string& content);
std::string addAssertionToMethod(const std::string& dafnyCode, const std::string& var, const std::string& assertion);
std::string modifyDafnyCode(const std::string& dafnyCode, const std::string &var, const std::string& assertion);
#endif