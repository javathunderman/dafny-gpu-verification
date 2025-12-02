#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>

std::string readFile(const std::string& filename) {
    std::ifstream file(filename);
    std::ostringstream ss;
    if (file) {
        ss << file.rdbuf();
    } else {
        std::cerr << "Error reading file: " << filename << std::endl;
    }
    return ss.str();
}

void writeFile(const std::string& filename, const std::string& content) {
    std::ofstream file(filename);
    if (file) {
        file << content;
    } else {
        std::cerr << "Error writing to file: " << filename << std::endl;
    }
}


std::string modifyDafnyCode(const std::string& dafnyCode, const std::string &var, const std::string &var_end, const std::string& assertion, bool insert_all) {
    std::string modifiedCode = dafnyCode;
    size_t pos = 0;
    if (insert_all) {
        while ((pos = modifiedCode.find(var, pos)) != std::string::npos) {
            size_t methodEnd = modifiedCode.find(var_end, pos);
            // std::cout << "\nmethodEnd " << methodEnd << std::endl;
            if (methodEnd != std::string::npos) {
                modifiedCode.replace(pos + var.length(), methodEnd - (pos + var.length()), assertion);
                pos = methodEnd + 1;
            }
        }
    } else {
        pos = modifiedCode.find(var, 0);
        size_t pos2 = modifiedCode.find(var, pos + 1);
        if (pos2 != std::string::npos) {
            size_t methodEnd = modifiedCode.find(var_end, pos2);
            if (methodEnd != std::string::npos) {
                modifiedCode.insert(methodEnd, assertion + "\n");
            }
        }
        
    }
    return modifiedCode;
}