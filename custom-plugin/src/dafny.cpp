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

std::string addAssertionToMethod(const std::string& dafnyCode, const std::string& var, const std::string& assertion) {
    std::string modifiedCode = dafnyCode;
    size_t pos = 0;

    while ((pos = modifiedCode.find(var, pos)) != std::string::npos) {
        size_t methodEnd = modifiedCode.find(";", pos);
        if (methodEnd != std::string::npos) {
            modifiedCode.replace(pos + var.length(), methodEnd - (pos + var.length()), assertion);
            pos = methodEnd + 1;
        }
    }

    return modifiedCode;
}

std::string modifyDafnyCode(const std::string& dafnyCode, const std::string &var, const std::string& assertion) {
    return addAssertionToMethod(dafnyCode, var, assertion);
}