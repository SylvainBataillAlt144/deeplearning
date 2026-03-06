#pragma once

#include <vector>
#include<string>

#include <fstream>
#include <iostream>
#include <iomanip>
#include <stdexcept>

#include "Matrix.h"

class Serialize
{
private:   
    std::ifstream iFile;
    std::ofstream oFile;

public:
template<typename T>
bool Load(const std::string name, T& val)
{
    iFile.exceptions(iFile.exceptions() | std::ios::failbit);
    iFile.open(name, std::ios::in);
    if (iFile.is_open()) {
        iFile >> val;
        iFile.close();
        return true;
    }
    return false;
}

template<typename T>
bool Save(const std::string name, const T& val, bool wo = true)
{
    oFile.exceptions(oFile.exceptions() | std::ios::failbit);
    if (wo) {
        oFile.open(name, std::ios::out);
    }
    else {
        oFile.open(name, std::ios::out | std::ios::app);
    }
    if (oFile.is_open()) {
        oFile << val;
        oFile.close();
        return true;
    }
    return false;
}

};


template <typename T>
std::ostream& operator <<(std::ostream& os, const std::vector<T>& data) {
    const auto default_precision = std::cout.precision();
    const auto max_precision = std::numeric_limits<T>::digits10 + 1;
    os << "[" << data.size() << "] { ";
    for (std::size_t i = 0; i < data.size(); i++) {
        if (i > 0) {
            os << ",";
        }
        os << std::setprecision(max_precision) << data[i];
    }
    os << "}" << std::endl;
    os << std::setprecision(default_precision);
    return os;
}

template<typename T>
std::istream& operator >>(std::istream& is, std::vector<T>& data) {
    data.clear();
    char brace, comma;
    std::size_t s = 0;
    is >> brace;
    is >> s;
    is >> brace;
    is >> brace;
    for (std::size_t i = 0; i < s; i++) {
        T val;
        is >> val >> comma;
        data.push_back(val);
    }
    return is;
}
