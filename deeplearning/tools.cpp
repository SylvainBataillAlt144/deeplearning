
#include "tools.h"

#include <iostream>
#include <sstream>
#include <iomanip>

int64_t GetFileSize(std::ifstream& Fichier)
{
    // sauvegarder la position courante 
    int64_t pos = Fichier.tellg();
    // se placer en fin de fichier 
    Fichier.seekg(0, std::ios_base::end);
    // récupérer la nouvelle position = la taille du fichier 
    int64_t size = Fichier.tellg();
    // restaurer la position initiale du fichier 
    Fichier.seekg(pos, std::ios_base::beg);
    return size;
}

std::string intToStringWithZeros(int number, uint8_t width) {
    std::ostringstream oss;
    oss << std::setw(width) << std::setfill('0') << number;
    return oss.str();
}