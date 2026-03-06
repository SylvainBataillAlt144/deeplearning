
#include "tools.h"

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