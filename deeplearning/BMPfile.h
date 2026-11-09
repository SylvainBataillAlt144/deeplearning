#pragma once

#include <cstdint>
#include <string>
#include <vector>

#pragma pack(push, 1) // Empêche l'alignement mémoire automatique

// En-tête BMP (14 octets)
struct BMPFileHeader {
    uint16_t bfType;      // Signature "BM"
    uint32_t bfSize;      // Taille totale du fichier
    uint16_t bfReserved1; // Réservé (0)
    uint16_t bfReserved2; // Réservé (0)
    uint32_t bfOffBits;   // Offset vers les données d'image
};

// En-tête DIB (BITMAPINFOHEADER - 40 octets)
struct BMPInfoHeader {
    uint32_t biSize;          // Taille de cette structure
    int32_t  biWidth;         // Largeur en pixels
    int32_t  biHeight;        // Hauteur en pixels
    uint16_t biPlanes;        // Nombre de plans (1)
    uint16_t biBitCount;      // Bits par pixel (8 pour 256 couleurs)
    uint32_t biCompression;   // Compression (0 = BI_RGB)
    uint32_t biSizeImage;     // Taille des données d'image
    int32_t  biXPelsPerMeter; // Résolution horizontale
    int32_t  biYPelsPerMeter; // Résolution verticale
    uint32_t biClrUsed;       // Nombre de couleurs utilisées
    uint32_t biClrImportant;  // Nombre de couleurs importantes
};

// Entrée de palette (4 octets par couleur)
struct RGBQuad {
    uint8_t rgbBlue;
    uint8_t rgbGreen;
    uint8_t rgbRed;
    uint8_t rgbReserved;
};

#pragma pack(pop)



class BMPfile
{
public:
    BMPfile();
    ~BMPfile();
    void writeBMP8bit(const std::string& filename, int width, int height, const std::vector<uint8_t>& pixels);
};