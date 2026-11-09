#include "BMPfile.h"

#include <iostream>
#include <fstream>
#include <stdexcept>


BMPfile::BMPfile()
{

}

BMPfile::~BMPfile()
{ }

void BMPfile::writeBMP8bit(const std::string& filename, int width, int height, const std::vector<uint8_t>& pixels) {
    if (pixels.size() != static_cast<size_t>(width * height)) {
        throw std::runtime_error("Taille du tableau pixels incorrecte.");
    }

    // Calcul du padding (chaque ligne doit être multiple de 4 octets)
    int rowSize = (width + 3) & ~3;
    int padding = rowSize - width;

    // Taille de la palette (256 couleurs × 4 octets)
    const int paletteSize = 256 * sizeof(RGBQuad);

    // Taille des données d'image
    uint32_t imageSize = rowSize * height;

    // Création des en-têtes
    BMPFileHeader fileHeader{};
    BMPInfoHeader infoHeader{};

    fileHeader.bfType = 0x4D42; // "BM"
    fileHeader.bfOffBits = sizeof(BMPFileHeader) + sizeof(BMPInfoHeader) + paletteSize;
    fileHeader.bfSize = fileHeader.bfOffBits + imageSize;

    infoHeader.biSize = sizeof(BMPInfoHeader);
    infoHeader.biWidth = width;
    infoHeader.biHeight = height;
    infoHeader.biPlanes = 1;
    infoHeader.biBitCount = 8;
    infoHeader.biCompression = 0;
    infoHeader.biSizeImage = imageSize;
    infoHeader.biXPelsPerMeter = 2835; // ~72 DPI
    infoHeader.biYPelsPerMeter = 2835;
    infoHeader.biClrUsed = 256;
    infoHeader.biClrImportant = 256;

    // Ouverture du fichier
    std::ofstream out(filename, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Impossible de créer le fichier BMP.");
    }

    // Écriture des en-têtes
    out.write(reinterpret_cast<const char*>(&fileHeader), sizeof(fileHeader));
    out.write(reinterpret_cast<const char*>(&infoHeader), sizeof(infoHeader));

    // Écriture de la palette (dégradé de gris)
    for (int i = 0; i < 256; ++i) {
        RGBQuad color{ static_cast<uint8_t>(i), static_cast<uint8_t>(i), static_cast<uint8_t>(i), 0 };
        out.write(reinterpret_cast<const char*>(&color), sizeof(color));
    }

    // Écriture des pixels (BMP stocke les lignes de bas en haut)
    for (int y = 0; y < height; ++y) {
        int rowIndex = (height - 1 - y) * width;
        out.write(reinterpret_cast<const char*>(&pixels[rowIndex]), width);
        // Padding
        for (int p = 0; p < padding; ++p) {
            out.put(0);
        }
    }

    out.close();
}