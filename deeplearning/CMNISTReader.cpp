#include "CMNISTReader.h"
#include <stdexcept>
#include "tools.h"

uint32_t swap_endian(uint32_t val) {
    val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
    return (val << 16) | (val >> 16);
}

CMNISTReader::CMNISTReader(const std::string& labels, const std::string& images) :
    m_labels(labels),
    m_images(images),
    m_counter(0),
    m_offsetImages(0),
    m_offsetLabels(0),
    m_sizeImageFile(-1),
    m_sizeLabelFile(-1),
    m_current(0) {
    m_image_file.exceptions(m_image_file.exceptions() | std::ios::failbit);
    m_label_file.exceptions(m_label_file.exceptions() | std::ios::failbit);
}


void CMNISTReader::Init(uint32_t offset)
{
    // Open files
    m_image_file.open(m_images, std::ios::in | std::ios::binary);
    if (m_image_file.is_open()) {
        HeaderImage header;
        m_offsetImages = sizeof(header);
        m_image_file.read(reinterpret_cast<char*>(&header), sizeof(header));
        header.m_MNimages = swap_endian(header.m_MNimages);
        if (header.m_MNimages != MNimages) {
            throw std::invalid_argument("bad data file, magic number shall be" + std::to_string(MNimages) + "not " + std::to_string(header.m_MNimages));
        }
        header.m_images = swap_endian(header.m_images);
        header.m_cols = swap_endian(header.m_cols);
        header.m_rows = swap_endian(header.m_rows);
        m_counter = header.m_images;
        m_dims.push_back(header.m_rows);
        m_dims.push_back(header.m_cols);
        m_sizeImageFile = GetFileSize(m_image_file);

        std::size_t size = 0;
        if (!m_dims.empty()) {
            size = 1;
        }
        for (std::size_t i = 0; i < m_dims.size(); i++) {
            size *= m_dims[i];
        }
        m_offsetImages += size * dataSize * offset;
    }

    m_label_file.open(m_labels, std::ios::in | std::ios::binary);
    if (m_label_file.is_open()) {
        HeaderLabel header;
        m_offsetLabels = sizeof(header);
        m_label_file.read(reinterpret_cast<char*>(&header), sizeof(header));
        header.m_MNlabels = swap_endian(header.m_MNlabels);
        if (header.m_MNlabels != MNlabels) {
            throw std::invalid_argument("bad data file, magic number shall be" + std::to_string(MNlabels) + "not " + std::to_string(header.m_MNlabels));
        }
        header.m_labels = swap_endian(header.m_labels);
        m_counter = header.m_labels;
        m_sizeLabelFile = GetFileSize(m_label_file);

        m_offsetLabels += labelSize * offset;
    }
}

CMNISTReader::~CMNISTReader() {
    if (m_image_file.is_open()) {
        m_image_file.close();
    }
    if (m_label_file.is_open()) {
        m_label_file.close();
    }
}

std::pair<std::vector<uint8_t>, uint8_t> CMNISTReader::Next(uint32_t offset, uint32_t final) {

    std::pair<std::vector<uint8_t>, uint8_t> output;
    if (m_image_file.is_open()) {
        std::size_t size = 0;
        if (!m_dims.empty()) {
            size = 1;
        }
        for (std::size_t i=0; i < m_dims.size(); i++) {
            size *= m_dims[i];
        }
        std::vector<uint8_t> data(size* dataSize);
        m_image_file.seekg(m_offsetImages, std::ios::beg);
        if (final <= m_counter)
        {
            if ((m_offsetImages + size * dataSize <= m_sizeImageFile) || (m_offsetImages + labelSize <= sizeof(HeaderImage) + final * size * dataSize)) {
                m_image_file.read(reinterpret_cast<char*>(&data[0]), data.size());
                output.first = data;
                m_offsetImages += data.size();
            }
            if (m_label_file.is_open()) {
                uint8_t label = 0xFF;
                m_label_file.seekg(m_offsetLabels, std::ios::beg);
                if ((m_offsetLabels + labelSize <= m_sizeLabelFile) || (m_offsetLabels + labelSize <= sizeof(HeaderLabel) + final * labelSize)) {
                    m_label_file.read(reinterpret_cast<char*>(&label), labelSize);
                    output.second = label;
                    m_offsetLabels += labelSize;
                }
            }
        }
        else
        {
            if (m_offsetImages + size * dataSize <= m_sizeImageFile) {
                m_image_file.read(reinterpret_cast<char*>(&data[0]), data.size());
                output.first = data;
                m_offsetImages += data.size();
            }
            if (m_label_file.is_open()) {
                uint8_t label = 0xFF;
                m_label_file.seekg(m_offsetLabels, std::ios::beg);
                if (m_offsetLabels + labelSize <= m_sizeLabelFile) {
                    m_label_file.read(reinterpret_cast<char*>(&label), labelSize);
                    output.second = label;
                    m_offsetLabels += labelSize;
                }
            }
        }
        if (NotRead(offset, final)) {
            m_current++;
        }
        else {
            m_label_file.close();
            m_image_file.close();
        }
    }
    return output;
}

bool CMNISTReader::NotRead(uint32_t offset, uint32_t final)
{
    if (final <= m_counter){
        return m_current <= m_counter && (final - offset >= m_current);
    }
    else {
        return m_current <= m_counter;
    }
}

uint32_t CMNISTReader::getPCReading() {
    return uint32_t(float(m_current )/ float(m_counter) * 100.0f);
}

uint32_t CMNISTReader::getCols() {
    return m_dims.at(1);
}
uint32_t CMNISTReader::getRows() {
    return m_dims.at(0);
}

void BuildData(const std::string labelsFile, const std::string imagesFile, const std::string dataPath,
    std::vector<Matrix<double>>& expectedY, std::vector<Matrix<double>>& dataImg,std::size_t nb_class,
    bool t, uint32_t offset, uint32_t last)
{
    std::cout << "data building: " << imagesFile << " " << labelsFile << std::endl;
    CMNISTReader mnistReader(dataPath + labelsFile, dataPath + imagesFile);
    mnistReader.Init();
    Matrix<double> img(mnistReader.getCols(), mnistReader.getRows());
    std::pair<std::vector<uint8_t>, uint8_t> data = mnistReader.Next(offset, last);
    while (mnistReader.NotRead(offset, last)) {
        std::size_t i = 0;
        for (uint32_t c = 0; c < mnistReader.getCols(); c++) {
            for (uint32_t r = 0; r < mnistReader.getRows(); r++) {
                img.val(c, r) = normalize<uint8_t, double>(data.first[i], 0x00, 0xFF);
                i++;
            }
        }
        if (t) img = img.transpose();
        dataImg.push_back(img);

        std::vector<double> v_data(nb_class, 0.0);
        Matrix<double> Y(1, nb_class, v_data);
        Y.val(data.second, 0) = 1.0;
        expectedY.push_back(Y);
        data = mnistReader.Next(offset, last);
        std::cout << "\r";
        std::cout << mnistReader.getPCReading() << "%";
    }
    std::cout << std::endl;
}