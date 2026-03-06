#pragma once

#include <string>
#include <vector>
#include <fstream>

#include "Matrix.h"

constexpr uint32_t	MNlabels = 0x0801;
constexpr uint32_t	MNimages = 0x0803;
constexpr uint32_t	labelSize = 0x1;
constexpr uint32_t	dataSize = 0x1;
constexpr int64_t	offsetFirstLabel = 0x8;
constexpr int64_t	offsetFisrtPixel = 0x10;
struct HeaderLabel {
	uint32_t	m_MNlabels;
	uint32_t	m_labels;
};

struct HeaderImage {
	uint32_t	m_MNimages;
	uint32_t	m_images;
	uint32_t	m_rows;
	uint32_t	m_cols;
};

class CMNISTReader
{
public:
	CMNISTReader(const std::string & labels, const std::string & images);
	~CMNISTReader();
	CMNISTReader(CMNISTReader&) = delete;

	void Init();
	std::pair<std::vector<uint8_t>, uint8_t> Next();
	bool NotRead();
	uint32_t getPCReading();

	uint32_t getCols();
	uint32_t getRows();

protected:
	std::ifstream m_image_file;
	std::ifstream m_label_file;
	std::string m_labels;
	std::string m_images;
	uint32_t m_counter;
	std::vector<uint32_t> m_dims;
	uint64_t m_offsetImages;
	int64_t m_offsetLabels;
	uint64_t m_sizeImageFile;
	int64_t m_sizeLabelFile;
	uint32_t m_current;

};

void BuildData(const std::string labelsFile, const std::string imagesFile, const std::string dataPath,
	std::vector<Matrix<double>>& expectedY, std::vector<Matrix<double>>& dataImg);
