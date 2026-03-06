#include "dataExporter.h"

#include <fstream>
#include <iostream>

DataExporter::DataExporter()
{
}

DataExporter::~DataExporter()
{
}


void DataExporter::AddData(std::string collection, std::string serie, double data)
{
	auto it = m_collections.find(collection);
	if (it != m_collections.end()) {
		auto itS = it->second.find(serie);
		if (itS != it->second.end()) {
			itS->second.push_back(data);
		} else {
			std::vector<double> newSerie;
			newSerie.push_back(data);
			it->second[serie] = newSerie;
		}
	}
	else {
		std::map<std::string, std::vector<double>> newCollection;
		m_collections[collection] = newCollection;
		AddData(collection, serie, data);
	}
}

void DataExporter::ExportToCVS(std::string collection, std::string fileName)
{

	auto it = m_collections.find(collection);
	if (it != m_collections.end()) {
		auto itSerie = it->second;

		std::ofstream file;
		file.exceptions(file.exceptions() | std::ios::failbit);

		try {
			file.open(fileName, std::ios::trunc | std::ios::out);

			std::size_t max = 0;
			for (auto it = itSerie.begin(); it != itSerie.end(); it++) {
				file << it->first << ";";
				if (it->second.size() > max) {
					max = it->second.size();
				}
			}
			file << std::endl;
			size_t i = 0;
			while (i < max) {
				for (auto it = itSerie.begin(); it != itSerie.end(); it++) {
					if (i < it->second.size()) {
						file << std::to_string(it->second[i]);
					}
					file << ";";
				}
				file << std::endl;
				i++;
			}
			file << std::endl;
			file.close();
		}
		catch (std::ofstream::failure& e) {
			std::cout << "cvs file error " << fileName << " : " << e.what() << std::endl;
		}
	}
	else {
		std::cout << "DataExporter::ExportToCVS unknow collection " << collection  << std::endl;
		return;
	}
}

