#pragma once


#include <map>
#include <vector>
#include <string>

#include "singleton.h"


class DataExporter : public Singleton<DataExporter>
{
    friend class Singleton<DataExporter>;
private:
    DataExporter();
    ~DataExporter();

    std::map<std::string, std::map<std::string, std::vector<double>>> m_collections;
public:

    void AddData(std::string collection, std::string serie, double data);

    void ExportToCVS(std::string collection, std::string fileName);
};