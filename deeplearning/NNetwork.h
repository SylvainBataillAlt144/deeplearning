#pragma once

#include <memory>

template<class T>
class Layer {
public:
	Layer(std::shared_ptr<OperatorNode<T>> ope, std::string name) : m_operator(ope), m_name(name) {

	}

	std::shared_ptr<OperatorNode<T>> m_operator;
	std::string m_name;
	std::vector<Matrix<T>> inputForward;
	std::vector<Matrix<T>> outputForward;
	std::vector<Matrix<T>> DinputBackward;
	std::vector<Matrix<T>> DoutputBackward;

	friend std::ostream& operator<<(std::ostream& os, const Layer& layer) {
		os << layer.m_name << std::endl;
		layer.m_operator->Serialize(os);
		return os;
	}
	friend std::istream& operator>>(std::istream& is, Layer& layer) {
		std::string name;
		is >> name;
		if (layer.m_name == name) {
			layer.m_operator->Deserialize(is);
		}
		return is;
	}
};

class NNetwork {
public:
	NNetwork(std::string name, std::function < std::vector<Matrix<double>>(const std::vector<Matrix<double>>&, const std::vector<Matrix<double>>&)> func):
		m_name(name), m_lastBackwardId(0), m_gradient(func), m_debug(false){
	}
	void AddLayer(const std::shared_ptr<Layer<double>> layer) {
		if (!layer->m_operator->isLearnable() && m_layers.size() == m_lastBackwardId) {
			m_lastBackwardId++;
		}
		m_layers.push_back(layer);
	}
	template<class T>
	void Forward(const std::vector<Matrix<T>>& data, std::vector<Matrix<T>>& output, bool isTraining = false) {

		std::vector<Matrix<T>> currentData = data;
		std::vector<Matrix<T>> currentoutput;
		for (std::size_t i = 0; i < m_layers.size(); i++) {
			currentoutput = m_layers[i]->m_operator->Forward(currentData, isTraining);
			if (isTraining && m_layers[i]->m_operator->isInputUse()) {
				m_layers[i]->inputForward = currentData;
			}
			if (isTraining && m_layers[i]->m_operator->isOutputUse()) {
				m_layers[i]->outputForward = currentoutput;
			}
			currentData = currentoutput;
		}
		output = currentoutput;
	}

	template<class T>
	void Backward(const std::vector<Matrix<T>>& expected, const std::vector<Matrix<T>>& outputA, double learningRate) {

		Serialize serialiser;
		if (m_debug) {
			serialiser.Save(".//networkDump.txt", m_name + "\n");
		}

		std::vector<Matrix<T>> dOutputA = m_gradient(expected, outputA);
		m_layers[m_layers.size() - 1]->DoutputBackward = dOutputA;
		for (int64_t i = m_layers.size() - 1;!m_layers.empty() && i >= m_lastBackwardId; i--) {
			m_layers[i]->DinputBackward = m_layers[i]->m_operator->Backward(m_layers[i]->DoutputBackward, m_layers[i]->inputForward, m_layers[i]->outputForward, learningRate, i);
			if (i > 0) {
				m_layers[i - 1]->DoutputBackward = m_layers[i]->DinputBackward;
			}

			if (m_debug) {
				serialiser.Save(".//networkDump.txt", m_layers[i]->m_name + "\n", false);
				serialiser.Save(".//networkDump.txt", "Doutput = \n", false);
				serialiser.Save(".//networkDump.txt", m_layers[i]->DoutputBackward, false);
				serialiser.Save(".//networkDump.txt", "Dinput = \n", false);
				serialiser.Save(".//networkDump.txt", m_layers[i]->DinputBackward, false);
				serialiser.Save(".//networkDump.txt", "output = \n", false);
				serialiser.Save(".//networkDump.txt", m_layers[i]->outputForward, false);
				serialiser.Save(".//networkDump.txt", "input = \n", false);
				serialiser.Save(".//networkDump.txt", m_layers[i]->inputForward, false);
				serialiser.Save(".//networkDump.txt", "data= \n", false);
			}
		}


	}
	friend std::ostream& operator<<(std::ostream& os, const NNetwork& data) {
		os << data.m_name << std::endl;
		os << "nb_Layers= " << data.m_layers.size() << std::endl;
		for (std::size_t i = 0; i < data.m_layers.size(); i++) {
			os << *(data.m_layers[i].get());
		}
		return os;
	}
	friend std::istream& operator>>(std::istream& is , NNetwork& data) {
		is >> data.m_name;
		std::size_t layer_size = 0;
		std::string str;
		is >> str; 
		is >> layer_size;
		if (layer_size == data.m_layers.size()) {
			for (std::size_t i = 0; i < data.m_layers.size(); i++) {
				is >> *data.m_layers[i].get();
			}
		}
		return is;
	}

	std::string m_name;
	std::vector< std::shared_ptr<Layer<double>>> m_layers;
	bool m_debug;
protected:
	int64_t m_lastBackwardId;

	std::function < std::vector<Matrix<double>>(const std::vector<Matrix<double>>&, const std::vector<Matrix<double>>&)> m_gradient;
};