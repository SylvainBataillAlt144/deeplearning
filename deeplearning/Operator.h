#pragma once
#include <memory>
#include "tools.h"
#include "Matrix.h"
#include "dataExporter.h"

constexpr double DECAY = 0.9;


template <class T>
class OperatorNode
{
public: 
	OperatorNode(){}
	virtual ~OperatorNode() = 0 {}

	virtual std::ostream& Serialize(std::ostream& os) {
		return os;
	}

	virtual std::istream& Deserialize(std::istream& is) {
		return is;
	}

	virtual std::shared_ptr<OperatorNode<T>> copy() {
		return nullptr;
	}

	virtual bool isLearnable() = 0;
	virtual bool isInputUse() = 0;
	virtual bool isOutputUse() = 0;

	virtual std::vector<Matrix<T>> Forward(std::vector<Matrix<T>>& input, bool isTraining = false) = 0;
	virtual std::vector<Matrix<T>> Backward(const std::vector<Matrix<T>>& Doutput, const std::vector<Matrix<T>>& input, const std::vector<Matrix<T>>& output, 
		const T learningRate, std::size_t) = 0;
};

enum class ConvKerSize : uint8_t {
	K1X1 = 1,
	K3X3 = 3,
	K5X5 = 5,
	K7X7 = 7,
	K9X9 = 9,
	K11X11 = 11,
	K11X13 = 13,
};

template<class T, ConvKerSize S = ConvKerSize::K3X3>//S = size of kernels of convolution
class OperatorConvolution : public OperatorNode<T>
{
public:
	OperatorConvolution(uint8_t number, bool rand = true) : m_number(number), m_weights(m_number* uint8_t(S)* uint8_t(S), T(0.1)), m_biais(m_number, T(0.1)),
		m_indexKernel(computeKernelIndex(uint8_t(S), uint8_t(S), 0)),
		m_dbiaisS(m_number, T(0)), m_dWeightS(m_number* uint8_t(S)* uint8_t(S), T(0))
	{
		if(rand){
			CRandom<double>::getInstance().InitDistribution(-1.0, 1.0);
			RandData(); 
		} else {
			CRandom<T>::getInstance().getDistribVector(m_weights, -0.1, 0.1);
		}
	}
	OperatorConvolution(uint8_t number, const std::vector<T> & weights, const std::vector<T> & m_biais) :
		m_number(number), m_weights(weights), m_biais(m_biais),
		m_indexKernel(computeKernelIndex(uint8_t(S), uint8_t(S), 0)),
		m_dbiaisS(m_number, T(0)), m_dWeightS(m_number* uint8_t(S)* uint8_t(S), T(0))
	{
	}
	virtual std::shared_ptr<OperatorNode<T>> copy() override {
		return std::make_shared<OperatorConvolution<T>>(m_number, m_weights, m_biais);
	}

	~OperatorConvolution() {
	}

	bool isLearnable() override{
		return true;
	}

	bool isInputUse() override {
		return true;
	}

	bool isOutputUse()override {
		return false;
	}

	std::vector<Matrix<T>> Backward(const std::vector<Matrix<T>>& Doutput, const std::vector<Matrix<T>>& input, const std::vector<Matrix<T>>&,
		const T learningRate, std::size_t) override {
		std::vector<Matrix<T>> Dinput;
		uint32_t s = uint32_t(S);
		std::vector<Matrix<T>> Dkernel(m_number, Matrix<T>(s, s));

		m_indexOutput = computeKernelIndex(Doutput[0].getCols(), Doutput[0].getRows(), uint32_t(S) / 2);
		IndexImage indexI_O = computeIndexImage(input[0].getCols(), input[0].getRows(), Doutput[0].getCols() - uint32_t(S) / 2 * 2, Doutput[0].getRows() - uint32_t(S) / 2 * 2, 1);
		//kernels_gradient[i, j] = signal.correlate2d(self.input[j], output_gradient[i], "valid")
		std::size_t in = 0;
		std::vector<Matrix<T>> dkers(Doutput.size());
		uint8_t num = m_number;
		auto kernelConvolutionCorrelation = [&dkers, &num, &Doutput, &input](uint32_t d, uint32_t f,
			const std::vector<Matrix<T>>& output,
			std::uint32_t size,
			std::size_t number,
			const std::vector<T>& kernel,
			const std::vector<IndexK>& index,
			const std::vector<Index2D>& indexI,
			const std::vector<T> biais,
			uint32_t offset = 0) {
				for (std::size_t ou = d; ou < f; ou++) {
					std::size_t in = ou / num;
					Matrix<T> Dout = Doutput[ou];
					Matrix<T> dout = Dout.Extract(uint32_t(S) / 2, uint32_t(S) / 2, Doutput[ou].getCols() - uint32_t(S) / 2 * 2, Doutput[ou].getRows() - uint32_t(S) / 2 * 2);
					std::vector<Matrix<T>> dkernel = Matrix<T>::Convolution(input[in], Doutput[ou].getCols() - uint32_t(S) / 2 * 2, 1, dout.getData(),
						index, indexI, std::vector<T>(1, T(0)), 1);
					Matrix<T> dker = dkernel[0].Extract(Doutput[ou].getCols() / 2 - uint32_t(S) / 2, Doutput[ou].getRows() / 2 - uint32_t(S) / 2, uint32_t(S), uint32_t(S));
					dkers[ou] = dker;
				}

			};
		ThreadingTemplate(Doutput.size(), kernelConvolutionCorrelation, Doutput, Doutput[0].getCols() - uint32_t(S) / 2 * 2, 1, Doutput[0].getData(),
					m_indexOutput.m_index, indexI_O.m_index, std::vector<T>(1, T(0)), 1);
		for (std::size_t i = 0; i < dkers.size(); i++) {
				Dkernel[i% m_number] = Dkernel[i % m_number] + dkers[i];
		}

		//kernel is turning to 180°
		//input_gradient[j] += signal.convolve2d(output_gradient[i], self.kernels[i, j], "full")
		std::vector<Matrix<T>> dint(Doutput.size());
		IndexImage indexI = computeIndexImage(input[0].getCols(), input[0].getRows(), uint32_t(S), uint32_t(S));
		std::vector<T> weight = m_weights;
		auto kernelConvolutionDinput = [&weight, &num, &Doutput, &dint](uint32_t d, uint32_t f,
			const std::vector<Matrix<T>>& output,
			std::uint32_t size,
			std::size_t number,
			const std::vector<T>& ker,
			const std::vector<IndexK>& index,
			const std::vector<Index2D>& indexI,
			const std::vector<T> biais,
			uint32_t offset = 0) {
				for (std::size_t ou = d; ou < f; ou++) {
					std::size_t pos = (ou % num) * uint32_t(S) * uint32_t(S);
					std::vector<T> kernel(weight.begin() + pos, weight.begin() + pos + uint32_t(S) * uint32_t(S));
					std::vector<Matrix<T>> di = Matrix<T>::Convolution(Doutput[ou], uint32_t(S), 1, kernel,
						index, indexI, std::vector<T>(1, T(0)));
					dint[ou] = di[0];
				}

			};
		ThreadingTemplate(Doutput.size(), kernelConvolutionDinput, Doutput, uint32_t(S), 1, weight, m_indexKernel.m_index, indexI.m_index, std::vector<T>(1, T(0)), T(0));


		for (std::size_t i = 0; i < dint.size(); i += m_number) {
			Matrix<T> somInt(dint[i].getCols(), dint[i].getRows());
			for (std::size_t n = 0; n < m_number; n++) {
				somInt = somInt + dint[i + n];
			}
			Dinput.push_back(somInt);
		}
		//m_weights[i] -= learningRate * Dkernel[i]
		std::vector<T> dKernelV;
		for (std::size_t ou = 0; ou < Dkernel.size(); ou++) {
			dKernelV.insert(dKernelV.end(), Dkernel[ou].getData().begin(), Dkernel[ou].getData().end());
		}
		for(std::size_t k = 0; k < m_weights.size(); k++){
			m_weights[k] -= dKernelV[k] *learningRate / input.size();
		}

		//self.bias -= learning_rate * output_gradient
		std::vector<T> DBiais(m_number, T(0));
		for (std::size_t i = 0; i < Doutput.size(); i++) {
			DBiais[i % m_number] += Doutput[i % m_number].SumElm() / T(Doutput[i % m_number].getData().size());
		}
		for (std::size_t k = 0; k < m_number; k++) {
			m_biais[k] -= DBiais[k] * learningRate / input.size();
		}
		return Dinput;
	}

	std::vector<Matrix<T>> Forward(std::vector<Matrix<T>>& input, bool isTraining = false) override {
		std::vector<Matrix<T>> output(input.size() * m_number);
		if (input.empty()) {
			return output;
		}

		IndexImage indexI = computeIndexImage(input[0].getCols(), input[0].getRows(), uint32_t(S), uint32_t(S));
		auto kernelConvolutionForward = [&output](uint32_t d, uint32_t f,
			const std::vector<Matrix<T>>& input,
			std::uint32_t size,
			std::size_t number,
			const std::vector<T>& kernel,
			const std::vector<IndexK>& index,
			const std::vector<Index2D>& indexI,
			const std::vector<T> biais,
			uint32_t offset = 0) {
			for (std::size_t in = d; in < f; in++) {
				std::vector<Matrix<T>> out = Matrix<T>::Convolution(input[in], size, number, kernel, index, indexI, biais, offset);
				for (std::size_t o = 0; o < out.size(); o++) {
					output[in * number + o] = out[o];
				}
			}
		};
		ThreadingTemplate(input.size(), kernelConvolutionForward, input, uint32_t(S), m_number, m_weights, m_indexKernel.m_index, indexI.m_index, m_biais, 0);

		return output;
	}

	void SetKernel(uint8_t id, const std::vector<T> & m_kernel) {
		if (m_kernel.size() != m_weights.size() / m_number) {
			throw std::range_error("OperatorConvolution::SetKernel: bad size of kernel");
		}
		if (id > m_number) {
			throw std::range_error("OperatorConvolution::SetKernel: bad id of kernel");
		}
		memcpy(&m_weights[0] + id * m_weights.size() / m_number, &m_kernel[0], sizeof(T)*m_kernel.size());
	}

	Matrix<T> GetKernel(std::size_t id)
	{
		if (id > m_number) {
			throw std::range_error("OperatorConvolution::GetKernel: bad id of kernel");
		}
		std::vector<T> k(uint8_t(S) * uint8_t(S));
		memcpy(&k[0], &m_weights[0] + id * m_weights.size() / m_number, sizeof(T) * k.size());
		return Matrix<T>(uint8_t(S), uint8_t(S), k);
	}

	T SetBiais(std::vector<T> biais) {
		if (m_number != biais.size()) {
			throw std::range_error("OperatorConvolution::GetBiais: bad size of biais");
		}
		m_biais = biais;
	}

	T GetBiais(std::size_t id)
	{
		if (id > m_number) {
			throw std::range_error("OperatorConvolution::GetBiais: bad id of biais");
		}
		return m_biais[id];
	}

	uint8_t NbKernel() {
		return m_number;
	}
	std::ostream& Serialize(std::ostream& os) override{
		os << "nb_kernels= " << std::to_string(m_number) << std::endl;
		os << "weights= " << m_weights;
		os << "biais= " << m_biais;
		return os;
	}

	std::istream& Deserialize(std::istream& is) override {
		std::string label;
		uint8_t kernels = 0;
		std::vector<T> weights;
		std::vector<T> biais;
		is >> label;
		if (label == "nb_kernels=") {
			is >> kernels;
		}
		is >> label;
		if (label == "weights=") {
			is >> weights;
		}
		is >> label;
		if (label == "biais=") {
			is >> biais;
		}
		m_weights = weights;
		m_biais = biais;
		return is;
	}
protected:
	 

uint8_t m_number; // number of kernels of convolution
std::vector<T> m_weights; //weight for each parameters of each kernel of convolution 
std::vector<T> m_biais; // biais for each kernel of convolution

std::vector<T> m_dWeightS;
std::vector<T> m_dbiaisS;

IndexKernel m_indexKernel;
IndexKernel m_indexOutput;

void RandData() {
	CRandom<T>::getInstance().getRandVector(m_weights);
	CRandom<T>::getInstance().getRandVector(m_biais);
}
public:
static IndexKernel computeKernelIndex(const uint32_t nbCols, const uint32_t nbRows,uint32_t offset = 0)
{
	IndexKernel ret;
	ret.m_index.resize((nbCols - offset * 2) * (nbRows- offset * 2));
	ret.m_indexR.resize((nbCols - offset * 2) * (nbRows - offset * 2));
	int32_t mC = -int32_t(nbCols) / 2;
	if (nbCols % 2 == 0)
		mC += offset;
	int32_t mR = -int32_t(nbRows) / 2;
	if (nbRows % 2 == 0)
		mR += offset;
	for (uint32_t c = 0; c < nbCols - offset * 2; c++) {
		for (uint32_t r = 0; r < nbRows - offset * 2; r++) {
			ret.m_index[c * (nbCols - offset * 2) + r].m_w = c * (nbCols - offset * 2) + r;
			ret.m_index[c * (nbCols - offset * 2) + r].m_c = mC + r;
			ret.m_index[c * (nbCols - offset * 2) + r].m_r = mR + c;
			ret.m_indexR[c * (nbCols - offset * 2) + r].m_w = ((nbCols - offset * 2) * (nbRows - offset * 2) - 1) - (c * (nbCols - offset * 2) + r);
			ret.m_indexR[c * (nbCols - offset * 2) + r].m_c = mC + r;
			ret.m_indexR[c * (nbCols - offset * 2) + r].m_r = mR + c;
		}
	}
	return ret;
}
static IndexImage computeIndexImage(const uint32_t nbCols, const uint32_t nbRows, const uint32_t nbKerCols, const uint32_t nbKerRows, uint32_t offset = 0)
{
	IndexImage ret;
	for (uint32_t r = nbKerRows / 2; r < nbRows - nbKerRows / 2 + offset; r++) {
		for (uint32_t c = nbKerCols / 2; c < nbCols - nbKerCols / 2 + offset; c++) {
			ret.m_index.push_back(Index2D(c, r));
		}
	}
	return ret;
}
};

//rows = inputs
//cols = number of neuron
template<class T>
class OperatorPerceptron: public OperatorNode<T>
{
public:
	OperatorPerceptron(uint32_t inputs, uint32_t neurons) : m_weight(inputs, neurons, T(0.0)), m_biais(neurons, 0.0),
		m_dbiaisS(neurons, 0.0), m_dWeightS(inputs, neurons, T(0.0)) {
		CRandom<double>::getInstance().InitDistribution(-1.0, 1.0);
		m_weight.initRand();
		CRandom<T>::getInstance().getRandVector(m_biais);
	}
	OperatorPerceptron(uint32_t inputs, uint32_t neurons, const Matrix<T> & weight, const std::vector<T> & biais) : m_weight(inputs, neurons, weight.getData()), m_biais(biais),
		m_dbiaisS(neurons, 0.0), m_dWeightS(inputs, neurons, T(0.0)) {
		if (m_biais.size() != neurons) {
			throw std::range_error("OperatorPerceptron::OperatorPerceptron: biais size = " +std::to_string(biais.size()) + "!= neurons= " + std::to_string(neurons));
		}
	}
	virtual std::shared_ptr<OperatorNode<T>> copy() override {
		return std::make_shared<OperatorPerceptron<T>>(m_weight.getCols(), m_weight.getRows(), m_weight, m_biais);
	}
	virtual ~OperatorPerceptron() {
	}

	bool isLearnable() override {
		return true;
	}

	bool isInputUse() override {
		return true;
	}

	bool isOutputUse()override {
		return false;
	}

	std::vector<Matrix<T>> Backward(const std::vector<Matrix<T>>& Doutput, const std::vector<Matrix<T>>& input, const std::vector<Matrix<T>>& ,
		const T learningRate, std::size_t) override {
		if (input.size() != 1) {
			throw std::invalid_argument("OperatorPerceptron::Backward : size input shall be equal to 1");
		}
		if (Doutput.size() != 1) {
			throw std::invalid_argument("OperatorPerceptron::Backward : size Doutput shall be equal to 1");
		}

		std::vector<Matrix<T>> Dinput;
		Matrix<T> dWeight;
		T m = T(Doutput[0].getCols());

		//weights_gradient = np.dot(output_gradient, self.input.T)
		dWeight = Doutput[0] * input[0].transpose();
		dWeight = dWeight / m;

		//input_gradient = np.dot(self.weights.T, output_gradient)
		Dinput.push_back(m_weight.transpose() * Doutput[0]);

		//self.weights -= learning_rate * weights_gradient
		m_dWeightS = m_dWeightS *DECAY + dWeight.applyFunction([](const T& a) {
			return (1- DECAY) *(a * a);});
		m_weight = m_weight - dWeight /*/ (m_dWeightS.applyFunction([](const T& a) {
			return std::sqrt(a+ 1e-12);
		}))*/* learningRate;

		//self.bias -= learning_rate * output_gradient
		std::vector<T> DBiais =  Doutput[0].SumAxis(AxisFormat::ROW);
		for (std::size_t i = 0; i < DBiais.size(); i++) {
			DBiais[i] *= T(1) / m;
			m_dbiaisS[i] = DECAY * m_dbiaisS[i] + (1-DECAY) * DBiais[i]* DBiais[i];
			m_biais[i] -= DBiais[i]/*/(std::sqrt(m_dbiaisS[i])+1e-12)*/ *learningRate;
		}
		return Dinput;
	}

	// vector<Matrix<T> input.size == 1 else error
	// each column of input correspond of an datum of a dataset.
	std::vector<Matrix<T>> Forward(std::vector<Matrix<T>>& input, bool isTraining = false) override {
		if (input.size() != 1) {
			throw std::invalid_argument("OperatorPerceptron::Forward : size input shall be equal to 1");
		}
		std::vector<Matrix<T>> output;
		output.push_back(m_weight* input[0] + m_biais);
		return output;
	} 
	std::ostream& Serialize(std::ostream& os) override {
		os << "weights= "  << m_weight;
		os << "biais= " << m_biais;
		return os;
	}
	std::istream& Deserialize(std::istream& is) override {
		std::string label("weights=");
		Matrix<T> weights;
		std::vector<T> biais;
		is >> label;
		if (label == "weights=") {
			is >> weights;
		}
		is >> label;
		if (label == "biais=") {
			is >> biais;
		}
		m_weight = weights;
		m_biais = biais;
		return is;
	}
protected:
	Matrix<T> m_weight;
	std::vector<T> m_biais;

	Matrix<T> m_dWeightS;
	std::vector<T> m_dbiaisS;
};

enum class PollingSize : uint8_t {
	P2 = 2,
	P3 = 3,
	P4 = 4
};
template<class T, PollingSize S = PollingSize::P2>//S = size of pooling
class OperatorPooling: public OperatorNode<T>
{
public:
	OperatorPooling(): m_cI(uint8_t(S)* uint8_t(S), 0), m_rI(uint8_t(S)* uint8_t(S), 0)
	{	 
		Init();
	}
	~OperatorPooling() {
	}

	bool isLearnable() override {
		return false;
	}

	bool isInputUse() override {
		return true;
	}

	bool isOutputUse()override {
		return true;
	}

	std::vector<Matrix<T>> Backward(const std::vector<Matrix<T>>& Doutput, const std::vector<Matrix<T>>& input, const std::vector<Matrix<T>>& output,
		const T learningRate, std::size_t) override {

		std::vector<Matrix<T>> Dinput;
		for (size_t i = 0; i < input.size(); i++) {
			Matrix<T> din(input[i].getCols(), input[i].getRows(), T(0));
			for (uint32_t c = 0; c < Doutput[i].getCols(); c++) {
				for (uint32_t r = 0; r < Doutput[i].getRows(); r++) {
					bool isFinded = false;
					for (uint8_t p = 0; p < uint8_t(S) * uint8_t(S); p++) {
						if (input[i].val(c * uint8_t(S) + m_cI[p], r * uint8_t(S) + m_rI[p]) == output[i].val(c, r))
						{
							din.val(c * uint8_t(S) + m_cI[p], r * uint8_t(S) + m_rI[p]) = Doutput[i].val(c, r);
							break;
						}
					}
				}
			}
			Dinput.push_back(din);
		}
		return Dinput;
	}

	std::vector<Matrix<T>> Forward(std::vector<Matrix<T>>& input, bool isTraining = false) override {
		std::vector<Matrix<T>> output;
		for (std::size_t in = 0;in < input.size(); in++) {
			Matrix<T> out(input[in].getCols()/uint8_t(S), input[in].getRows() / uint8_t(S));
			for (uint32_t c = 0; c < out.getCols(); c++) {
				for (uint32_t r = 0; r < out.getRows(); r++) {
					for (uint8_t p = 0; p < uint8_t(S) * uint8_t(S); p++) {
						out.val(c, r) = maxAbs(out.val(c, r), input[in].val(c* uint8_t(S) +m_cI[p], r* uint8_t(S) +m_rI[p]));
					}
				}
			}
			output.push_back(out);
		}
		return output;
	}

protected:
std::vector<int8_t> m_cI;
std::vector<int8_t> m_rI;

void Init() {
	for (int8_t c = 0; c < uint8_t(S); c++) {
		for (int8_t r = 0; r < uint8_t(S); r++) {
			m_cI[c * uint8_t(S) + r] = r;
			m_rI[c * uint8_t(S) + r] = c;
		}
	}
};
};

enum class FlattenFormat {
	COL,
	ROW
};
template< class T, FlattenFormat F = FlattenFormat::COL>
class OperatorFlatten : public OperatorNode<T>
{
public:
	OperatorFlatten(uint16_t nbFlat=1): m_nbFlat(nbFlat){
		if (m_nbFlat == 0) {
			throw std::range_error("OperatorFlatten::OperatorFlatten: flat shall be not null");
		}
	}
	~OperatorFlatten() {
	}


	bool isLearnable() override {
		return false;
	}

	bool isInputUse() override {
		return true;
	}

	bool isOutputUse()override {
		return false;
	}

	std::vector<Matrix<T>> Backward(const std::vector<Matrix<T>>& Doutput, const std::vector<Matrix<T>>& input, const std::vector<Matrix<T>>& ,
		const T learningRate, std::size_t) override {
		std::vector<Matrix<T>> Dinput;
		std::size_t div = input.size() / Doutput.size();
		for (std::size_t i = 0; i < Doutput.size(); i++) {
			std::vector<T> dataOut = Doutput[i].getData();
			for (std::size_t d = 0; d < div; d++) {
				std::vector<T> data(input[i].getCols()*input[i].getRows());
				memcpy(&data[0], &dataOut[dataOut.size()/ div * d], sizeof(T)* dataOut.size()/div);
				Matrix<T> din(input[i].getCols(), input[i].getRows(), data);
				Dinput.push_back(din);
			}
		}
		return Dinput;
	}

	std::vector<Matrix<T>> Forward(std::vector<Matrix<T>>& input, bool isTraining = false) override {
		std::vector<T> out;
		std::vector<Matrix<T>> output;
		if (input.size() % m_nbFlat != 0) {
			throw std::range_error("OperatorFlatten::Forward: bad flat range");
		}
		for (std::size_t i = 0; i < input.size(); i++) {
			std::vector<T> v = input[i].Flatten();
			out.insert(out.end(), v.begin(), v.end());
			if ((i+1) % m_nbFlat == 0) {
				if (F == FlattenFormat::ROW) {
					Matrix<T> outM(uint32_t(out.size()), 1, out);
					output.push_back(outM);
				}
				else {
					Matrix<T> outM(1, uint32_t(out.size()), out);
					output.push_back(outM);
				}
				out.clear();
			}
		}
		return output;
	}
protected:
	uint16_t m_nbFlat;
};

template< class T, FlattenFormat F = FlattenFormat::COL>
class OperatorConcat : public OperatorNode<T>
{
public:
	OperatorConcat(uint16_t nbSlice = 1) : m_nbSlice(nbSlice) {
		if (m_nbSlice == 0) {
			throw std::range_error("OperatorConcat::OperatorConcat: slice shall be not null");
		}
	}
	~OperatorConcat() {
	}


	bool isLearnable() override {
		return false;
	}

	bool isInputUse() override {
		return true;
	}

	bool isOutputUse()override {
		return false;
	}

	std::vector<Matrix<T>> Backward(const std::vector<Matrix<T>>& Doutput, const std::vector<Matrix<T>>& input, const std::vector<Matrix<T>>& ,
		const T learningRate, std::size_t) override {

		if (Doutput.empty()) {
			throw std::range_error("OperatorConcat::Backward: Doutput not empty");
		}
		std::vector<Matrix<T>> Dinput;
		if (FlattenFormat::ROW == F) {
			for (std::size_t i = 0; i < input.size(); i++) {
				std::vector<T> d(Doutput[0].getData().begin()+ i*input[0].getData().size(),
					Doutput[0].getData().begin() + i * input[0].getData().size()+ input[0].getData().size());
				Dinput.push_back(Matrix<T>(input[i].getCols(), input[i].getRows(), d));
			}
		}
		else {
			for (std::size_t i = 0; i < input.size(); i++) {
				std::vector<T> d;
				for (std::size_t r = 0; r < input[i].getRows(); r++) {
					d.insert(d.end(), Doutput[0].getData().begin() + r * Doutput[0].getCols() + i * input[i].getCols(),
						Doutput[0].getData().begin() + r * Doutput[0].getCols() + i * input[i].getCols() + input[i].getCols());
				}
				Dinput.push_back(Matrix<T>(input[i].getCols(), input[i].getRows(), d));
			}
		}
		return Dinput;
	}
	std::vector<Matrix<T>> Forward(std::vector<Matrix<T>>& input, bool isTraining = false) override {
		std::vector<Matrix<T>> output;
		if (input.empty()) {
				throw std::range_error("OperatorConcat::Forward: input empty");
		}
		if (input.size() % m_nbSlice != 0) {
			throw std::range_error("OperatorConcat::Forward: bad slice range");
		}
		Matrix<T> out;
		out.Concat(input, F == FlattenFormat::COL? AxisFormat::COL: AxisFormat::ROW);
		output.push_back(out);
		return output;
	}
protected:
	uint16_t m_nbSlice;
};

template< class T, FlattenFormat F = FlattenFormat::COL>
class OperatorSplit : public OperatorNode<T>
{
public:
	OperatorSplit(uint16_t nbSlice = 1) : m_nbSlice(nbSlice) {
		if (m_nbSlice == 0) {
			throw std::range_error("OperatorSplit::OperatorSplit: slice shall be not null");
		}
	}
	~OperatorSplit() {
	}


	bool isLearnable() override {
		return false;
	}

	bool isInputUse() override {
		return true;
	}

	bool isOutputUse()override {
		return false;
	}

	std::vector<Matrix<T>> Backward(const std::vector<Matrix<T>>& Doutput, const std::vector<Matrix<T>>& input, const std::vector<Matrix<T>>& ,
		const T learningRate, std::size_t) override {
		std::vector<Matrix<T>> Dinput;
		if (input.empty()) {
			throw std::range_error("OperatorSplit::Backward: input empty");
		}
		if (input.size() % m_nbSlice != 0) {
			throw std::range_error("OperatorSplit::Backward: bad slice range");
		}
		Matrix<T> din;
		din.Concat(input, F == FlattenFormat::COL ? AxisFormat::COL : AxisFormat::ROW);
		Dinput.push_back(din);
		return Dinput;
	}
	std::vector<Matrix<T>> Forward(std::vector<Matrix<T>>& input, bool isTraining = false) override {
		if (input.empty()) {
			throw std::range_error("OperatorConcat::Forward: input not empty");
		}
		std::vector<Matrix<T>> output;

		if (FlattenFormat::ROW == F) {
			for (std::size_t i = 0; i < input.size(); i++) {
				for (uint32_t r = 0; r < input[i].getRows(); r++) {
					std::vector<T> row = input[i].getData(AxisFormat::ROW, r);
					output.push_back(Matrix<T>(input[i].getCols(), 1, row));
				}
			}
		}
		else {
			for (std::size_t i = 0; i < input.size(); i++) {
				for (uint32_t c = 0; c < input[i].getCols(); c++) {
					std::vector<T> col = input[i].getData(AxisFormat::COL, c);
					output.push_back(Matrix<T>(1, input[i].getRows(), col));
				}
			}
		}
		return output;
	}
protected:
	uint16_t m_nbSlice;
};


template< class T>
class OperatorFunction : public OperatorNode<T>
{
public:
	OperatorFunction(std::function<T(const T&)> func, std::function<T(const T&)> Dfunc): m_function(func), m_Dfunction(Dfunc){
	}
	~OperatorFunction() {
	}


	bool isLearnable() override {
		return false;
	}

	bool isInputUse() override {
		return false;
	}

	bool isOutputUse()override {
		return true;
	}

	std::vector<Matrix<T>> Backward(const std::vector<Matrix<T>>& Doutput, const std::vector<Matrix<T>>& , const std::vector<Matrix<T>>& output,
		const T learningRate, std::size_t) override {
		std::vector<Matrix<T>> Dinput;
		for (std::size_t i = 0; i < output.size(); i++) {
			Matrix<T> d = output[i].applyFunction(m_Dfunction);
			Matrix<T> dI = Doutput[i];
			dI.MultElm(d);
			Dinput.push_back(dI);
		}
		return Dinput;
	}

	std::vector<Matrix<T>> Forward(std::vector<Matrix<T>>& input, bool isTraining = false) override {
		std::vector<Matrix<T>> output;
		for (std::size_t i = 0; i < input.size(); i++) {
			output.push_back(input[i].applyFunction(m_function));
		}
		return output;
	}
protected:
	std::function<T(const T&)> m_function;
	std::function<T(const T&)> m_Dfunction;
};

template <class T>
class OperatorSoftMax: public OperatorNode<T>
{
public:
	OperatorSoftMax() {}
	virtual ~OperatorSoftMax() {}

	bool isLearnable() override {
		return false;
	}

	bool isInputUse() override {
		return false;
	}

	bool isOutputUse()override {
		return true;
	}

	std::vector<Matrix<T>> Forward(std::vector<Matrix<T>>& input, bool isTraining = false) override {
		std::vector<Matrix<T>> output;
		for (std::size_t i = 0; i < input.size(); i++) {
			std::vector<T> vout = softmax_forward(input[i]);
			Matrix<T> out(input[i].getCols(), input[i].getRows(), vout);
			output.push_back(out);
		}
		return output;
	}

	std::vector<Matrix<T>> Backward(const std::vector<Matrix<T>>& Doutput, const std::vector<Matrix<T>>& , const std::vector<Matrix<T>>& output,
		const T learningRate, std::size_t) override {
		std::vector<Matrix<T>> Dinput;
		for (std::size_t i = 0; i < output.size(); i++) {
			Matrix<T> dint = softmax_backward(output[i], Doutput[i]);
			Dinput.push_back(dint);
		}
		return Dinput;
	}
};

template <class T>
class OperatorDropout : public OperatorNode<T>
{
public:
	OperatorDropout(uint8_t percentDropout): m_percentDropout(percentDropout) {}
	virtual ~OperatorDropout() {}

	bool isLearnable() override {
		return false;
	}

	bool isInputUse() override {
		return false;
	}

	bool isOutputUse()override {
		return false;
	}

	std::vector<Matrix<T>> Forward(std::vector<Matrix<T>>& input, bool isTraining = false) override {
		std::vector<Matrix<T>> output;
		if (isTraining) {
			m_filter.clear();
			CRandom<double>::getInstance().InitDistribution(0.0, 1.0);
			for (std::size_t i = 0; i < input.size(); i++) {
				Matrix<T> out = input[i];
				std::vector<T> data(input[i].getCols(), input[i].getRows());
				Matrix<T> m(input[i].getCols(), input[i].getRows());
				m.initRand();
				m_filter.push_back(m.applyFunction([this](const T& a) {
					T v = T(m_percentDropout%100) / 100.0;
					T r = 1.0/ (1.0 - v);
					if (v > a) {
						r = 0.0;
					}
					return r;
					}));
				out.MultElm(m);
				output.push_back(out);
			}
		}
		else {
			output = input;
		}
		return output;
	}

	std::vector<Matrix<T>> Backward(const std::vector<Matrix<T>>& Doutput, const std::vector<Matrix<T>>&, const std::vector<Matrix<T>>& output,
		const T learningRate, std::size_t) override {
		std::vector<Matrix<T>> Dinput;

		for (std::size_t i = 0; i < Doutput.size(); i++) {
			Matrix<T> m = Doutput[i];
			m.MultElm(m_filter[i]);
			Dinput.push_back(m);
		}

		return Dinput;
	}
private:
	std::vector<Matrix<T>> m_filter;
	uint8_t m_percentDropout;
};