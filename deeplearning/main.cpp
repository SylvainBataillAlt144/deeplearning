

#include <iostream>
#include <memory>
#include <iomanip>
#include "Operator.h"
#include "NNetwork.h"
#include "sigmoide.h"
#include "CRandom.h"
#include "CMNISTReader.h"
#include "tools.h"
#include "dataExporter.h"

const std::string dataPath = "C:\\Users\\SylvainBATAILLÉ\\Downloads\\idx_dataset\\";

//void CNN(bool m_training = false, bool m_testing = true, bool m_load = true, std::string m_file = ".\\CNN_bis.txt", bool m_save = false) {
void CNN(bool m_training = true, bool m_testing = true, bool m_load = false, std::string m_file = ".\\CNN.txt", bool m_save = true) {
	uint8_t nbConv = 4;
	uint32_t imgC = 28, imgR = 28;
	uint32_t nbNi1 = 120, nbNi2 = 84, nbNO = 10;
	uint16_t batchSize = 1000;

	NNetwork m_nNetwork("CNN_Network", DCrossEntropyLoss_Matrix<double>);

	uint32_t nbImputPerceptron1 = ((imgC * imgR) * nbConv / 4)*nbConv / 4;
	std::shared_ptr<OperatorConvolution<double, ConvKerSize::K5X5>> opeConv = std::make_shared<OperatorConvolution<double, ConvKerSize::K5X5>>(nbConv, true);
	std::shared_ptr<OperatorConvolution<double, ConvKerSize::K3X3>> opeConv2 = std::make_shared<OperatorConvolution<double, ConvKerSize::K3X3>>(nbConv, true);
	std::shared_ptr<OperatorPooling<double, PollingSize::P2>> opePooling = std::make_shared<OperatorPooling<double, PollingSize::P2>>();
	std::shared_ptr<OperatorFlatten<double, FlattenFormat::COL>> opeFlatten = std::make_shared <OperatorFlatten<double, FlattenFormat::COL>>(nbConv*nbConv );
	std::shared_ptr<OperatorConcat<double, FlattenFormat::COL>> opeConcat = std::make_shared <OperatorConcat<double, FlattenFormat::COL>>();
	std::shared_ptr<OperatorFunction<double>> opeSigmoid = std::make_shared<OperatorFunction<double>>(sigmoidStable<double>, DSigmoid<double>);
	std::shared_ptr<OperatorFunction<double>> opeReLU = std::make_shared<OperatorFunction<double>>(Relu<double>, DRelu<double>);
	std::shared_ptr<OperatorPerceptron<double>> opePerceptron1 = std::make_shared<OperatorPerceptron<double>>(nbImputPerceptron1, nbNi1);
	std::shared_ptr<OperatorDropout<double>> opeDropout = std::make_shared<OperatorDropout<double>>(20);
	std::shared_ptr<OperatorPerceptron<double>> opePerceptron2 = std::make_shared<OperatorPerceptron<double>>(nbNi1, nbNi2);
	std::shared_ptr<OperatorPerceptron<double>> opeOutput = std::make_shared<OperatorPerceptron<double>>(nbNi2, nbNO);

	m_nNetwork.AddLayer(std::shared_ptr<Layer<double>>(std::make_shared<Layer<double>>(opeConv, "3conv5x5")));
	m_nNetwork.AddLayer(std::shared_ptr<Layer<double>>(std::make_shared<Layer<double>>(opePooling, "pooling2x2")));
	m_nNetwork.AddLayer(std::shared_ptr<Layer<double>>(std::make_shared<Layer<double>>(opeReLU, "activationReLUConv1")));

	m_nNetwork.AddLayer(std::shared_ptr<Layer<double>>(std::make_shared<Layer<double>>(opeConv2, "3conv3x3")));
	m_nNetwork.AddLayer(std::shared_ptr<Layer<double>>(std::make_shared<Layer<double>>(opePooling, "pooling2x2")));
	m_nNetwork.AddLayer(std::shared_ptr<Layer<double>>(std::make_shared<Layer<double>>(opeReLU, "activationReLUConv2")));

	m_nNetwork.AddLayer(std::shared_ptr<Layer<double>>(std::make_shared<Layer<double>>(opeFlatten, "flattenCol")));
	m_nNetwork.AddLayer(std::shared_ptr<Layer<double>>(std::make_shared<Layer<double>>(opeConcat, "concatCol")));
	m_nNetwork.AddLayer(std::shared_ptr<Layer<double>>(std::make_shared<Layer<double>>(opePerceptron1, "perceptrion1")));
	m_nNetwork.AddLayer(std::shared_ptr<Layer<double>>(std::make_shared<Layer<double>>(opeSigmoid, "activation1")));
	m_nNetwork.AddLayer(std::shared_ptr<Layer<double>>(std::make_shared<Layer<double>>(opeDropout, "dropout_perceptron1")));
	m_nNetwork.AddLayer(std::shared_ptr<Layer<double>>(std::make_shared<Layer<double>>(opePerceptron2, "perceptrion2")));
	m_nNetwork.AddLayer(std::shared_ptr<Layer<double>>(std::make_shared<Layer<double>>(opeSigmoid, "activation1")));
	m_nNetwork.AddLayer(std::shared_ptr<Layer<double>>(std::make_shared<Layer<double>>(opeOutput, "output10")));
	m_nNetwork.AddLayer(std::shared_ptr<Layer<double>>(std::make_shared<Layer<double>>(opeSigmoid, "activation10")));

	Serialize serialiser;
	if (m_load) {
		bool res = serialiser.Load(m_file, m_nNetwork);
		if (!res) {
			m_training = false;
		}

	}

	if (m_training) {
		std::string labelsFile = "train-labels.idx1-ubyte";
		std::string imagesFile = "train-images.idx3-ubyte";

		std::vector<Matrix<double>> expectedY;
		std::vector<Matrix<double>> dataImg;
		std::cout << "training:" << std::endl;
		std::cout << "	read data training:" << std::endl;
		BuildData(labelsFile, imagesFile, dataPath, expectedY, dataImg);

		bool learningIsFinish = false;
		std::cout << "	training with " << expectedY.size() << " data:" << std::endl;
		std::size_t nbIter = 350;
		uint64_t ttotal = 0;
		uint64_t p = 0;
		double Emin = 1.0;
		size_t last = 0;
		double learningRate = 10.0; // 5.0;//2.5;
		double var = learningRate/2.0;
		std::size_t nbBatch = dataImg.size() / batchSize;
		for (std::size_t iter = 0; iter < nbIter && !learningIsFinish; iter++) {
			for (std::size_t b = 0; b < nbBatch && !learningIsFinish; b++) {

				auto t1 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
				std::cout << "\r";
				std::cout << std::to_string(double(p) / double(nbBatch * nbIter) * 100.0) << "%";
				std::vector<Matrix<double>> input;
				std::vector<Matrix<double>> eY;
				std::vector<Matrix<double>> expY(1);
				std::vector<Matrix<double>> outputA;
				input.insert(input.end(), dataImg.begin() + b * batchSize, dataImg.begin() + b * batchSize + batchSize);
				eY.insert(eY.end(), expectedY.begin() + b * batchSize, expectedY.begin() + b * batchSize + batchSize);
				expY[0].Concat(eY, AxisFormat::COL);

				m_nNetwork.Forward(input, outputA, true);

				//cost function 
				double E = MSE(expY[0], outputA[0]);
				if (E < Emin) {
					Emin = E;
					last = p;
					if (m_save) {
						serialiser.Save(".\\CNN_bis.txt", m_nNetwork);
					}
				}
				if (std::abs(E) < 0.0001) {
					learningIsFinish = true;
				}
				else {
					m_nNetwork.Backward(expY, outputA, learningRate);
					p++;
				}
				auto t2 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
				auto tempo = t2.count() - t1.count();
				ttotal += tempo;
				std::cout << " E= " << std::to_string(E) << " Emin= " << std::to_string(Emin) << " lEmin= " << std::to_string(last)
					<< " lr=" << std::to_string(learningRate) << " time= " << std::to_string(tempo)
					<< " ms step " << std::to_string(p) << "/" << std::to_string(nbBatch * nbIter) << " total= " << std::to_string(ttotal / 1000) << "s        ";
			}
		}
		std::cout << std::endl << "100%" << std::endl;
		if (m_save) {
			serialiser.Save(".\\CNN.txt", m_nNetwork);
		}
	}
	if (m_testing) {
		//test
		std::string labelsFile = "t10k-labels.idx1-ubyte";
		std::string imagesFile = "t10k-images.idx3-ubyte";
		std::vector<Matrix<double>> expectedY;
		std::vector<Matrix<double>> dataImg;
		std::cout << "testing:" << std::endl;
		std::cout << "	read data testing:" << std::endl;
		BuildData(labelsFile, imagesFile, dataPath, expectedY, dataImg);
		std::cout << "	testing with " << expectedY.size() << " data:" << std::endl;

		std::vector<Matrix<double>> input = dataImg;
		std::vector<Matrix<double>> expY(1);
		std::vector<Matrix<double>> outputA;
		expY[0].Concat(expectedY, AxisFormat::COL);

		m_nNetwork.Forward(input, outputA);
		Matrix<double> result = outputA[0].applyFunction([](double x) {
			return x >= 0.5 ? 1.0 : 0.0;});

		for (std::size_t i = 0; i < opeConv->NbKernel(); i++) {
			std::cout << "kernel" << std::to_string(i) << "=" << std::endl;
			opeConv->GetKernel(i).printMatrix();
			std::cout << "biais= " << std::to_string(opeConv->GetBiais(i)) << std::endl;
		}
		for (std::size_t i = 0; i < opeConv2->NbKernel(); i++) {
			std::cout << "kernel" << std::to_string(i) << "=" << std::endl;
			opeConv2->GetKernel(i).printMatrix();
			std::cout << "biais= " << std::to_string(opeConv2->GetBiais(i)) << std::endl;
		}

		std::vector<char> cR(result.getCols(), 'E');
		std::vector<char> cY(result.getCols());
		uint32_t error = 0;

		for (uint32_t r = 0; r < result.getRows(); r++) {
			for (uint32_t c = 0; c < result.getCols(); c++) {
				if (result.val(c, r) > 0) {
					cR[c] = char('0' + r);
				}
				if (expY[0].val(c, r) > 0) {
					cY[c] = char('0' + r);
				}
			}
		}
		std::cout << "compare = ";
		for (std::size_t i = 0; i < cR.size(); i++) {
			std::cout << cR[i];
			if (cR[i] == cY[i])
				std::cout << "=";
			else {
				std::cout << "!=";
				error++;
			}
			std::cout << cY[i] << " ";
		}
		std::cout << std::endl;

		std::cout << "errors = " << std::to_string(error) << "/" << std::to_string(result.getCols()) << std::endl;
		std::cout << "errors% = " << std::to_string(double(error) / (double)result.getCols() * 100) << std::setprecision(1) << " %" << std::endl;
	}
}

void PerceptronT(bool batch = true) {
	CRandom<double>::getInstance().InitDistribution(0.0, 10.0);
	std::vector<double> d(2000 * 2);
	CRandom<double>::getInstance().getRandVector(d);
	Matrix<double> data(2000, 2, d);
	std::vector<double> line = { -1.0, 10.0 };
	//std::vector<double> line = { 1.0, 0.0 };
	//std::vector<double> line = { 0.0, 5.0 };

	CRandom<double>::getInstance().InitDistribution(-1.0, 1.0);


	NNetwork m_nNetwork("Perceptron1_Network", DMSE_Matrix<double>);
	std::shared_ptr<OperatorPerceptron<double>> opePerceptron = std::make_shared<OperatorPerceptron<double>>(2, 1);
	std::shared_ptr<OperatorFunction<double>> opeSigmoid = std::make_shared<OperatorFunction<double>>(sigmoidStable<double>, DSigmoid<double>);
	std::shared_ptr<OperatorFunction<double>> opeReLu = std::make_shared<OperatorFunction<double>>(Relu<double>, DRelu<double>);
	std::shared_ptr<OperatorFunction<double>> opeInterupt = std::make_shared<OperatorFunction<double>>(interuptor<double>, Dinteruptor<double>);
	std::shared_ptr<OperatorFunction<double>> opeTanh = std::make_shared<OperatorFunction<double>>(tanh<double>, Dtanh<double>);

	m_nNetwork.AddLayer(std::shared_ptr<Layer<double>>(std::make_shared<Layer<double>>(opePerceptron, "perceptron")));
	m_nNetwork.AddLayer(std::shared_ptr<Layer<double>>(std::make_shared<Layer<double>>(opeSigmoid, "activation")));

	// training
	std::vector<double> Y;
	uint32_t trainingSize = 1000;
	std::vector<Matrix<double>> input;
	std::vector<Matrix<double>> outputA;
	std::vector<Matrix<double>> dOutputA;
	input.push_back(Matrix<double>(trainingSize, 2));
	for (uint32_t index = 0; index < trainingSize; index++) {
		input[0].val(index, 0) = data.val(index, 0);
		input[0].val(index, 1) = data.val(index, 1);
		bool y = topLine(line[0], line[1], data.val(index, 0), data.val(index, 1)) ? 1.0 : 0.0;
		Y.push_back(y);

		if (y) {
			DataExporter::getInstance().AddData("data", "X1", data.val(index, 0));
			DataExporter::getInstance().AddData("data", "Y1", data.val(index, 1));
		}
		else {
			DataExporter::getInstance().AddData("data", "X0", data.val(index, 0));
			DataExporter::getInstance().AddData("data", "Y0", data.val(index, 1));
		}
	}
	std::vector<Matrix<double>> expectedY;
	expectedY.push_back(Matrix<double>(uint32_t(Y.size()), 1, Y));

	bool learningIsFinish = false;
	if(!batch) {
		for (std::size_t iter = 0; iter < 100 && !learningIsFinish; iter++) {
			for (uint32_t i = 0; i < trainingSize; i++) {
				std::vector<Matrix<double>> input1;
				std::vector<Matrix<double>> outputA1;
				std::vector<Matrix<double>> expectedY1;
				input1.push_back(Matrix<double>(1, 2));
				input1[0].val(0, 0) = input[0].val(i, 0);
				input1[0].val(0, 1) = input[0].val(i, 1);
				m_nNetwork.Forward(input1, outputA1, true);
				std::vector<double> Y1;
				Y1.push_back(Y[i]);
				expectedY1.push_back(Matrix<double>(uint32_t(Y1.size()), 1, Y1));
				//cost function 
				double E = MSE(expectedY1[0], outputA1[0]);
				DataExporter::getInstance().AddData("perceptron", "MSE", E);
				if (std::abs(E) < 0.01) {
					learningIsFinish = true;
				}
				else {
					m_nNetwork.Backward(expectedY1, outputA1, 0.001);
				}
			}
		}
	} else {
		for (std::size_t iter = 0; iter < 1000 && !learningIsFinish; iter++) {

			m_nNetwork.Forward(input, outputA, true);

			//cost function 
			double E = MSE(expectedY[0], outputA[0]);

			DataExporter::getInstance().AddData("perceptron", "MSE", E);

			if (std::abs(E) < 0.01) {
				learningIsFinish = true;
			}
			else {
				m_nNetwork.Backward(expectedY, outputA, 1.0);
			}
		}
	}

	//test
	input.clear();
	Y.clear();
	uint32_t testSize = 1000;
	input.push_back(Matrix<double>(testSize, 2));
	for (uint32_t index = 0; index < testSize; index++) {
		input[0].val(index, 0) = data.val(trainingSize +index, 0);
		input[0].val(index, 1) = data.val(trainingSize +index, 1);
		Y.push_back(topLine(line[0], line[1], data.val(trainingSize +index, 0), data.val(trainingSize +index, 1)) ? 1.0 : 0.0);
	}
	expectedY.clear();
	expectedY.push_back(Matrix<double>(uint32_t(Y.size()), 1, Y));



	m_nNetwork.Forward(input, outputA);
	outputA[0].printMatrix();
	expectedY[0].printMatrix();

	Matrix<double> result = outputA[0].applyFunction([](double x) {
		return x >= 0.5 ? 1.0 : 0.0;});
	result.printMatrix();

	uint32_t error = 0;
	for (std::size_t i = 0; i < result.getData().size(); i++) {
		if (result.getData()[i] != expectedY[0].getData()[i]) {
			error++;
		}
	}
	DataExporter::getInstance().ExportToCVS("perceptron", ".\\debugPerceptron.csv");
	DataExporter::getInstance().ExportToCVS("data", ".\\debugData.csv");
	std::cout << "errors = " << std::to_string(error) << std::endl;
	std::cout << "errors% = " << std::to_string(double(error)/ (double)result.getData().size() * 100)<< std::setprecision(1)  << " %" << std::endl;


	Serialize serialiser;
	serialiser.Save(".\\perceptron1.txt", m_nNetwork);
}

void Perceptron3_1() {

	CRandom<double>::getInstance().InitDistribution(0.0, 10.0);
	std::vector<double> d(20000 * 2);
	CRandom<double>::getInstance().getRandVector(d);
	Matrix<double> data(20000, 2, d);
	std::vector<double> line = { 0.5, -5.0, 0.0 , 1,0};

	CRandom<double>::getInstance().InitDistribution(-1.0, 1.0);


	NNetwork m_nNetwork("Perceptron3_1_Network", DMSE_Matrix<double>);
	std::shared_ptr<OperatorPerceptron<double>> opePerceptron1 = std::make_shared<OperatorPerceptron<double>>(2, 3);
	std::shared_ptr<OperatorPerceptron<double>> opePerceptron2 = std::make_shared<OperatorPerceptron<double>>(3, 1);
	std::shared_ptr<OperatorFunction<double>> opeSigmoid = std::make_shared<OperatorFunction<double>>(sigmoidStable<double>, DSigmoid<double>);

	m_nNetwork.AddLayer(std::shared_ptr<Layer<double>>(std::make_shared<Layer<double>>(opePerceptron1, "perceptron1")));
	m_nNetwork.AddLayer(std::shared_ptr<Layer<double>>(std::make_shared<Layer<double>>(opeSigmoid, "activation1")));
	m_nNetwork.AddLayer(std::shared_ptr<Layer<double>>(std::make_shared<Layer<double>>(opePerceptron2, "perceptron2")));
	m_nNetwork.AddLayer(std::shared_ptr<Layer<double>>(std::make_shared<Layer<double>>(opeSigmoid, "activation2")));


	// training
	std::vector<double> Y;
	uint32_t trainingSize = 1000;
	std::vector<Matrix<double>> input;
	std::vector<Matrix<double>> outputA;
	input.push_back(Matrix<double>(trainingSize, 2));
	for (uint32_t index = 0; index < trainingSize; index++) {
		input[0].val(index, 0) = data.val(index, 0);
		input[0].val(index, 1) = data.val(index, 1);
		bool y = topParabole(line[0], line[1], line[2], line[3], data.val(index, 0), data.val(index, 1)) ? 1.0 : 0.0;
		Y.push_back(y);
 	}
	std::vector<Matrix<double>> expectedY;
	expectedY.push_back(Matrix<double>(uint32_t(Y.size()), 1, Y));
	std::shared_ptr<OperatorNode<double>> opLayerPer3 = m_nNetwork.m_layers[0]->m_operator->copy();
	std::shared_ptr<OperatorNode<double>> opLayerPer1 = m_nNetwork.m_layers[2]->m_operator->copy();

	//training
	bool learningIsFinish = false;
	std::size_t last = 0;
	double Emin = 1.0;
	std::cout << "training:" << std::endl;
	for (std::size_t iter = 0; iter < 10000 && !learningIsFinish; iter++) {
		std::cout << "\r";
		if (iter % 100 == 0) {
			std::cout << std::to_string(iter / 100) << "%";
		}
		m_nNetwork.Forward(input, outputA, true);

		//cost function 
		double E = MSE(expectedY[0], outputA[0]);

		if (E < Emin) {
			opLayerPer3 = m_nNetwork.m_layers[0]->m_operator->copy();
			opLayerPer1 = m_nNetwork.m_layers[2]->m_operator->copy();
			Emin = E;
			last = iter;
		}

		if (std::abs(E) < 0.01) {
			learningIsFinish = true;
		}
		else {
			m_nNetwork.Backward(expectedY, outputA, 1.0);
		}
	}
	std::cout << "\r100%" << std::endl;

	m_nNetwork.m_layers[0]->m_operator = opLayerPer3->copy();
	m_nNetwork.m_layers[2]->m_operator = opLayerPer1->copy();

	Serialize serialiser;
	serialiser.Save(".\\perceptro3_1.txt", m_nNetwork);

	//test
	input.clear();
	Y.clear();
	uint32_t testSize = 1000;
	input.push_back(Matrix<double>(testSize, 2));
	for (uint32_t index = 0; index < testSize; index++) {
		input[0].val(index, 0) = data.val(trainingSize + index, 0);
		input[0].val(index, 1) = data.val(trainingSize + index, 1);
		Y.push_back(topParabole(line[0], line[1], line[2], line[3], data.val(trainingSize + index, 0), data.val(trainingSize + index, 1)) ? 1.0 : 0.0);
	}
	expectedY.clear();
	expectedY.push_back(Matrix<double>(uint32_t(Y.size()), 1, Y));

	m_nNetwork.Forward(input, outputA);
	std::cout << "outputA=" << std::endl;
	outputA[0].printMatrix();
	std::cout << "expectedY=" << std::endl;
	expectedY[0].printMatrix();

	Matrix<double> result = outputA[0].applyFunction([](double x) {
		return x >= 0.5 ? 1.0 : 0.0;});
	std::cout << "result=" << std::endl;
	result.printMatrix();

	uint32_t error = 0;
	for (std::size_t i = 0; i < result.getData().size(); i++) {
		if (result.getData()[i] != expectedY[0].getData()[i]) {
			error++;
		}
	}
	std::cout << "errors = " << std::to_string(error) << std::endl;
	std::cout << "errors% = " << std::to_string(double(error) / (double)result.getData().size() * 100) << std::setprecision(1) << " %" << std::endl;
}

void PerceptronChar() {

	uint32_t imgC = 28, imgR = 28;
	uint32_t nbNi = 100, nbNO = 10;
	uint16_t batchSize = 1000;
	bool m_training = true;
	bool m_testing = true;
	CRandom<double>::getInstance().InitDistribution(-1.0, 1.0);

	NNetwork m_nNetwork("PerceptronChar_Network", DMSE_Matrix<double>);
	std::shared_ptr<OperatorFlatten<double, FlattenFormat::COL>> opeFlatten = std::make_shared <OperatorFlatten<double, FlattenFormat::COL>>();
	std::shared_ptr<OperatorConcat<double, FlattenFormat::COL>> opeConcat = std::make_shared <OperatorConcat<double, FlattenFormat::COL>>();
	std::shared_ptr<OperatorPerceptron<double>> opePerceptron784 = std::make_shared<OperatorPerceptron<double>>(imgC * imgR, imgC * imgR);
	std::shared_ptr<OperatorPerceptron<double>> opePerceptron100 = std::make_shared<OperatorPerceptron<double>>(imgC * imgR, nbNi);
	std::shared_ptr<OperatorPerceptron<double>> opePerceptronOut = std::make_shared<OperatorPerceptron<double>>(nbNi, nbNO);
	std::shared_ptr<OperatorFunction<double>> opeSigmoid = std::make_shared<OperatorFunction<double>>(sigmoidStable<double>, DSigmoid<double>);

	m_nNetwork.AddLayer(std::shared_ptr<Layer<double>>(std::make_shared<Layer<double>>(opeFlatten, "flatten")));
	m_nNetwork.AddLayer(std::shared_ptr<Layer<double>>(std::make_shared<Layer<double>>(opeConcat, "concatCol")));
	m_nNetwork.AddLayer(std::shared_ptr<Layer<double>>(std::make_shared<Layer<double>>(opePerceptron784, "perceptron784")));
	m_nNetwork.AddLayer(std::shared_ptr<Layer<double>>(std::make_shared<Layer<double>>(opeSigmoid, "activation")));
	m_nNetwork.AddLayer(std::shared_ptr<Layer<double>>(std::make_shared<Layer<double>>(opePerceptron100, "perceptron10")));
	m_nNetwork.AddLayer(std::shared_ptr<Layer<double>>(std::make_shared<Layer<double>>(opeSigmoid, "activation")));
	m_nNetwork.AddLayer(std::shared_ptr<Layer<double>>(std::make_shared<Layer<double>>(opePerceptronOut, "perceptronOut")));
	m_nNetwork.AddLayer(std::shared_ptr<Layer<double>>(std::make_shared<Layer<double>>(opeSigmoid, "activation")));

	if (m_training) {
		std::string labelsFile = "train-labels.idx1-ubyte";
		std::string imagesFile = "train-images.idx3-ubyte";

		std::vector<Matrix<double>> expectedY;
		std::vector<Matrix<double>> dataImg;
		std::cout << "training:" << std::endl;
		std::cout << "	read data training:" << std::endl;
		BuildData(labelsFile, imagesFile, dataPath, expectedY, dataImg);

		bool learningIsFinish = false;
		std::cout << "	training with " << expectedY.size() << " data:" << std::endl;
		std::size_t nbIter = 350;
		uint64_t ttotal = 0;
		uint64_t p = 0;
		double Emin = 1.0;
		size_t last = 0;
		double leraningRate = 1.0;

		std::shared_ptr<OperatorNode<double>> opLayerCon784 = m_nNetwork.m_layers[2]->m_operator->copy();
		std::shared_ptr<OperatorNode<double>> opLayerPer100 = m_nNetwork.m_layers[4]->m_operator->copy();
		std::shared_ptr<OperatorNode<double>> opLayerPer10 = m_nNetwork.m_layers[6]->m_operator->copy();
		std::size_t nbBatch = dataImg.size() / batchSize;
		for (std::size_t iter = 0; iter < nbIter && !learningIsFinish; iter++) {
			for (std::size_t b = 0; b < nbBatch && !learningIsFinish; b++) {

				auto t1 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
				std::cout << "\r";
				std::cout <<  std::to_string(double(p)/ double(nbBatch * nbIter) * 100.0) <<"%";
				std::vector<Matrix<double>> input;
				std::vector<Matrix<double>> eY;
				std::vector<Matrix<double>> expY(1);
				std::vector<Matrix<double>> outputA;
				input.insert(input.end(), dataImg.begin() + b * batchSize, dataImg.begin() + b * batchSize + batchSize);
				eY.insert(eY.end(), expectedY.begin() + b * batchSize, expectedY.begin() + b * batchSize + batchSize);
				expY[0].Concat(eY, AxisFormat::COL);

				m_nNetwork.Forward(input, outputA, true);

				//cost function 
				double E = MSE(expY[0], outputA[0]);
				if (E < Emin) {
					opLayerCon784 = m_nNetwork.m_layers[2]->m_operator->copy();
					opLayerPer100 = m_nNetwork.m_layers[4]->m_operator->copy();
					opLayerPer10 = m_nNetwork.m_layers[6]->m_operator->copy();
					Emin = E;
					last = p;
				}
				if (std::abs(E) < 0.016) {
					learningIsFinish = true;
				}
				else {
					if (p == 6000) leraningRate /= 2.0;
					if (p == 8000) leraningRate /= 2.0;
					m_nNetwork.Backward(expY, outputA, leraningRate);
					p++;
				}
				auto t2 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
				auto tempo = t2.count() - t1.count();
				ttotal += tempo;
				std::cout << " E= " << std::to_string(E) << " Emin= " << std::to_string(Emin) << " lastEmin = " << std::to_string(last)
					<< " lr=" << std::to_string(leraningRate) << " time= " << std::to_string(tempo)
					<< " ms step " << std::to_string(p) << "/" << std::to_string(nbBatch * nbIter) << " total= " << std::to_string(ttotal / 1000) << "s        ";
			}
		}
		std::cout << "\r100%" << std::endl;

		Serialize serialiser;
		serialiser.Save(".\\perceptronChar.txt", m_nNetwork);

		m_nNetwork.m_layers[2]->m_operator = opLayerCon784->copy();
		m_nNetwork.m_layers[4]->m_operator = opLayerPer100->copy();
		m_nNetwork.m_layers[6]->m_operator = opLayerPer10->copy();

		serialiser.Save(".\\perceptronCharBis.txt", m_nNetwork);
	}
	if(m_testing) {
		//test
		std::string labelsFile = "t10k-labels.idx1-ubyte";
		std::string imagesFile = "t10k-images.idx3-ubyte";
		std::vector<Matrix<double>> expectedY;
		std::vector<Matrix<double>> dataImg;
		std::cout << "testing:" << std::endl;
		std::cout << "	read data testing:" << std::endl;
		BuildData(labelsFile, imagesFile, dataPath, expectedY, dataImg);
		std::cout << "	testing with " << expectedY.size() << " data:" << std::endl;

		std::vector<Matrix<double>> input = dataImg;
		std::vector<Matrix<double>> expY(1);
		std::vector<Matrix<double>> outputA;
		expY[0].Concat(expectedY, AxisFormat::COL);

		m_nNetwork.Forward(input, outputA);
		Matrix<double> result = outputA[0].applyFunction([](double x) {
			return x >= 0.5 ? 1.0 : 0.0;});
		std::cout << "result=" << std::endl;
		result.printMatrix();


		std::vector<char> cR(result.getCols(), 'E');
		std::vector<char> cY(result.getCols());
		uint32_t error = 0;

		for (uint32_t r = 0; r < result.getRows(); r++) {
			for (uint32_t c = 0; c < result.getCols(); c++) {
				if (result.val(c, r) > 0) {
					cR[c] = char('0' + r);
				}
				if (expY[0].val(c, r) > 0) {
					cY[c] = char('0' + r);
				}
			}
		}
		std::cout << "compare = ";
		for (std::size_t i = 0; i < cR.size(); i++) {
			std::cout << cR[i];
			if (cR[i] == cY[i])
				std::cout << "=";
			else {
				std::cout << "!=";
				error++;
			}
			std::cout << cY[i] << " ";
		}
		std::cout << std::endl;

		std::cout << "errors = " << std::to_string(error) << "/" << std::to_string(result.getCols())<< std::endl;
		std::cout << "errors% = " << std::to_string(double(error) / (double)result.getCols() * 100) << std::setprecision(1) << " %" << std::endl;
	}
}


int main(int argc, char* argv[])
{
	std::cout << "V20260306" << std::endl;
	try {
		CNN();
		//PerceptronT(true);
		//Perceptron3_1();
		//PerceptronChar();
	}
	catch (std::ios_base::failure& e) {
		std::cout << "std::ios_base::failure : " << e.what() << '\n';
	}
	catch (std::range_error& re) {
		std::cout << "std::range_error : " << re.what() << '\n';
	}
	catch (...) {
		std::cout << "unknow error" << '\n';
	}
	return 0;
}