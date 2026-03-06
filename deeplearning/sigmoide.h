#pragma once

#include <iostream>
#include <cmath>
#include <vector>
#include <limits>

// Function to calculate the sigmoid of a given value
//simple activation function
template<class T>
T sigmoid(T x) {
    return 1.0 / (1.0 + std::exp(-x));
}

//derivative of activation function
// x = sigmoid(input);
template<class T>
inline T DSigmoid(T x)
{
    return (x * (1 - x));
}

template<class T>
T sigmoidStable(T x)
{
    T result;
    if (x >= 45.0) result = 1.;
    else if (x <= -45.0) result = 0.;
    else result = 1. / (1. + exp(-x));
    return result;
}

// Function to apply ReLU to a single value
template<class T>
T Relu(T x) {
    return std::max(T(0), x);
}

template<class T>
T DRelu(T x) {
    return x > T(0) ? T(1) : T(0);
}

template<class T>
T interuptor(T x) {
    return x > T(0) ? T(1) : T(0);
}

template<class T>
T Dinteruptor(T x) {
    return x > T(0) ? T(0) : T(0);
}

template<class T>
T tanh(T x) {
    if (x >= 45.0) return 1.;
    else if (x <= -45.0) return -1.;
    return (1-std::exp(2*x)/ (1 + std::exp(2*x)));
}

template<class T>
T Dtanh(T x) {
    return (1 - (tanh(x)*tanh(x)));
}


//constexpr  double min_ = std::numeric_limits<double>::min();
template<typename T> 
constexpr  T min_ = std::numeric_limits<T>::min();

// Function to calculate log loss (cost function)
double logLoss(const std::vector<double>& predicted, const std::vector<double>& actual) {
    double loss = 0.0;
    std::size_t n = actual.size();
    for (size_t i = 0; i < n; ++i) {
        // Ensure predicted value is between 0 and 1
        double p = std::max(std::min(predicted[i], 1.0 - 1e-15), 1e-15);
        loss -= (actual[i] * std::log(p+ min_<double>) + (1 - actual[i]) * std::log((1 - p)+ min_<double>));
    }
    return loss / double(n);
}

// Function to calculate log loss (cost function)
double logLoss(const Matrix<double>& expectedY, const Matrix<double>& actual) {
    double loss = 0.0;
    for (uint32_t c = 0; c < actual.getCols(); ++c) {
        for (uint32_t r = 0; r < actual.getRows(); ++r) {
            // Ensure predicted value is between 0 and 1
            double p = std::max(std::min(expectedY.val(c,r), 1.0 - 1e-15), 1e-15);
            loss -= (actual.val(c, r) * std::log(p + min_<double>) + (1 - actual.val(c, r)) * std::log((1 - p) + min_<double>));
        }
    }
    return loss / double(actual.getData().size());
}

double logLoss(const std::vector<Matrix<double>>& expectedY, const std::vector<Matrix<double>>& actual) {
    double loss = 0.0;
    for (std::size_t i = 0; i < actual.size(); i++) {
        loss -= logLoss(expectedY[i], actual[i]);
    }
    return loss / double(actual.size());
}


double MSE(const Matrix<double>& expectedY, const Matrix<double>& actual) {
    double mse = 0.0;
    Matrix<double> err = actual - expectedY;
    for (std::size_t i = 0; i < err.getData().size(); i++) {
        mse += err.getData()[i]* err.getData()[i];
    }
    return mse/double(err.getData().size());
}


double MSE(const std::vector<Matrix<double>>& expectedY, const std::vector<Matrix<double>>& actual) {
    double mse = 0.0;
    for (std::size_t i = 0; i < actual.size(); i++) {
        mse += MSE(expectedY[i], actual[i]);
    }
    return mse / double(actual.size());
}

template<class T>
std::vector <Matrix<T>> DMSE_Matrix(const std::vector<Matrix<T>>& expectedY, const std::vector<Matrix<T>>& outputA)
{
    std::vector<Matrix<T>> dOutputA(expectedY.size());
    for (std::size_t i = 0; i < dOutputA.size(); i++) {
        //mse_prime= 2 * (y_pred - y_true) / np.size(y_true)
        uint32_t s = expectedY[i].getRows();
        dOutputA[i] = outputA[i] - expectedY[i];
        dOutputA[i] = dOutputA[i] * 2. / double(s);
    }
    return dOutputA;
}

double DMSE(const Matrix<double>& expectedY, const Matrix<double>& actual) {
    double mse = 0.0;
    Matrix<double> err = actual - expectedY;
    for (std::size_t i = 0; i < err.getData().size(); i++) {
        mse += err.getData()[i];
    }
    return 2.0 * mse / double(err.getData().size());
}


double DMSE(const std::vector<Matrix<double>>& expectedY, const std::vector<Matrix<double>>& actual) {
    double mse = 0.0;
    for (std::size_t i = 0; i < actual.size(); i++) {
        mse += DMSE(expectedY[i], actual[i]);
    }
    return mse / double(actual.size());
}


//def binary_cross_entropy_prime(y_true, y_pred) :
//return ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / np.size(y_true)
template<class T>
std::vector<Matrix<T>> DCrossEntropyLoss_Matrix(const std::vector<Matrix<T>>& expectedY, const std::vector<Matrix<T>>& outputA) {
    std::vector<Matrix<T>> gradient(expectedY.size());
    for (size_t i = 0; i < expectedY.size(); ++i) {
        gradient[i] = expectedY[i].applyFunction([](const T& a) {return 1.0 - a;}) / outputA[i].applyFunction([](const T& a) {return 1.0 - a;});
        Matrix<T> Y = expectedY[i];
        Matrix<T> O = outputA[i];
        gradient[i] = gradient[i] - (Y / O);
        gradient[i] = gradient[i] / Y.getRows();
    }
    return gradient;
}

// Function to calculate Cross Entropy Loss
// def binary_cross_entropy(y_true, y_pred) :
    //return np.mean(-y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred))
double crossEntropyLoss(const std::vector<double>& predicted, const std::vector<double>& actual) {
    double loss = 0.0;
    for (size_t i = 0; i < predicted.size(); ++i) {
        loss -= actual[i] * std::log(predicted[i]) + (1 - actual[i]) * std::log(1 - predicted[i]);
    }
    return loss / double(predicted.size());
}

// Function to calculate Cross Entropy Loss
std::vector<double> crossEntropyLossV(const std::vector<double>& predicted, const std::vector<double>& actual) {
    std::vector<double> loss(predicted.size());
    for (size_t i = 0; i < predicted.size(); ++i) {
        loss[i] =  actual[i] * std::log(predicted[i]+ 1e-15) + (1 - actual[i]) * std::log(1 - predicted[i]+1e-15);
    }
    return loss;
}

// Function to calculate likelihood of Bernoulli trials
double calculateLikelihood(const std::vector<int>& data, double probability) {
    double likelihood = 1.0;
    for (int outcome : data) {
        if (outcome == 1) {
            likelihood *= probability;
        }
        else {
            likelihood *= (1 - probability);
        }
    }
    return likelihood;
}

// Function to calculate log-likelihood of Bernoulli trials
double calculateLogLikelihood(const std::vector<int>& data, double probability) {
    double logLikelihood = 0.0;
    for (int outcome : data) {
        if (outcome == 1) {
            logLikelihood += std::log(probability);
        }
        else {
            logLikelihood += std::log(1 - probability);
        }
    }
    return logLikelihood;
}
// Softmax forward function
template<class T>
std::vector<T> softmax_forward(const Matrix<T>& input) {
    std::vector<T> output(input.getData().size());
    T maxInput = *std::max_element(input.getData().begin(), input.getData().end()); // For numerical stability

    // Compute exponentials and subtract maxInput for stability
    T sumExp = T(0);
    for (std::size_t i = 0; i < input.getData().size(); ++i) {
        output[i] = std::exp(input.getData()[i] - maxInput);
        if (output[i] < std::numeric_limits<T>::min()*T(100)) {
            output[i] = std::numeric_limits<T>::min() * T(100);
        }
        sumExp += output[i];
    }
    // Normalize by dividing by the sum of exponentials
    for (std::size_t i = 0; i < output.size(); ++i) {
        output[i] /= sumExp;
    }
    return output;
}

// Softmax backward function
template<class T>
Matrix<T> softmax_backward(
    const Matrix<T>& softmax_output,
    const Matrix<T>& target) {

    // Gradient of the loss with respect to the input
    // Gradient for cross-entropy loss: softmax_output - target
    Matrix<T> gradient = softmax_output - target;
    return gradient;
}
