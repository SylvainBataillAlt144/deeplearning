#pragma once
#include <vector>
#include <exception>
#include <functional>
#include <stdexcept>
#include <iostream>
#include <utility>
#include <string>
#include "CRandom.h"
#include "Serialize.h"
#include <chrono>
#include <immintrin.h>

struct IndexK
{
	int32_t m_c;
	int32_t m_r;
	int32_t m_w;
};
struct IndexKernel
{
	std::vector<IndexK> m_index;
	std::vector<IndexK> m_indexR;
};
struct Index2D
{
	Index2D(const int32_t& c, const int32_t& r) : m_c(c), m_r(r) {}
	int32_t m_c;
	int32_t m_r;
};
struct IndexImage
{
	std::vector<Index2D> m_index;
};

inline bool isPow2(uint32_t n) {
	return n & (n - 1);
}

inline uint8_t nearPow2plus(uint32_t n) {
	uint8_t res = 1;
	while (n < std::pow(2, res)) {
		res++;
	}
	return res;
}

enum class AxisFormat {
	COL,
	ROW
};
template<class T>
class Matrix
{
public:
	Matrix(uint32_t cols=0, uint32_t rows=0, T init = 0): m_rows(rows), m_cols(cols), m_data(rows* cols, T(init))
	{}
	Matrix(uint32_t size) : m_rows(size), m_cols(size), m_data(size* size, T(0))
	{}
	Matrix(const uint32_t& cols, const uint32_t& rows, std::vector<T> data) : m_rows(rows), m_cols(cols), m_data(rows* cols, T(0)) {
		if (m_rows * m_cols == m_data.size()) {
			m_data = data;
		}
		else {
			throw std::range_error("Matrix::Matrix: data size invalid "
				+ std::to_string(m_cols) + " * " + std::to_string(m_cols) + " != data.size= "
				+ std::to_string(m_data.size()));
		}
	}
	Matrix(const Matrix<T>& m) {
		m_rows = m.m_rows;
		m_cols = m.m_cols;
		m_data = m.m_data;
	}
	Matrix(const Matrix<T>&& m) noexcept  {
		m_rows = m.m_rows;
		m_cols = m.m_cols;
		m_data = m.m_data;
	}
	~Matrix(){}

	inline Matrix<T> &operator=(const Matrix<T>& m) {
		m_rows = m.m_rows;
		m_cols = m.m_cols;
		m_data = m.m_data;
		return (*this);
	}
	inline bool operator==(const Matrix<T>& m) {
		if (m.m_cols != m_cols || m.m_rows != m_rows) {
			throw std::range_error("Matrix::operator==: columns and rows shall be equal in each matrix "
				+ std::to_string(m.m_cols) + " !=  " + std::to_string(m_cols) + " or "
				+ std::to_string(m.m_rows) + " !=  " + std::to_string(m_rows));
		}
		bool output = true;
		for (size_t i = 0; i < m_data.size(); i++) {
			if (std::abs(m_data[i] - m.m_data[i]) > std::numeric_limits<T>::epsilon()*100) {
				output = false;
				break;
			}
		}
		return output;
	}
	inline Matrix<T> operator+(const Matrix<T>& m) {
		if (m.m_cols != m_cols || m.m_rows != m_rows) {
			throw std::range_error("Matrix::operator+: columns and rows shall be equal in each matrix " 
				+ std::to_string(m.m_cols) + " !=  " + std::to_string(m_cols) + " or " 
				+ std::to_string(m.m_rows) + " !=  " + std::to_string(m_rows));
		}
		Matrix<T> output(m_cols, m_rows);
		for (size_t i = 0; i+3 < m_data.size(); i += 4) {
			__m256d va = _mm256_load_pd(&m_data[i]);
			__m256d vb = _mm256_load_pd(&m.m_data[i]);
			__m256d vsum = _mm256_add_pd(va, vb);
			_mm256_store_pd(&output.m_data[i], vsum);
		}
		for (size_t p = m_data.size() - m_data.size() % 4; p < m_data.size(); p++) {
			output.m_data[p] = m_data[p] + m.m_data[p];
		}
		return output;
	}
	inline Matrix<T> operator+(const T& s) const {
		Matrix<T> output(m_cols, m_rows);
		T vecs[4] = { s, s, s, s };
		__m256d vs = _mm256_load_pd(&vecs[0]);
		for (size_t i = 0; i+3 < m_data.size(); i+=4) {
			__m256d va = _mm256_load_pd(&m_data[i]);
			__m256d vsum = _mm256_add_pd(va, vs);
			_mm256_storeu_pd(&output.m_data[i], vsum);
		}
		for (size_t p = m_data.size() - m_data.size() % 4; p < m_data.size(); p++) {
			output.m_data[p] = m_data[p] + s;
		}
		return output;
	}
	inline Matrix<T> operator+(const std::vector<T>& v) const{
		if (v.size() != m_rows) {
			throw std::range_error("Matrix::operator+: rows  shall be equal to vector size "
				+ std::to_string(m_rows) + " != " + std::to_string(v.size()));
		}
		Matrix<T> output(m_cols, m_rows);
		for (size_t c = 0; c < output.getCols(); c++) {
			for (size_t r = 0; r < output.getRows(); r++) {
				output.m_data[r * output.m_cols + c] = m_data[r * output.m_cols + c] + v[r];
			}
		}
		return output;
	}
	inline Matrix<T> operator-(const Matrix<T>& m) const {
		if (m.m_cols != m_cols || m.m_rows != m_rows) {
			throw std::range_error("Matrix::operator-: columns and rows shall be equal in each matrix " 
				+ std::to_string(m.m_cols) + " !=  " + std::to_string(m_cols) + " or "
				+ std::to_string(m.m_rows) + " !=  " + std::to_string(m_rows)); 
		}
		Matrix<T> output(m_cols, m_rows);
		for (size_t i = 0; i + 3 < m_data.size(); i += 4) {
			__m256d va = _mm256_loadu_pd(&m_data[i]);
			__m256d vb = _mm256_loadu_pd(&m.m_data[i]);
			__m256d vsub = _mm256_sub_pd(va, vb);
			_mm256_storeu_pd(&output.m_data[i], vsub);
		}
		for (size_t p = m_data.size() - m_data.size() % 4; p < m_data.size(); p++) {
			output.m_data[p] = m_data[p] - m.m_data[p];
		}
		return output;
	}
	inline Matrix<T> operator-(const T& s) const {
		Matrix<T> output(m_cols, m_rows);
		T vs[4] = { s, s, s, s };
		__m256d vs = _mm256_loadu_pu(vs);
		for (size_t i = 0; i + 3 < m_data.size(); i += 4) {
			__m256d va = _mm256_loadu_pd(&m_data[i]);
			__m256d vsub = _mm256_sub_pd(va, vs);
			_mm256_storeu_pd(&output.m_data[i], vsub);
		}
		for (size_t p = m_data.size() - m_data.size() % 4; p < m_data.size(); p++) {
			output.m_data[p] = m_data[p] - s;
		}
		return output;
	}
	inline Matrix<T> operator*(const Matrix<T>& m) const{
		if (m.m_rows != m_cols) {
			throw std::range_error("Matrix::operator*: columns number shall be equal to rows number of input matrix "
				+ std::to_string(m.m_rows) + " !=  " + std::to_string(m_cols));
		}
		Matrix<T>mT = m.transpose();
		Matrix<T> output(m.m_cols, m_rows);

		auto kerMultAVX = [&output](const uint32_t y0, const uint32_t yn, const Matrix<T>& a, const Matrix<T>& mT) {
			for (uint32_t y = 0; y < output.m_rows; y++) {
				for (uint32_t x = 0; x < output.m_cols; x++) {
					T result = 0, res_val[4] = { 0, 0, 0, 0 };
					__m256d sum_vec = _mm256_setzero_pd();
					for (uint32_t k = 0; k + 3 < a.m_cols; k += 4) {
						//1 appel car les data sont contigues
						__m256d va = _mm256_loadu_pd(&a.val(k, y));
						//1 appel car les data sont contigues grace à la transposée
						__m256d vb = _mm256_loadu_pd(&mT.val(k, x));
						// Multiply and accumulate
						sum_vec = _mm256_fmadd_pd(va, vb, sum_vec);
					}
					_mm256_storeu_pd(res_val, sum_vec);
					result = res_val[0] + res_val[1] + res_val[2] + res_val[3];
					for (uint32_t p = a.m_cols - a.m_cols%4; p < a.m_cols; p++) {
						result += a.val(p, y) * mT.val(p, x);
					}
					output.val(x, y) = result;
				}
			}
		};

		auto kerMult = [&output](const uint32_t y0, const uint32_t yn, const Matrix<T>& a, const Matrix<T>& mT) {
			for (uint32_t y = y0; y < yn; y++) {
				for (uint32_t x = 0; x < output.m_cols; x++) {
					T result = 0;
					for (uint32_t k = 0; k + 3 < a.m_cols; k += 4) {
						result += a.val(k, y) * mT.val(k, x);
						result += a.val(k + 1, y) * mT.val(k + 1, x);
						result += a.val(k + 2, y) * mT.val(k + 2, x);
						result += a.val(k + 3, y) * mT.val(k + 3, x);
					}
					for (uint32_t p = a.m_cols - a.m_cols % 4; p < a.m_cols; p++) {
						result += a.val(p, y) * mT.val(p, x);
					}
					output.val(x, y) = result;
				}
			}
		};
		kerMultAVX(0, output.m_rows, *this, mT);

		//ThreadingTemplate(output.m_rows, kerMult, *this, mT);

		return output;
	}

	inline Matrix<T> multAVX(const Matrix<T>& m)
	{
		if (m.m_rows != m_cols) {
			throw std::range_error("Matrix::multAVX: columns number shall be equal to rows number of input matrix "
				+ std::to_string(m.m_rows) + " !=  " + std::to_string(m_cols));
		}
		Matrix<T>mT = m.transpose();
		Matrix<T> output(m.m_cols, m_rows);
		for (uint32_t y = 0; y < output.m_rows; y++) {
			for (uint32_t x = 0; x < output.m_cols; x++) {
				T result = 0, res_val[4] = { 0, 0, 0, 0 };
				__m256d sum_vec = _mm256_setzero_pd();
				for (uint32_t k = 0; k + 3 < m_cols; k += 4) {
					//1 appel car les data sont contigues
					__m256d va = _mm256_loadu_pd(&val(k, y));
					//1 appel car les data sont contigues grace à la transposée
					__m256d vb = _mm256_loadu_pd(&mT.val(k, x));
					// Multiply and accumulate
					sum_vec = _mm256_fmadd_pd(va, vb, sum_vec);
				}
				_mm256_storeu_pd(res_val, sum_vec);
				result = res_val[0] + res_val[1] + res_val[2] + res_val[3];

				for (uint32_t p = m_cols - m_cols % 4; p < m_cols; p++) {
					result += val(p, y) * mT.val(p, x);
				}
				output.val(x, y) = result;
			}
		}
		return output;
	}

	inline Matrix<T> operator*(const T& s) {
		Matrix<T> output(m_cols, m_rows);
		for (size_t i = 0; i < m_data.size(); i++) {
			output.m_data[i] = m_data[i] * s;
		}
		T vecs[4] = { s, s, s, s };
		__m256d vs = _mm256_load_pd(&vecs[0]);
		for (size_t i = 0; i + 3 < m_data.size(); i += 4) {
			__m256d va = _mm256_load_pd(&m_data[i]);
			__m256d vsum = _mm256_mul_pd(va, vs);
			_mm256_storeu_pd(&output.m_data[i], vsum);
		}
		for (size_t p = m_data.size() - m_data.size() % 4; p < m_data.size(); p++) {
			output.m_data[p] = m_data[p] * s;
		}
		return output;
	}
	inline Matrix<T> operator/(const T& s) {
		Matrix<T> output(m_cols, m_rows);
		for (size_t i = 0; i < m_data.size(); i++) {
			output.m_data[i] = m_data[i] / s;
		}
		T vecs[4] = { s, s, s, s };
		__m256d vs = _mm256_load_pd(&vecs[0]);
		for (size_t i = 0; i + 3 < m_data.size(); i += 4) {
			__m256d va = _mm256_load_pd(&m_data[i]);
			__m256d vdiv = _mm256_div_pd(va, vs);
			_mm256_storeu_pd(&output.m_data[i], vdiv);
		}
		for (size_t p = m_data.size() - m_data.size() % 4; p < m_data.size(); p++) {
			output.m_data[p] = m_data[p] / s;
		}
		return output;
	}

	inline Matrix<T> operator/(const  Matrix<T>& m) {
		if (m.m_cols != m_cols || m.m_rows != m_rows) {
			throw std::range_error("Matrix::operator/: columns and rows shall be equal in each matrix "
				+ std::to_string(m.m_cols) + " !=  " + std::to_string(m_cols) + " or "
				+ std::to_string(m.m_rows) + " !=  " + std::to_string(m_rows));
		}
		Matrix<T> output(m_cols, m_rows);
		for (size_t i = 0; i + 3 < m_data.size(); i += 4) {
			__m256d va = _mm256_load_pd(&m_data[i]);
			__m256d vb = _mm256_load_pd(&m.m_data[i]);
			__m256d vdiv = _mm256_div_pd(va, vb);
			_mm256_store_pd(&output.m_data[i], vdiv);
		}
		for (size_t p = m_data.size() - m_data.size() % 4; p < m_data.size(); p++) {
			output.m_data[p] = m_data[p] + m.m_data[p];
		}
		return output;
	}

	inline T & val(const uint32_t& cols, const uint32_t& rows)
	{
		return (m_data[rows * m_cols + cols]);
	}

	inline const T& val(const uint32_t& cols, const uint32_t& rows) const
	{
		return (m_data[rows * m_cols + cols]);
	}

	inline T & at(const uint32_t& cols, const uint32_t& rows)
	{
		if (cols >= m_cols) {
			throw std::range_error("Matrix::operator[]: columns number input is out of range "
				+ std::to_string(cols) + " >=  " + std::to_string(m_cols));
		}
		if (rows >= m_rows) {
			throw std::range_error("Matrix::operator[]: rows number input is out of range "
				+ std::to_string(rows) + " >=  " + std::to_string(m_rows));
		}
		return m_data[rows * m_cols + cols];
	}

	inline const T& at(const uint32_t& cols, const uint32_t& rows) const
	{
		if (cols >= m_cols) {
			throw std::range_error("Matrix::operator[]: columns number input is out of range "
				+ std::to_string(cols) + " >=  " + std::to_string(m_cols));
		}
		if (rows >= m_rows) {
			throw std::range_error("Matrix::operator[]: rows number input is out of range "
				+ std::to_string(rows) + " >=  " + std::to_string(m_rows));
		}
		return m_data[rows * m_cols + cols];
	}

	inline Matrix<T> transpose() const{
		Matrix<T> output(m_rows, m_cols);
		if ((m_rows == 1) || (m_cols == 1)) {
			output.m_data = m_data;
		}
		else {
			for (std::size_t i = 0; i < m_data.size(); i++) {
				output.m_data[(i / m_cols) + (i % m_cols)*m_rows  ] = m_data[i];
			}
		}
		return output;
	}

	inline Matrix<T> applyFunction(std::function<T(const T&)> func) const {
		Matrix<T> output(m_cols, m_rows);
		for (uint32_t i = 0; i < output.m_data.size(); i++){
			output.m_data[i] = func(m_data[i]);
		}
		return output;
	}

	inline uint32_t getRows() const {
		return m_rows;
	}

	inline uint32_t getCols() const {
		return m_cols;
	}
	inline const std::vector<T> & getData() const {
		return m_data;
	}

	inline const std::vector<T> getData(const AxisFormat axis, uint32_t id) const {
		std::vector<T> out;
		if (axis == AxisFormat::ROW) {
			if (id >= m_rows) {
				throw std::range_error("Matrix::getData: row number shall be inferior to rows numbers: "
					+ std::to_string(id) + " <=  " + std::to_string(m_rows));
			}
			out.insert(out.end(), m_data.begin()+(id*m_cols), m_data.begin()+(id*m_cols+m_cols));
		}
		if (axis == AxisFormat::COL) {
			if (id >= m_cols) {
				throw std::range_error("Matrix::getData: column number shall be inferior to cols numbers: "
					+ std::to_string(id) + " <=  " + std::to_string(m_cols));
			}
			for (uint32_t r = 0; r < m_rows; r++) {
				out.push_back(val(id, r));
			}
		}
		return out;
	}

	template <class U = T> 
	void printMatrix()
	{
		for (uint32_t r = 0; r < getRows(); r++) {
			std::cout << "[";
			for (uint32_t c = 0; c < getCols(); c++) {
				std::cout << U(val(c, r)) << " ";
			}
			std::cout << "]" << std::endl;
		}
	}

	std::vector<T> Flatten() {
		return m_data;
	}

	void initRand() {
		CRandom<T>::getInstance().getRandVector(m_data);
	}

	std::vector<T> SumAxis(const AxisFormat & axis) const {
		std::vector<T> output;
		if (axis == AxisFormat::ROW) {
			output.resize(m_rows);
			for (std::size_t i = 0; i < m_data.size();  i++) {
				output[i / m_cols] += m_data[i];
			}

		}
		else if (axis == AxisFormat::COL) {
			output.resize(m_cols);
			for (std::size_t i = 0; i < m_data.size();  i++) {
				output[i % m_cols] += m_data[i];
			}
		}
		return output;
	}
	T SumElm() const{
		T output = 0;
		for (std::size_t i = 0; i < m_data.size(); i++) {
			output += m_data[i];
		}
		return output;
	}

	void MultElm(const Matrix<T> & m) {
		if ((m_rows != m.m_rows) || (m_cols != m.m_cols)) {
			throw std::range_error("Matrix::MultElm[]: rows or cols of m and this shall be equal "
				+ std::to_string(m.m_rows) + " !=  " + std::to_string(m_rows) + " or " +std::to_string(m.m_cols) + " !=  " + std::to_string(m_cols));
			return;
		}
		for (uint32_t i = 0; i < m.m_data.size(); i++) {
			m_data[i] *= m.m_data[i];
		}
	}

	void Concat(const std::vector<Matrix<T>> m, const AxisFormat& axis) {
		if (m.empty()) {
			return;
		}
		std::vector<T> d;
		if (axis == AxisFormat::ROW) {
			for (std::size_t i = 0; i < m.size(); i++) {
				if (m[i].m_rows != m[0].m_rows || m[i].m_cols != m[0].m_cols) {
					throw std::range_error("Matrix::Concat : rows and cols of each matrix shall be equal ");
				}
				d.insert(d.end(), m[i].m_data.begin(), m[i].m_data.end());
			}
		}
		else {
			for (uint32_t j = 0; j < m[0].getRows(); j++) {
				for (std::size_t i = 0; i < m.size(); i++) {
					std::vector<T> v;
					v = m[i].getData(AxisFormat::ROW, j);
					d.insert(d.end(), v.begin(), v.end());
				}
			}
		}
		if (axis == AxisFormat::COL) {
			m_rows = m[0].m_rows;
			m_cols = m[0].m_cols * m.size();
		}
		else {
			m_rows = m[0].m_rows * m.size();
			m_cols = m[0].m_cols;
		}
		m_data = d;
	}

	T Max() {
		T max = T(0);
		for (std::size_t i = 0; i < m_data.size(); i++) {
			if (m_data[i] > max) {
				max = m_data[i];
			}
		}
		return max;
	}
	std::size_t MaxIndex() {
		std::size_t max = 0;
		T dMax = T(0);
		for (std::size_t i = 0; i < m_data.size(); i++) {
			if (m_data[i] > dMax) {
				max = i;
				dMax = m_data[i];
			}
		}
		return max;
	}

	bool empty() {
		return m_data.empty();
	}

	static std::vector<Matrix<T>> Convolution(const Matrix<T> & input, const std::uint32_t size, const std::size_t number, const std::vector<T> & kernel,
		const std::vector<IndexK>& index, const std::vector<Index2D> & indexI, const std::vector<T> & biais, const uint32_t offset = 0) {
		std::vector<Matrix<T>> out;
		T v = 0;
		for (std::size_t i = 0; i < number; i++) {
			out.push_back(Matrix<T>(input.getCols(), input.getRows(), T(0)));
		}
		for (std::size_t i = 0; i < number; i++) {
			for (std::size_t indx = 0; indx < indexI.size(); indx++) {
				v = 0;
				for (size_t j = 0; j+3 < size * size; j+=4) {
					T x1 = input.val(indexI[indx].m_c + index[j].m_c, indexI[indx].m_r + index[j].m_r);
					T x2 = input.val(indexI[indx].m_c + index[j+1].m_c, indexI[indx].m_r + index[j+1].m_r);
					T x3 = input.val(indexI[indx].m_c + index[j+2].m_c, indexI[indx].m_r + index[j+2].m_r);
					T x4 = input.val(indexI[indx].m_c + index[j+3].m_c, indexI[indx].m_r + index[j+3].m_r);
					T k1 = kernel[(i * size * size) + index[j].m_w];
					T k2 = kernel[(i * size * size) + index[j+1].m_w];
					T k3 = kernel[(i * size * size) + index[j+2].m_w];
					T k4 = kernel[(i * size * size) + index[j+3].m_w];
					T r = x1*k1 + x2*k2 + x3*k3 + x4*k4 ;
					v += r;
				}
				for (size_t j = size*size - size*size % 4; j < size*size; j++) {
					T x = input.val(indexI[indx].m_c + index[j].m_c, indexI[indx].m_r + index[j].m_r);
					T k = kernel[(i * size * size) + index[j].m_w];
					v += x * k;
				}
				out[i].val(indexI[indx].m_c, indexI[indx].m_r) = biais[i] + v;
			}
		}
		return out;
	}

	Matrix<T> rot180()
	{
		Matrix<T> output(m_cols, m_rows);
		for (std::size_t i = 0; i < m_data.size(); i++) {
			output.m_data[i] = m_data[m_data.size()-1 - i];
		}
		return output;
	}

	Matrix<T> Extract(uint32_t col, uint32_t row, uint32_t cols, uint32_t rows) {
		std::vector<T> ret;
		if (col + cols > m_cols) {
			throw std::range_error("Matrix::Extract*: col + cols shall be <= m_cols:  "
				+ std::to_string(col) + " + " + std::to_string(cols) + " > " + std::to_string(m_cols));
		}
		if (row + rows > m_rows) {
			throw std::range_error("Matrix::Extract*: row + rows shall be <= m_rows:  "
				+ std::to_string(row) + " + " + std::to_string(rows) + " > " + std::to_string(m_rows));
		}
		for (uint32_t r = 0; r < rows; r++) {
			ret.insert(ret.end(), m_data.begin() + ((row+r) * m_cols + col), m_data.begin() + ((row+r) * m_cols + col + cols));
		}
		return Matrix<T>(cols, rows, ret);
	}

	friend std::ostream& operator <<(std::ostream& os, const Matrix<T>& data) {
		os << "[" << data.getCols() << "]";
		os << "[" << data.getRows() << "]";
		os << data.getData();
		return os;
	}

	friend std::istream& operator >>(std::istream& is, Matrix<T>& data) {
		char brace;
		uint32_t cols;
		uint32_t rows;
		std::vector<T> vec;
		is >> brace;
		is >> cols;
		is >> brace;
		is >> brace;
		is >> rows;
		is >> brace;
		is >> vec;
		data = Matrix<T>(cols, rows, vec);
		return is;
	}


protected:
	uint32_t m_rows;
	uint32_t m_cols;
	std::vector<T> m_data;
};
template<class T>
inline Matrix<T> operator-(const T& s, const Matrix<T> & m){
	Matrix<T> output(m.m_cols, m.m_rows);
	for (size_t i = 0; i < m.m_data.size(); i++) {
		output.m_data[i] = s - m.m_data[i];
	}
	return output;
}

template<class T>
inline Matrix<T> operator+(const T& s, const Matrix<T>& m){
	Matrix<T> output(m.m_cols, m.m_rows);
	for (size_t i = 0; i < m.m_data.size(); i++) {
		output.m_data[i] = s + m.m_data[i];
	}
	return output;
}