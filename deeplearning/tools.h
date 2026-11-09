#pragma once

#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <vector>
#include <string>
#include <thread>

template <class T>
T maxAbs(const T& a, const T& b) {
	if (std::abs(a) > std::abs(b)) {
		return a;
	}
	else {
		return b;
	}
}

template <class T, class U>
U normalize(T value, T min, T max) {
	return U(U(value - min) / U(max - min));
}

template <class T, class U>
U denormalize(T value, T min,T max) {
	return U((value + min) * (max - min));
}

std::int64_t GetFileSize(std::ifstream & Fichier);


template<class T>
T scalar(const std::vector<T> a, const std::vector<T> b) {
	T ret = T(0);
	if (a.size() == b.size()) {
		for (std::size_t i = 0; i < a.size(); i++) {
			ret += a[i] * b[i];
		}
	}
	else {
		throw std::range_error("scalar: vector size shall be equal "
			+ std::to_string(a.size()) + " != " + std::to_string(b.size()));
	}
	return ret;
} 

template<class T>
bool topLine(T a, T b, T xp, T yp) {
	if (yp - (a * xp + b) > 0) {
		return true;
	}
	return false;

}

template<class T>
bool topParabole(T a, T a2, T b, T c, T xp, T yp) {
	if (yp - (a * (xp +a2) * (xp+a2) + b * xp + c) > 0) {
		return true;
	}
	return false;

}

template <typename Fn, typename... Args>
void ThreadingTemplate(const uint32_t size, Fn&& function, Args&&... args) {
	std::size_t nbThtread = 4;
	std::vector<std::thread> l_thread;
	if (nbThtread > 1) {
		if (size <= nbThtread) {
			for (uint32_t y = 0; y < size; y++) {
				l_thread.push_back(std::thread(function, y, y + 1, std::forward<Args>(args)...));
			}
		}
		else {
			for (uint32_t y = 0; y < nbThtread; y++) {
				uint32_t b = y * size / nbThtread;
				uint32_t e = y * size / nbThtread + size / nbThtread + 1;
				if (y + 1 == nbThtread) e = size;
				l_thread.push_back(std::thread(function, b, e, std::forward<Args>(args)...));
			}
		}
		for (std::size_t i = 0; i < l_thread.size(); i++) {
			l_thread[i].join();
		}
	}
	else
	{
		function(0, size, std::forward<Args>(args)...);
	}
}

std::string intToStringWithZeros(int number, uint8_t width);