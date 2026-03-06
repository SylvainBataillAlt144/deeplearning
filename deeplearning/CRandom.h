#pragma once

#include "Singleton.h"

#include <random>
#include <type_traits>
#include <limits>

template<class T>
class CRandom : public Singleton<CRandom<T>>
{
    friend class Singleton<CRandom<T>>;
private:
    CRandom() : m_rd(), m_isInit(false)
    {
    }
    ~CRandom() {}

    std::random_device m_rd;
    std::default_random_engine m_eng;

    std::uniform_real_distribution<double> m_distr;
    bool m_isInit;

public:
    template<class T>
    void InitDistribution(T min, T max) {
        m_eng = std::default_random_engine(m_rd());
        m_distr = std::uniform_real_distribution<T>(min, max);
        m_isInit = true;
    }
    //template<class T = std::numeric_limits<T>::is_integer>
    //void InitDistribution(T min, T max) {
    //    m_eng = std::default_random_engine(m_rd());
    //        m_distr = std::uniform_int_distribution<T>(min, max);
    //    }
    template<class T>
    T getRand() {
            if (!m_isInit) {
                m_isInit = true;
                m_distr = std::uniform_real_distribution<T>(T(0), T(1));
            }
        return m_distr(m_eng);
    }

    template<class T>
    void getRandVector(std::vector<T>& v) {
        if (!m_isInit) {
            m_isInit = true;
            m_distr = std::uniform_real_distribution<T>(T(0), T(1));
        }
        for (std::size_t i = 0; i < v.size(); i++) {
            v[i] = m_distr(m_eng);
        }

    }
    void getDistribVector(std::vector<T>& v,T min, T max) {
        bool bmin = true;
        for (std::size_t i = 0; i < v.size(); i++) {
            if(bmin) { 
                v[i] =min;
            }
            else {
                v[i] = max;
            }
            bmin = !bmin;
        }
    }
};

