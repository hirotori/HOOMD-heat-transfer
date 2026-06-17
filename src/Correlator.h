#pragma once

#include <memory>
#include <vector>
#include <string>
#include <fstream>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#ifndef __CORRELATOR_H__
#define __CORRELATOR_H__

namespace py = pybind11;

namespace hoomd
{

class Correlator
{
    public:
    Correlator(unsigned int output_interval,
               unsigned int max_lag,
               unsigned int sample_interval);

    virtual ~Correlator() {}

    py::array_t<double> getCorrelation() const;

    void accumulate(py::array_t<double, py::array::c_style | py::array::forcecast> values,
                    uint64_t timestep);

    void setColumnNames(const std::vector<std::string>& column_names);

private:
    // --- 設定 ---
    unsigned int m_output_interval;
    unsigned int m_max_lag;
    unsigned int m_sample_interval;
    std::vector<std::string> m_column_names;

    // --- データ構造 ---
    unsigned int m_nvalues;      // 物理量の数
    unsigned int m_firstindex;
    unsigned int m_lastindex;
    unsigned int m_nsample;

    std::vector<double> m_buffer; // [lag * m_nvalues + value]
    std::vector<double> m_corr;   // 累積相関
    std::vector<unsigned int> m_counts;   // 各相関のサンプル個数

    // --- 内部処理 ---
    void initialize(unsigned int nvalues);
    void accumulateValues();
    void normalize(std::vector<double>& result) const;
    void write_output(uint64_t timestep);
};
namespace detail {
   void export_Correlator(py::module &m); 
} // namespace detail
} // namespace hoomd

#endif // __CORRELATOR_H_
