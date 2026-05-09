#pragma once

#include <memory>
#include <vector>
#include <string>
#include <fstream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "hoomd/Trigger.h"
#include "hoomd/Analyzer.h"

#ifndef __CORRELATOR_H__
#define __CORRELATOR_H__

namespace py = pybind11;

namespace hoomd
{

class Correlator : public Analyzer
{
    public:
    // log_writer: Python 側の _CorrelatorLogWrapper
    Correlator(std::shared_ptr<SystemDefinition> sysdef,
               std::shared_ptr<Trigger> trigger, //use as sample_interval
               unsigned int output_interval,
               unsigned int max_lag);

    virtual ~Correlator() {}
    
    void setLogWriter(py::object log_writer);
    
    py::object getLogWriter() const;
    
    py::array_t<double> getCorrelation() const;

    virtual void analyze(uint64_t timestep) override;

private:
    // --- 設定 ---
    unsigned int m_output_interval;
    unsigned int m_max_lag;

    // --- Python 側ラッパ ---
    py::object m_log_writer;  // log() を持つ

    // --- データ構造 ---
    unsigned int m_nvalues;      // 物理量の数
    int m_firstindex;
    int m_lastindex;
    int m_nsample;

    std::vector<std::vector<double>> m_buffer; // [lag][value]
    std::vector<std::vector<double>> m_corr;   // 累積相関
    std::vector<double> m_counts;   // 各相関のサンプル個数

    // --- 内部処理 ---
    void initialize(unsigned int nvalues);
    void accumulate(const std::vector<double>& values);
    void normalize(std::vector<std::vector<double>>& result) const;
    void write_output(uint64_t timestep);
};
namespace detail {
   void export_Correlator(py::module &m); 
} // namespace detail
} // namespace hoomd

#endif // __CORRELATOR_H_
