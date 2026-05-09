#include "Correlator.h"

namespace hoomd {
Correlator::Correlator(std::shared_ptr<SystemDefinition> sysdef,
                       std::shared_ptr<Trigger> trigger,
                       unsigned int output_interval,
                       unsigned int max_lag)
    : Analyzer(sysdef, trigger),
      m_output_interval(output_interval),
      m_max_lag(max_lag),
      m_nvalues(0),
      m_firstindex(0),
      m_lastindex(-1),
      m_nsample(0)
{
      m_log_writer = py::none();
}

void Correlator::setLogWriter(py::object log_writer)
{
    m_log_writer = log_writer;
}

py::object Correlator::getLogWriter() const
{
    return m_log_writer;
}

py::array_t<double> Correlator::getCorrelation() const
{
    if (m_nvalues == 0)
        return py::array_t<double>();

    std::vector<std::vector<double>> result;
    normalize(result);

    py::array_t<double> array({m_max_lag, m_nvalues});

    auto buf = array.mutable_unchecked<2>();

    for (unsigned int lag = 0; lag < m_max_lag; ++lag)
    {
        for (unsigned int k = 0; k < m_nvalues; ++k)
        {
            buf(lag, k) = result[lag][k];
        }
    }

    return array;
}


void Correlator::initialize(unsigned int nvalues)
{
    m_nvalues = nvalues;

    m_buffer.resize(m_max_lag, std::vector<double>(m_nvalues, 0.0));
    m_corr.resize(m_max_lag, std::vector<double>(m_nvalues, 0.0));
    m_counts.resize(m_max_lag, 0);
    m_firstindex = 0;
    m_lastindex = -1;
    m_nsample = 0;
}

void Correlator::analyze(uint64_t timestep)
{
    // --- Python側 log() 呼び出し ---
    py::object result = m_log_writer.attr("log")();

    // ここでは dict を想定（flatten済み）
    py::dict data = result.cast<py::dict>();

    std::vector<double> values;
    values.reserve(data.size());

    for (auto item : data)
    {
        values.push_back(py::cast<double>(item.second));
    }

    if (m_nvalues == 0)
        initialize(values.size());

    // save to ring-buffer
    m_lastindex++;
    if (m_lastindex == (int)m_max_lag) 
        m_lastindex = 0;
    
    m_buffer[m_lastindex] = values;
    
    if (m_nsample < m_max_lag)
    {
        m_nsample++;
    }
    else
    {
        m_firstindex++;
        if (m_firstindex == (int)m_max_lag)
            m_firstindex = 0;
    }

    accumulate(values);

    if (timestep % m_output_interval == 0)
        write_output(timestep);
}

void Correlator::accumulate(const std::vector<double>& values)
{

    // --- count更新 ---
    for (unsigned int k = 0; k < m_nsample; k++)
        m_counts[k]++;

    int m = m_lastindex;
    int n = m_lastindex;

    for (unsigned int lag = 0; lag < m_nsample; lag++)
    {
        for (unsigned int i = 0; i < m_nvalues; i++)
        {
            m_corr[lag][i] +=
                m_buffer[m][i] * m_buffer[n][i];
        }

        m--;
        if (m < 0) m = m_max_lag - 1;
    }

}

void Correlator::normalize(std::vector<std::vector<double>>& result) const
{
    result = m_corr;

    for (unsigned int lag = 0; lag < m_max_lag; ++lag)
    {
        if (m_counts[lag] == 0)
            continue;

        for (unsigned int k = 0; k < m_nvalues; ++k)
        {
            result[lag][k] /= (double)m_counts[lag];
        }
    }
}

void Correlator::write_output(uint64_t timestep)
{
    std::vector<std::vector<double>> result;
    normalize(result);

    std::ofstream file("correlation_" + std::to_string(timestep) + ".dat");

    // --- ヘッダ ---
    file << "# Time-correlated data\n";
    file << "# Timestep " << timestep << "\n";
    file << "# Index TimeLag Count";

    for (unsigned int k = 0; k < m_nvalues; k++)
        file << " C" << k;

    file << "\n";

    // --- データ ---
    for (unsigned int lag = 0; lag < m_max_lag; lag++)
    {
        unsigned int count = m_counts[lag];

        file << lag << " "           // Index
             << lag << " "           // TimeLag（※後でnevery対応可）
             << count;              // サンプル数

        for (unsigned int k = 0; k < m_nvalues; k++)
        {
            double val = 0.0;
            if (count > 0)
                val = result[lag][k];

            file << " " << val;
        }

        file << "\n";
    }

    file.close();
}

namespace detail {
    void export_Correlator(py::module& m)
    {
        py::class_<Correlator, Analyzer, std::shared_ptr<Correlator>>(m, "Correlator")
            .def(py::init<std::shared_ptr<SystemDefinition>,
                          std::shared_ptr<Trigger>,
                          unsigned int, 
                          unsigned int>())
            .def_property("log_writer",
                          &Correlator::getLogWriter,
                          &Correlator::setLogWriter)
            .def_property_readonly("correlation",
                               &Correlator::getCorrelation);
    }
}
} // namespace hoomd

