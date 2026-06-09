#include "Correlator.h"

namespace hoomd {
Correlator::Correlator(unsigned int output_interval,
                       unsigned int max_lag)
    : m_output_interval(output_interval),
      m_max_lag(max_lag),
      m_nvalues(0),
      m_firstindex(0),
      m_lastindex(0),
      m_nsample(0)
{
    if (m_output_interval == 0)
        throw std::invalid_argument("Correlator: output_interval must be positive.");
    if (m_max_lag == 0)
        throw std::invalid_argument("Correlator: max_lag must be positive.");
}

py::array_t<double> Correlator::getCorrelation() const
{
    if (m_nvalues == 0)
        return py::array_t<double>();

    std::vector<double> result;
    normalize(result);

    py::array_t<double> array({m_max_lag, m_nvalues});

    auto buf = array.mutable_unchecked<2>();

    for (unsigned int lag = 0; lag < m_max_lag; ++lag)
    {
        for (unsigned int k = 0; k < m_nvalues; ++k)
        {
            buf(lag, k) = result[static_cast<size_t>(lag) * m_nvalues + k];
        }
    }

    return array;
}


void Correlator::initialize(unsigned int nvalues)
{
    m_nvalues = nvalues;

    m_buffer.resize(static_cast<size_t>(m_max_lag) * m_nvalues, 0.0);
    m_corr.resize(static_cast<size_t>(m_max_lag) * m_nvalues, 0.0);
    m_counts.resize(m_max_lag, 0);
    m_firstindex = 0;
    m_lastindex = 0;
    m_nsample = 0;
}

void Correlator::accumulate(const std::vector<double>& values, uint64_t timestep)
{
    if (values.empty())
    {
        throw std::runtime_error("Correlator: no logged values were provided.");
    }

    if (m_nvalues == 0)
    {
        if (values.size() > std::numeric_limits<unsigned int>::max())
            throw std::runtime_error("Correlator: too many logged values.");
        initialize(static_cast<unsigned int>(values.size()));
    }
    else if (values.size() != m_nvalues)
    {
        throw std::runtime_error("Correlator: logged value count changed.");
    }

    if (m_nsample == 0)
    {
        m_lastindex = 0;
    }
    else
    {
        m_lastindex++;
        if (m_lastindex == m_max_lag)
            m_lastindex = 0;
    }

    const size_t offset = static_cast<size_t>(m_lastindex) * m_nvalues;
    std::copy(values.begin(), values.end(), m_buffer.begin() + offset);

    if (m_nsample < m_max_lag)
    {
        m_nsample++;
    }
    else
    {
        m_firstindex++;
        if (m_firstindex == m_max_lag)
            m_firstindex = 0;
    }

    accumulateValues();

    if (timestep % m_output_interval == 0)
        write_output(timestep);
}

void Correlator::accumulateValues()
{

    // --- count更新 ---
    for (unsigned int k = 0; k < m_nsample; k++)
        m_counts[k]++;

    if (m_nsample > m_max_lag || m_lastindex >= m_max_lag)
        throw std::runtime_error("Correlator: invalid ring-buffer state.");

    unsigned int m = m_lastindex;
    const unsigned int n = m_lastindex;

    for (unsigned int lag = 0; lag < m_nsample; lag++)
    {
        const size_t corr_offset = static_cast<size_t>(lag) * m_nvalues;
        const size_t buffer_m_offset = static_cast<size_t>(m) * m_nvalues;
        const size_t buffer_n_offset = static_cast<size_t>(n) * m_nvalues;

        for (unsigned int i = 0; i < m_nvalues; i++)
        {
            m_corr[corr_offset + i] +=
                m_buffer[buffer_m_offset + i] * m_buffer[buffer_n_offset + i];
        }

        if (m == 0)
            m = m_max_lag - 1;
        else
            m--;
    }

}

void Correlator::normalize(std::vector<double>& result) const
{
    result = m_corr;

    for (unsigned int lag = 0; lag < m_max_lag; ++lag)
    {
        if (m_counts[lag] == 0)
            continue;

        for (unsigned int k = 0; k < m_nvalues; ++k)
        {
            result[static_cast<size_t>(lag) * m_nvalues + k] /= (double)m_counts[lag];
        }
    }
}

void Correlator::write_output(uint64_t timestep)
{
    std::vector<double> result;
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
                val = result[static_cast<size_t>(lag) * m_nvalues + k];

            file << " " << val;
        }

        file << "\n";
    }

    file.close();
}

namespace detail {
    void export_Correlator(py::module& m)
    {
        py::class_<Correlator, std::shared_ptr<Correlator>>(m, "Correlator")
            .def(py::init<unsigned int, unsigned int>())
            .def("accumulate",
                 &Correlator::accumulate)
            .def_property_readonly("correlation",
                               &Correlator::getCorrelation);
    }
}
} // namespace hoomd
