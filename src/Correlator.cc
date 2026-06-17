#include "Correlator.h"

namespace hoomd {
Correlator::Correlator(unsigned int output_interval,
                       unsigned int max_lag,
                       unsigned int sample_interval)
    : m_output_interval(output_interval),
      m_max_lag(max_lag),
      m_sample_interval(sample_interval),
      m_nvalues(0),
      m_firstindex(0),
      m_lastindex(0),
      m_nsample(0)
{
    if (m_output_interval == 0)
        throw std::invalid_argument("Correlator: output_interval must be positive.");
    if (m_max_lag == 0)
        throw std::invalid_argument("Correlator: max_lag must be positive.");
    if (m_sample_interval == 0)
        throw std::invalid_argument("Correlator: sample_interval must be positive.");
    if (m_output_interval % m_sample_interval != 0)
        throw std::invalid_argument(
            "Correlator: output_interval must be a multiple of sample_interval.");
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

    if (!m_column_names.empty() && m_column_names.size() != m_nvalues)
    {
        throw std::runtime_error("Correlator: column name count does not match values.");
    }

    m_buffer.resize(static_cast<size_t>(m_max_lag) * m_nvalues, 0.0);
    m_corr.resize(static_cast<size_t>(m_max_lag) * m_nvalues, 0.0);
    m_counts.resize(m_max_lag, 0);
    m_firstindex = 0;
    m_lastindex = 0;
    m_nsample = 0;
}

void Correlator::setColumnNames(const std::vector<std::string>& column_names)
{
    if (m_nvalues != 0 && column_names.size() != m_nvalues)
    {
        throw std::runtime_error("Correlator: column name count does not match values.");
    }

    m_column_names = column_names;
}

void Correlator::accumulate(py::array_t<double, py::array::c_style | py::array::forcecast> values,
                            uint64_t timestep)
{
    py::buffer_info values_info = values.request();

    if (values_info.ndim != 1)
    {
        throw std::runtime_error("Correlator: logged values must be a 1D array.");
    }

    if (values_info.shape[0] <= 0)
    {
        throw std::runtime_error("Correlator: no logged values were provided.");
    }

    const size_t nvalues = static_cast<size_t>(values_info.shape[0]);
    const double* values_data = static_cast<const double*>(values_info.ptr);

    if (m_nvalues == 0)
    {
        if (nvalues > std::numeric_limits<unsigned int>::max())
            throw std::runtime_error("Correlator: too many logged values.");
        initialize(static_cast<unsigned int>(nvalues));
    }
    else if (nvalues != m_nvalues)
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
    if (offset + m_nvalues > m_buffer.size())
    {
        throw std::runtime_error("Correlator: ring-buffer write would exceed storage.");
    }
    std::copy(values_data, values_data + m_nvalues, m_buffer.begin() + offset);

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

    if (m_nsample > m_max_lag || m_lastindex >= m_max_lag
        || m_counts.size() != m_max_lag
        || m_buffer.size() != static_cast<size_t>(m_max_lag) * m_nvalues
        || m_corr.size() != static_cast<size_t>(m_max_lag) * m_nvalues)
        {
        throw std::runtime_error("Correlator: invalid ring-buffer state.");
        }

    unsigned int m = m_lastindex;
    const unsigned int n = m_lastindex;

    for (unsigned int lag = 0; lag < m_nsample; lag++)
    {
        const size_t corr_offset = static_cast<size_t>(lag) * m_nvalues;
        const size_t buffer_m_offset = static_cast<size_t>(m) * m_nvalues;
        const size_t buffer_n_offset = static_cast<size_t>(n) * m_nvalues;

        if (corr_offset + m_nvalues > m_corr.size()
            || buffer_m_offset + m_nvalues > m_buffer.size()
            || buffer_n_offset + m_nvalues > m_buffer.size())
            {
            throw std::runtime_error("Correlator: correlation access would exceed storage.");
            }

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
    {
        if (k < m_column_names.size())
            file << " " << m_column_names[k] << "*" << m_column_names[k];
        else
            file << " C" << k << "*C" << k;
    }

    file << "\n";

    // --- データ ---
    for (unsigned int lag = 0; lag < m_max_lag; lag++)
    {
        unsigned int count = m_counts[lag];

        file << lag + 1 << " "       // Index
             << lag * m_sample_interval << " "
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
            .def(py::init<unsigned int, unsigned int, unsigned int>())
            .def("accumulate",
                 &Correlator::accumulate)
            .def("set_column_names",
                 &Correlator::setColumnNames)
            .def_property_readonly("correlation",
                               &Correlator::getCorrelation);
    }
}
} // namespace hoomd
