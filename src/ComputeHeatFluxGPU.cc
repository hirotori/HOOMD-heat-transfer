#include "ComputeHeatFluxGPU.h"

#ifdef ENABLE_HIP

#include "ComputeHeatFluxGPU.cuh"
#include "hoomd/GlobalArray.h"

#include <algorithm>
#include <stdexcept>

namespace hoomd
    {

ComputeHeatFluxGPU::ComputeHeatFluxGPU(std::shared_ptr<SystemDefinition> sysdef,
                                       std::shared_ptr<ParticleGroup> group,
                                       bool include_enthalpy)
    : ComputeHeatFlux(sysdef, group, include_enthalpy), m_partial_kin(1, m_exec_conf),
      m_partial_vir(1, m_exec_conf), m_partial_count(1, m_exec_conf),
      m_partial_mvsq(1, m_exec_conf), m_partial_uesum(1, m_exec_conf),
      m_partial_ptrace(1, m_exec_conf), m_partial_vsum(1, m_exec_conf)
    {
    if (!m_exec_conf->isCUDAEnabled())
        {
        m_exec_conf->msg->error()
            << "ComputeHeatFluxGPU cannot run without a GPU execution configuration" << std::endl;
        throw std::runtime_error("Error initializing ComputeHeatFluxGPU");
        }

    }

void ComputeHeatFluxGPU::compute(uint64_t timestep)
    {
    if (!shouldCompute(timestep))
        return;

    if (m_group->getNumMembersGlobal() == 0)
        return;

    const unsigned int group_size = m_group->getNumMembers();
    const GlobalArray<unsigned int>& group_members = m_group->getIndexArray();

    ArrayHandle<Scalar4> d_vel(m_pdata->getVelocities(), access_location::device, access_mode::read);
    ArrayHandle<Scalar4> d_net_force(m_pdata->getNetForce(),
                                     access_location::device,
                                     access_mode::read);

    const GlobalArray<Scalar>& net_virial = m_pdata->getNetVirial();
    ArrayHandle<Scalar> d_virial(net_virial, access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_group_members(group_members,
                                              access_location::device,
                                              access_mode::read);

    const unsigned int block_size = 256;
    const unsigned int num_blocks = std::max(1u, (group_size + block_size - 1) / block_size);

    if (m_partial_kin.getNumElements() < num_blocks)
        {
        m_partial_kin.resize(num_blocks);
        m_partial_vir.resize(num_blocks);
        }

    {
    ArrayHandle<Scalar3> d_partial_kin(m_partial_kin,
                                       access_location::device,
                                       access_mode::overwrite);
    ArrayHandle<Scalar3> d_partial_vir(m_partial_vir,
                                       access_location::device,
                                       access_mode::overwrite);

    hipError_t error = kernel::gpu_compute_heat_flux(d_partial_kin.data,
                                                     d_partial_vir.data,
                                                     d_vel.data,
                                                     d_net_force.data,
                                                     d_virial.data,
                                                     net_virial.getPitch(),
                                                     d_group_members.data,
                                                     group_size,
                                                     block_size,
                                                     num_blocks);
    if (error != hipSuccess)
        {
        m_exec_conf->msg->error() << "Error launching gpu_compute_heat_flux: "
                                  << hipGetErrorString(error) << std::endl;
        throw std::runtime_error("Error computing heat flux on GPU");
        }
    }

    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

    m_J_kin = make_scalar3(0, 0, 0);
    m_J_vir = make_scalar3(0, 0, 0);
    m_Jh = make_scalar3(0, 0, 0);

    ArrayHandle<Scalar3> h_partial_kin(m_partial_kin,
                                       access_location::host,
                                       access_mode::read);
    ArrayHandle<Scalar3> h_partial_vir(m_partial_vir,
                                       access_location::host,
                                       access_mode::read);

    for (unsigned int block = 0; block < num_blocks; ++block)
        {
        m_J_kin += h_partial_kin.data[block];
        m_J_vir += h_partial_vir.data[block];
        }

    if (m_include_enthalpy)
        {
        const unsigned int N = m_pdata->getN();
        const unsigned int enthalpy_block_size = 256;
        const unsigned int enthalpy_num_blocks
            = std::max(1u, (N + enthalpy_block_size - 1) / enthalpy_block_size);
        const unsigned int partial_size = enthalpy_num_blocks * m_ntypes;

        if (m_partial_count.getNumElements() < partial_size)
            {
            m_partial_count.resize(partial_size);
            m_partial_mvsq.resize(partial_size);
            m_partial_uesum.resize(partial_size);
            m_partial_ptrace.resize(partial_size);
            m_partial_vsum.resize(partial_size);
            }

        {
        ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(),
                                   access_location::device,
                                   access_mode::read);
        ArrayHandle<Scalar> d_partial_count(m_partial_count,
                                            access_location::device,
                                            access_mode::overwrite);
        ArrayHandle<Scalar> d_partial_mvsq(m_partial_mvsq,
                                           access_location::device,
                                           access_mode::overwrite);
        ArrayHandle<Scalar> d_partial_uesum(m_partial_uesum,
                                            access_location::device,
                                            access_mode::overwrite);
        ArrayHandle<Scalar> d_partial_ptrace(m_partial_ptrace,
                                             access_location::device,
                                             access_mode::overwrite);
        ArrayHandle<Scalar3> d_partial_vsum(m_partial_vsum,
                                            access_location::device,
                                            access_mode::overwrite);

        hipError_t error = kernel::gpu_compute_enthalpy_flux_partial(d_partial_count.data,
                                                                     d_partial_mvsq.data,
                                                                     d_partial_uesum.data,
                                                                     d_partial_ptrace.data,
                                                                     d_partial_vsum.data,
                                                                     d_pos.data,
                                                                     d_vel.data,
                                                                     d_net_force.data,
                                                                     d_virial.data,
                                                                     net_virial.getPitch(),
                                                                     N,
                                                                     m_ntypes,
                                                                     enthalpy_block_size,
                                                                     enthalpy_num_blocks);
        if (error != hipSuccess)
            {
            m_exec_conf->msg->error()
                << "Error launching gpu_compute_enthalpy_flux_partial: "
                << hipGetErrorString(error) << std::endl;
            throw std::runtime_error("Error computing enthalpy flux on GPU");
            }
        }

        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();

        std::fill(m_count.begin(), m_count.end(), Scalar(0));
        std::fill(m_mvsq.begin(), m_mvsq.end(), Scalar(0));
        std::fill(m_uesum.begin(), m_uesum.end(), Scalar(0));
        std::fill(m_ptrace.begin(), m_ptrace.end(), Scalar(0));
        std::fill(m_vsum.begin(), m_vsum.end(), make_scalar3(0, 0, 0));

        ArrayHandle<Scalar> h_partial_count(m_partial_count,
                                            access_location::host,
                                            access_mode::read);
        ArrayHandle<Scalar> h_partial_mvsq(m_partial_mvsq,
                                           access_location::host,
                                           access_mode::read);
        ArrayHandle<Scalar> h_partial_uesum(m_partial_uesum,
                                            access_location::host,
                                            access_mode::read);
        ArrayHandle<Scalar> h_partial_ptrace(m_partial_ptrace,
                                             access_location::host,
                                             access_mode::read);
        ArrayHandle<Scalar3> h_partial_vsum(m_partial_vsum,
                                            access_location::host,
                                            access_mode::read);

        for (int type = 0; type < m_ntypes; ++type)
            {
            for (unsigned int block = 0; block < enthalpy_num_blocks; ++block)
                {
                const unsigned int offset = type * enthalpy_num_blocks + block;
                m_count[type] += h_partial_count.data[offset];
                m_mvsq[type] += h_partial_mvsq.data[offset];
                m_uesum[type] += h_partial_uesum.data[offset];
                m_ptrace[type] += h_partial_ptrace.data[offset];
                m_vsum[type] += h_partial_vsum.data[offset];
                }

            if (m_count[type] > 0)
                {
                m_h[type] = (Scalar(5.0 / 6.0) * m_mvsq[type] + m_uesum[type]
                             + m_ptrace[type] / Scalar(3.0))
                            / m_count[type];
                }
            else
                {
                m_h[type] = 0;
                }

            m_Jh += m_h[type] * m_vsum[type];
            }
        }

    m_J = m_J_kin + m_J_vir - m_Jh;
    }

namespace detail
    {
void export_ComputeHeatFluxGPU(pybind11::module& m)
    {
    pybind11::class_<ComputeHeatFluxGPU, ComputeHeatFlux, std::shared_ptr<ComputeHeatFluxGPU>>(
        m,
        "ComputeHeatFluxGPU")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            std::shared_ptr<ParticleGroup>,
                            bool>());
    }
    } // end namespace detail

    } // end namespace hoomd

#endif // ENABLE_HIP
