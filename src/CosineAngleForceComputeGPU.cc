// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "CosineAngleForceComputeGPU.h"

#ifdef ENABLE_HIP

using namespace std;

namespace hoomd
    {
namespace md
    {
CosineAngleForceComputeGPU::CosineAngleForceComputeGPU(std::shared_ptr<SystemDefinition> sysdef)
    : CosineAngleForceCompute(sysdef)
    {
    if (!m_exec_conf->isCUDAEnabled())
        {
        m_exec_conf->msg->error()
            << "Creating a CosineAngleForceComputeGPU with no GPU in the execution configuration"
            << endl;
        throw std::runtime_error("Error initializing CosineAngleForceComputeGPU");
        }

    GPUArray<Scalar> params(m_angle_data->getNTypes(), m_exec_conf);
    m_params.swap(params);

    m_tuner.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(m_exec_conf)},
                                   m_exec_conf,
                                   "cosine_angle"));
    m_autotuners.push_back(m_tuner);
    }

CosineAngleForceComputeGPU::~CosineAngleForceComputeGPU() { }

void CosineAngleForceComputeGPU::setParams(unsigned int type, Scalar K)
    {
    CosineAngleForceCompute::setParams(type, K);

    ArrayHandle<Scalar> h_params(m_params, access_location::host, access_mode::readwrite);
    h_params.data[type] = K;
    }

void CosineAngleForceComputeGPU::computeForces(uint64_t timestep)
    {
    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);

    BoxDim box = m_pdata->getGlobalBox();

    ArrayHandle<Scalar4> d_force(m_force, access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar> d_virial(m_virial, access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar> d_params(m_params, access_location::device, access_mode::read);

    ArrayHandle<AngleData::members_t> d_gpu_anglelist(m_angle_data->getGPUTable(),
                                                      access_location::device,
                                                      access_mode::read);
    ArrayHandle<unsigned int> d_gpu_angle_pos_list(m_angle_data->getGPUPosTable(),
                                                   access_location::device,
                                                   access_mode::read);
    ArrayHandle<unsigned int> d_gpu_n_angles(m_angle_data->getNGroupsArray(),
                                             access_location::device,
                                             access_mode::read);

    m_tuner->begin();
    kernel::gpu_compute_cosine_angle_forces(d_force.data,
                                            d_virial.data,
                                            m_virial.getPitch(),
                                            m_pdata->getN(),
                                            d_pos.data,
                                            box,
                                            d_gpu_anglelist.data,
                                            d_gpu_angle_pos_list.data,
                                            m_angle_data->getGPUTableIndexer().getW(),
                                            d_gpu_n_angles.data,
                                            d_params.data,
                                            m_angle_data->getNTypes(),
                                            m_tuner->getParam()[0]);

    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    m_tuner->end();
    }

namespace detail
    {
void export_CosineAngleForceComputeGPU(pybind11::module& m)
    {
    pybind11::class_<CosineAngleForceComputeGPU,
                     CosineAngleForceCompute,
                     std::shared_ptr<CosineAngleForceComputeGPU>>(m, "CosineAngleForceComputeGPU")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>>());
    }

    } // namespace detail
    } // namespace md
    } // namespace hoomd

#endif // ENABLE_HIP
