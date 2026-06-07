#pragma once

#include "ComputeheatFlux.h"

#ifdef ENABLE_HIP

#include "hoomd/GPUArray.h"

namespace hoomd
    {

class PYBIND11_EXPORT ComputeHeatFluxGPU : public ComputeHeatFlux
    {
    public:
    ComputeHeatFluxGPU(std::shared_ptr<SystemDefinition> sysdef,
                       std::shared_ptr<ParticleGroup> group,
                       bool include_enthalpy);

    virtual ~ComputeHeatFluxGPU() { }

    protected:
    virtual void compute(uint64_t timestep) override;

    private:
    GPUArray<Scalar3> m_partial_kin;
    GPUArray<Scalar3> m_partial_vir;
    GPUArray<Scalar> m_partial_count;
    GPUArray<Scalar> m_partial_mvsq;
    GPUArray<Scalar> m_partial_uesum;
    GPUArray<Scalar> m_partial_ptrace;
    GPUArray<Scalar3> m_partial_vsum;
    };

namespace detail
    {
void export_ComputeHeatFluxGPU(pybind11::module& m);
    } // namespace detail
    } // namespace hoomd

#endif // ENABLE_HIP
