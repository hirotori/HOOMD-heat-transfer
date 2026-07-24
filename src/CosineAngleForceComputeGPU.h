// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef __COSINEANGLEFORCECOMPUTEGPU_H__
#define __COSINEANGLEFORCECOMPUTEGPU_H__

#ifdef ENABLE_HIP

#include "CosineAngleForceCompute.h"
#include "CosineAngleForceGPU.cuh"
#include "hoomd/Autotuner.h"
#include "hoomd/GPUArray.h"

#include <memory>

namespace hoomd
    {
namespace md
    {
class PYBIND11_EXPORT CosineAngleForceComputeGPU : public CosineAngleForceCompute
    {
    public:
    CosineAngleForceComputeGPU(std::shared_ptr<SystemDefinition> sysdef);

    virtual ~CosineAngleForceComputeGPU();

    virtual void setParams(unsigned int type, Scalar K);

    protected:
    std::shared_ptr<Autotuner<1>> m_tuner;
    GPUArray<Scalar> m_params;

    virtual void computeForces(uint64_t timestep);
    };

namespace detail
    {
void export_CosineAngleForceComputeGPU(pybind11::module& m);
    } // namespace detail
    } // namespace md
    } // namespace hoomd

#endif // ENABLE_HIP

#endif // __COSINEANGLEFORCECOMPUTEGPU_H__
