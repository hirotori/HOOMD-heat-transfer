// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef __COSINEANGLEFORCECOMPUTE_H__
#define __COSINEANGLEFORCECOMPUTE_H__

#include "hoomd/BondedGroupData.h"
#include "hoomd/ForceCompute.h"

#include <memory>
#include <pybind11/pybind11.h>

namespace hoomd
    {
namespace md
    {
struct cosine_angle_params
    {
    Scalar k;

#ifndef __HIPCC__
    cosine_angle_params() : k(0) { }

    cosine_angle_params(pybind11::dict params) : k(params["k"].cast<Scalar>()) { }

    pybind11::dict asDict()
        {
        pybind11::dict v;
        v["k"] = k;
        return v;
        }
#endif
    }
#if HOOMD_LONGREAL_SIZE == 32
    __attribute__((aligned(4)));
#else
    __attribute__((aligned(8)));
#endif

class PYBIND11_EXPORT CosineAngleForceCompute : public ForceCompute
    {
    public:
    CosineAngleForceCompute(std::shared_ptr<SystemDefinition> sysdef);

    virtual ~CosineAngleForceCompute();

    virtual void setParams(unsigned int type, Scalar K);

    virtual void setParamsPython(std::string type, pybind11::dict params);

    virtual pybind11::dict getParams(std::string type);

#ifdef ENABLE_MPI
    virtual CommFlags getRequestedCommFlags(uint64_t timestep)
        {
        CommFlags flags = CommFlags(0);
        flags[comm_flag::tag] = 1;
        flags |= ForceCompute::getRequestedCommFlags(timestep);
        return flags;
        }
#endif

    protected:
    Scalar* m_K;

    std::shared_ptr<AngleData> m_angle_data;

    virtual void computeForces(uint64_t timestep);
    };

namespace detail
    {
void export_CosineAngleForceCompute(pybind11::module& m);
    } // namespace detail
    } // namespace md
    } // namespace hoomd

#endif // __COSINEANGLEFORCECOMPUTE_H__
