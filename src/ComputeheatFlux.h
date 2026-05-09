#include "hoomd/Compute.h"
#include "hoomd/HOOMDMath.h"
#include <pybind11/pybind11.h>
#include "hoomd/ParticleGroup.h"
/*! \file ComputeheatFlux.h
    \brief Declares a class for computing heat flux
*/

#ifndef __COMPUTE_HEAT_FLUX_H__
#define __COMPUTE_HEAT_FLUX_H__


namespace hoomd {

class PYBIND11_EXPORT ComputeHeatFlux : public Compute
{
    public:
    ComputeHeatFlux(std::shared_ptr<SystemDefinition> sysdef, 
                    std::shared_ptr<ParticleGroup> group,
                    bool include_enthalpy);

    virtual ~ComputeHeatFlux() {}

    Scalar3 getHeatFlux() const { return m_J; }
    Scalar3 getKineticHeatFlux() const { return m_J_kin; }
    Scalar3 getVirialHeatFlux() const { return m_J_vir; }
    Scalar3 getEnthalpyFlux() const {return m_Jh; }
    pybind11::list getHeatFluxPython()
    {
      pybind11::list toReturn;
      Scalar3 jj   = getHeatFlux();
      toReturn.append(jj.x);
      toReturn.append(jj.y);
      toReturn.append(jj.z);
      return toReturn;
    }
    pybind11::list getKineticHeatFluxPython()
    {
      pybind11::list toReturn;
      Scalar3 jj_k = getKineticHeatFlux();
      toReturn.append(jj_k.x);
      toReturn.append(jj_k.y);
      toReturn.append(jj_k.z);
      return toReturn;
    }
    pybind11::list getVirialHeatFluxPython()
    {
      pybind11::list toReturn;
      Scalar3 jj_v = getVirialHeatFlux();
      toReturn.append(jj_v.x);
      toReturn.append(jj_v.y);
      toReturn.append(jj_v.z);
      return toReturn;
    }
    pybind11::list getEnthalpyFluxPython()
    {
      pybind11::list toReturn;
      Scalar3 jj_h = getEnthalpyFlux();
      toReturn.append(jj_h.x);
      toReturn.append(jj_h.y);
      toReturn.append(jj_h.z);
      return toReturn;
    }

    protected:
    virtual void compute(uint64_t timestep);

    
    protected:
    Scalar3 m_J;
    Scalar3 m_J_kin;
    Scalar3 m_J_vir;
    std::shared_ptr<ParticleGroup> m_group;

    // enthalpy flux
    bool m_include_enthalpy;
    int  m_ntypes;
    Scalar3 m_Jh;
    // for computing partial enthalpy
    std::vector<Scalar> m_count;
    std::vector<Scalar> m_mvsq;
    std::vector<Scalar> m_uesum;
    std::vector<Scalar> m_ptrace;
    std::vector<Scalar> m_h;
    std::vector<Scalar3> m_vsum;
};

namespace detail {
    void export_ComputeHeatFlux(pybind11::module &m);
} // namespace detail
} // namespace hoomd
#endif // __COMPUTE_HEAT_FLUX_H__
