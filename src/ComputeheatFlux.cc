#include "ComputeheatFlux.h"
#include "hoomd/ParticleData.h"
#include "hoomd/BoxDim.h"
#include "hoomd/GlobalArray.h"
#include "hoomd/ParticleGroup.h"

#include <stdexcept>

namespace hoomd {
ComputeHeatFlux::ComputeHeatFlux(std::shared_ptr<SystemDefinition> sysdef, 
                                 std::shared_ptr<ParticleGroup> group,
                                 bool include_enthalpy)
    : Compute(sysdef), 
      m_J(make_scalar3(0,0,0)), 
      m_J_kin(make_scalar3(0,0,0)), 
      m_J_vir(make_scalar3(0,0,0)),
      m_group(group),
      m_include_enthalpy(include_enthalpy),
      m_Jh(make_scalar3(0,0,0))
{
    m_ntypes = m_pdata->getNTypes();

    if (m_include_enthalpy)
    {
        m_count.resize(m_ntypes);
        m_mvsq.resize(m_ntypes);
        m_uesum.resize(m_ntypes);
        m_ptrace.resize(m_ntypes);
        m_vsum.resize(m_ntypes);
        m_h.resize(m_ntypes);
    }

}

void ComputeHeatFlux::compute(uint64_t timestep)
{
    if (!shouldCompute(timestep))
        return;
    
    auto pdata = m_sysdef->getParticleData();
    
    // just drop out if the group is an empty group
    if (m_group->getNumMembersGlobal() == 0)
        return;

    unsigned int group_size = m_group->getNumMembers();

    ArrayHandle<Scalar4> h_vel(pdata->getVelocities(), access_location::host, access_mode::read);

    const GlobalArray<Scalar>& net_virial = pdata->getNetVirial();
    ArrayHandle<Scalar> h_virial(net_virial, access_location::host, access_mode::read);

    ArrayHandle<Scalar4> h_net_force(pdata->getNetForce(), access_location::host, access_mode::read);
    Scalar3 J_kin = make_scalar3(0,0,0);
    Scalar3 J_vir = make_scalar3(0,0,0);
    
    size_t virial_pitch = net_virial.getPitch();
    
    //for (unsigned int i = 0; i < N; ++i)
    for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
    {
        unsigned int i = m_group->getMemberIndex(group_idx);
        // velocity
        Scalar3 v = make_scalar3(h_vel.data[i].x,
                                 h_vel.data[i].y,
                                 h_vel.data[i].z);

        Scalar m = h_vel.data[i].w;

        // kinetic energy
        Scalar ke = Scalar(0.5) * m * (v.x*v.x + v.y*v.y + v.z*v.z);

        // total particle energy
        Scalar ei = ke + (double)h_net_force.data[i].w;

        // convective term
        J_kin += ei * v;

        // particle virial tensor (xx, yy, zz, xy, xz, yz)

        Scalar Wxx = h_virial.data[i + 0*virial_pitch];
        Scalar Wxy = h_virial.data[i + 1*virial_pitch];
        Scalar Wxz = h_virial.data[i + 2*virial_pitch];
        Scalar Wyy = h_virial.data[i + 3*virial_pitch];
        Scalar Wyz = h_virial.data[i + 4*virial_pitch];
        Scalar Wzz = h_virial.data[i + 5*virial_pitch];

        // virial term: W_i · v_i
        J_vir.x += Wxx * v.x + Wxy * v.y + Wxz * v.z;
        J_vir.y += Wxy * v.x + Wyy * v.y + Wyz * v.z;
        J_vir.z += Wxz * v.x + Wyz * v.y + Wzz * v.z;
    }

    m_J_kin = J_kin;
    m_J_vir = J_vir;

    // compute enthalpy flux in each atomic species
    m_Jh = make_scalar3(0, 0, 0);
    if (m_include_enthalpy)
    {
        const unsigned int current_ntypes = m_pdata->getNTypes();
        if (current_ntypes != m_ntypes)
        {
            m_ntypes = current_ntypes;
            m_count.resize(m_ntypes);
            m_mvsq.resize(m_ntypes);
            m_uesum.resize(m_ntypes);
            m_ptrace.resize(m_ntypes);
            m_vsum.resize(m_ntypes);
            m_h.resize(m_ntypes);
        }

        // zero clear
        for (unsigned int t = 0; t < m_ntypes; ++t)
        {
            m_count[t] = 0.0;
            m_mvsq[t]  = 0.0;
            m_uesum[t] = 0.0;
            m_ptrace[t]= 0.0;
            m_vsum[t]  = make_scalar3(0,0,0);
        }

        // compute partial enthalpy
        const unsigned int N = m_pdata->getN();
        ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);

        for (unsigned int i = 0; i < N; ++i)
        {
            const unsigned int t = __scalar_as_int(h_pos.data[i].w);
            if (t >= m_ntypes)
            {
                throw std::runtime_error("ComputeHeatFlux: particle type index out of range");
            }

            Scalar3 v = make_scalar3(h_vel.data[i].x,
                                    h_vel.data[i].y,
                                    h_vel.data[i].z);

            // reduce center-of-mass velocity to remove drift
            // v -= vcom;

            const Scalar m = h_vel.data[i].w;
            const Scalar pe = h_net_force.data[i].w;
            const Scalar vsq = dot(v, v);
            const Scalar ptrace = h_virial.data[i + 0*virial_pitch] + h_virial.data[i + 3*virial_pitch] + h_virial.data[i + 5*virial_pitch];

            m_count[t] += 1.0;
            m_mvsq[t]  += m * vsq;
            m_uesum[t] += pe;
            m_ptrace[t]+= ptrace;
            m_vsum[t]  = m_vsum[t] + v;
        }

        // --- h_α(t) ---
        for (unsigned int t = 0; t < m_ntypes; ++t)
        {
            if (m_count[t] > 0)
            {
                m_h[t] = ( (Scalar)(5.0/6.0) * m_mvsq[t]
                        + m_uesum[t]
                        + m_ptrace[t] / (Scalar)3.0 )
                        / m_count[t];
            }
            else
            {
                m_h[t] = 0.0;
            }
        }


        // compute enthalpy flux
        for (unsigned int t = 0; t < m_ntypes; ++t)
        {
            m_Jh = m_Jh + m_h[t] * m_vsum[t];
        }

    }
    m_J = J_kin + J_vir - m_Jh;
}

namespace detail {
    void export_ComputeHeatFlux(pybind11::module& m)
    {
        pybind11::class_<ComputeHeatFlux, Compute, std::shared_ptr<ComputeHeatFlux>>(m, "ComputeHeatFlux")
            .def(pybind11::init<std::shared_ptr<SystemDefinition>, std::shared_ptr<ParticleGroup>, bool>())
            .def_property_readonly("heatflux", &ComputeHeatFlux::getHeatFluxPython)
            .def_property_readonly("kinetic_heatflux", &ComputeHeatFlux::getKineticHeatFluxPython)
            .def_property_readonly("virial_heatflux", &ComputeHeatFlux::getVirialHeatFluxPython)
            .def_property_readonly("enthalpy_flux", &ComputeHeatFlux::getEnthalpyFluxPython);
    }
} // end namespace detail
} // namespace hoomd
