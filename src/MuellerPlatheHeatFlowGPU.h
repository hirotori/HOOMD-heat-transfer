#pragma once

#include "MuellerPlatheHeatFlow.h"

#ifdef ENABLE_HIP

#include "hoomd/Autotuner.h"
#include <memory>

namespace hoomd
    {

class PYBIND11_EXPORT MuellerPlatheHeatFlowGPU : public MuellerPlatheHeatFlow
    {
    public:
    MuellerPlatheHeatFlowGPU(std::shared_ptr<SystemDefinition> sysdef,
                             std::shared_ptr<Trigger> trigger,
                             std::shared_ptr<ParticleGroup> group,
                             std::shared_ptr<Variant> heat_flux_target,
                             std::string slab_direction_str,
                             unsigned int n_slabs,
                             unsigned int cold_slab,
                             unsigned int hot_slab,
                             Scalar heat_flux_epsilon);

    virtual ~MuellerPlatheHeatFlowGPU();

    protected:
    std::shared_ptr<Autotuner<1>> m_tuner;

    virtual void searchHotColdParticles() override;
    virtual bool exchangeVelocities(uint64_t timestep, Scalar& exchanged_energy) override;
    };

namespace detail
    {
    void export_MuellerPlatheHeatFlowGPU(pybind11::module& m);
    }

    } // namespace hoomd

#endif // ENABLE_HIP
