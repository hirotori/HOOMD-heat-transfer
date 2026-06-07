#pragma once

#include "hoomd/HOOMDMath.h"
#include "hoomd/ParticleGroup.h"
#include "hoomd/Updater.h"
#include "hoomd/Variant.h"
#include <pybind11/pybind11.h>

#include <memory>
#include <string>

namespace hoomd
    {

class PYBIND11_EXPORT MuellerPlatheHeatFlow : public Updater
    {
    public:
    enum Direction
        {
        X = 0,
        Y,
        Z
        };

    MuellerPlatheHeatFlow(std::shared_ptr<SystemDefinition> sysdef,
                          std::shared_ptr<Trigger> trigger,
                          std::shared_ptr<ParticleGroup> group,
                          std::shared_ptr<Variant> heat_flux_target,
                          std::string slab_direction_str,
                          unsigned int n_slabs,
                          unsigned int cold_slab,
                          unsigned int hot_slab,
                          Scalar heat_flux_epsilon);

    virtual ~MuellerPlatheHeatFlow();

    virtual void update(uint64_t timestep);

    Scalar getSummedExchangedEnergy() const
        {
        return m_exchanged_energy;
        }

    unsigned int getNSlabs() const
        {
        return m_n_slabs;
        }

    unsigned int getColdSlab() const
        {
        return m_cold_slab;
        }

    unsigned int getHotSlab() const
        {
        return m_hot_slab;
        }

    std::shared_ptr<Variant> getHeatFluxTarget() const
        {
        return m_heat_flux_target;
        }

    std::string getSlabDirectionPython() const
        {
        return getStringFromDirection(m_slab_direction);
        }

    Scalar getHeatFluxEpsilon() const
        {
        return m_heat_flux_epsilon;
        }

    static Direction getDirectionFromString(const std::string& direction_str);
    static std::string getStringFromDirection(Direction direction);

    void forceOrthorhombicBoxCheck()
        {
        m_needs_orthorhombic_check = true;
        }

    protected:
    struct HeatParticle
        {
        Scalar energy;
        Scalar mass;
        Scalar3 velocity;
        unsigned int tag;
        };

    void resetCandidates();
    void swapColdHotSlab();
    virtual void searchHotColdParticles();
    virtual bool exchangeVelocities(uint64_t timestep, Scalar& exchanged_energy);

    std::shared_ptr<ParticleGroup> m_group;
    HeatParticle m_hottest_cold_particle;
    HeatParticle m_coldest_hot_particle;
    Direction m_slab_direction;

    private:
    std::shared_ptr<Variant> m_heat_flux_target;
    Scalar m_heat_flux_epsilon;
    unsigned int m_n_slabs;
    unsigned int m_cold_slab;
    unsigned int m_hot_slab;
    Scalar m_exchanged_energy;
    bool m_needs_orthorhombic_check;

    void validateSlab(unsigned int slab_id, const std::string& name) const;
    void verifyOrthorhombicBox();
    Scalar getSlabArea() const;
    };

namespace detail
    {
    void export_MuellerPlatheHeatFlow(pybind11::module& m);
    }

    } // namespace hoomd
