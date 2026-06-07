#include "MuellerPlatheHeatFlow.h"
#include "hoomd/BoxDim.h"
#include "hoomd/HOOMDMath.h"
#include "hoomd/ParticleData.h"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace hoomd
    {
namespace
    {
    constexpr unsigned int invalid_tag = std::numeric_limits<unsigned int>::max();

    Scalar kineticEnergy(const Scalar4& vel)
        {
        return Scalar(0.5) * vel.w * (vel.x * vel.x + vel.y * vel.y + vel.z * vel.z);
        }

    Scalar3 velocityVector(const Scalar4& vel)
        {
        return make_scalar3(vel.x, vel.y, vel.z);
        }
    } // namespace

MuellerPlatheHeatFlow::MuellerPlatheHeatFlow(std::shared_ptr<SystemDefinition> sysdef,
                                             std::shared_ptr<Trigger> trigger,
                                             std::shared_ptr<ParticleGroup> group,
                                             std::shared_ptr<Variant> heat_flux_target,
                                             std::string slab_direction_str,
                                             unsigned int n_slabs,
                                             unsigned int cold_slab,
                                             unsigned int hot_slab,
                                             Scalar heat_flux_epsilon)
    : Updater(sysdef, trigger), m_group(group), m_slab_direction(getDirectionFromString(slab_direction_str)),
      m_heat_flux_target(heat_flux_target), m_heat_flux_epsilon(heat_flux_epsilon),
      m_n_slabs(n_slabs), m_cold_slab(cold_slab), m_hot_slab(hot_slab),
      m_exchanged_energy(0), m_needs_orthorhombic_check(true)
    {
    if (!m_heat_flux_target)
        {
        throw std::runtime_error("MuellerPlatheHeatFlow requires a heat_flux_target.");
        }
    if (m_n_slabs == 0)
        {
        throw std::runtime_error("MuellerPlatheHeatFlow requires n_slabs > 0.");
        }

    validateSlab(m_cold_slab, "cold_slab");
    validateSlab(m_hot_slab, "hot_slab");
    if (m_cold_slab == m_hot_slab)
        {
        throw std::runtime_error("MuellerPlatheHeatFlow requires distinct cold_slab and hot_slab.");
        }

    m_pdata->getBoxChangeSignal()
        .connect<MuellerPlatheHeatFlow, &MuellerPlatheHeatFlow::forceOrthorhombicBoxCheck>(this);

    resetCandidates();
    m_exec_conf->msg->notice(5) << "Constructing MuellerPlatheHeatFlow" << std::endl;
    }

MuellerPlatheHeatFlow::~MuellerPlatheHeatFlow()
    {
    m_exec_conf->msg->notice(5) << "Destroying MuellerPlatheHeatFlow" << std::endl;
    m_pdata->getBoxChangeSignal()
        .disconnect<MuellerPlatheHeatFlow, &MuellerPlatheHeatFlow::forceOrthorhombicBoxCheck>(this);
    }

void MuellerPlatheHeatFlow::update(uint64_t timestep)
    {
    Updater::update(timestep);
    if (m_needs_orthorhombic_check)
        verifyOrthorhombicBox();

    const Scalar area = getSlabArea();
    const Scalar target = (*m_heat_flux_target)(timestep);

    bool positive_swap_needed = target > m_exchanged_energy / area;
    positive_swap_needed &= m_cold_slab > m_hot_slab;
    bool negative_swap_needed = target < m_exchanged_energy / area;
    negative_swap_needed &= m_cold_slab < m_hot_slab;

    if ((positive_swap_needed || negative_swap_needed)
        && std::fabs(target - m_exchanged_energy / area) > m_heat_flux_epsilon)
        {
        swapColdHotSlab();
        }

    const int sign = m_hot_slab > m_cold_slab ? 1 : -1;
    unsigned int counter = 0;
    const unsigned int max_iteration = 100;
    bool stopped_at_discrete_exchange_limit = false;

    while (std::fabs(target - m_exchanged_energy / area) > m_heat_flux_epsilon
           && counter < max_iteration)
        {
        counter++;
        resetCandidates();
        searchHotColdParticles();

        if (m_hottest_cold_particle.tag == invalid_tag || m_coldest_hot_particle.tag == invalid_tag)
            {
            m_exec_conf->msg->warning() << "WARNING: at time " << timestep
                                        << " MuellerPlatheHeatFlow could not find a hot/cold pair."
                                        << std::endl;
            break;
            }

        const Scalar candidate_exchange_energy = m_hottest_cold_particle.energy
                                                 - m_coldest_hot_particle.energy;
        if (candidate_exchange_energy <= Scalar(0))
            {
            stopped_at_discrete_exchange_limit = true;
            break;
            }

        const Scalar remaining_energy_per_area = std::fabs(target - m_exchanged_energy / area);
        const Scalar candidate_energy_per_area = candidate_exchange_energy / area;
        // Heat exchange is quantized by particle pairs; stop when the next pair would overshoot.
        if (candidate_energy_per_area > remaining_energy_per_area + m_heat_flux_epsilon)
            {
            stopped_at_discrete_exchange_limit = true;
            break;
            }

        Scalar exchanged_energy = 0;
        if (exchangeVelocities(timestep, exchanged_energy))
            {
            m_exchanged_energy += sign * exchanged_energy;
            }
        else
            {
            break;
            }
        }

    if (counter >= max_iteration
        && !stopped_at_discrete_exchange_limit
        && std::fabs(target - m_exchanged_energy / area) > m_heat_flux_epsilon)
        {
        m_exec_conf->msg->warning()
            << "After " << counter
            << " MuellerPlatheHeatFlow exchanges, target " << target
            << " was not achieved. Current exchanged energy per area is "
            << m_exchanged_energy / area << "." << std::endl;
        }
    }

MuellerPlatheHeatFlow::Direction MuellerPlatheHeatFlow::getDirectionFromString(
    const std::string& direction_str)
    {
    if (direction_str == "x")
        return X;
    if (direction_str == "y")
        return Y;
    if (direction_str == "z")
        return Z;
    throw std::runtime_error("Direction must be x, y, or z.");
    }

std::string MuellerPlatheHeatFlow::getStringFromDirection(Direction direction)
    {
    if (direction == X)
        return "x";
    if (direction == Y)
        return "y";
    if (direction == Z)
        return "z";
    throw std::runtime_error("Direction must be x, y, or z.");
    }

void MuellerPlatheHeatFlow::validateSlab(unsigned int slab_id, const std::string& name) const
    {
    if (slab_id >= m_n_slabs)
        {
        throw std::runtime_error("MuellerPlatheHeatFlow initialized with invalid " + name + ".");
        }
    }

void MuellerPlatheHeatFlow::resetCandidates()
    {
    m_hottest_cold_particle.energy = -std::numeric_limits<Scalar>::max();
    m_hottest_cold_particle.mass = Scalar(0);
    m_hottest_cold_particle.velocity = make_scalar3(0, 0, 0);
    m_hottest_cold_particle.tag = invalid_tag;

    m_coldest_hot_particle.energy = std::numeric_limits<Scalar>::max();
    m_coldest_hot_particle.mass = Scalar(0);
    m_coldest_hot_particle.velocity = make_scalar3(0, 0, 0);
    m_coldest_hot_particle.tag = invalid_tag;
    }

void MuellerPlatheHeatFlow::swapColdHotSlab()
    {
    std::swap(m_cold_slab, m_hot_slab);
    std::swap(m_hottest_cold_particle, m_coldest_hot_particle);
    m_exec_conf->msg->notice(4) << "MuellerPlatheHeatFlow swapped cold/hot slabs: "
                                << m_cold_slab << " " << m_hot_slab << std::endl;
    }

void MuellerPlatheHeatFlow::searchHotColdParticles()
    {
    const unsigned int group_size = m_group->getNumMembers();
    if (group_size == 0)
        return;

    ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);

    const BoxDim& box = m_pdata->getGlobalBox();
    const Scalar3 L = box.getL();

    for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
        {
        const unsigned int j = m_group->getMemberIndex(group_idx);
        if (j >= m_pdata->getN())
            continue;

        unsigned int slab_index = 0;
        switch (m_slab_direction)
            {
        case X:
            slab_index = static_cast<unsigned int>((h_pos.data[j].x / L.x + Scalar(0.5)) * m_n_slabs);
            break;
        case Y:
            slab_index = static_cast<unsigned int>((h_pos.data[j].y / L.y + Scalar(0.5)) * m_n_slabs);
            break;
        case Z:
            slab_index = static_cast<unsigned int>((h_pos.data[j].z / L.z + Scalar(0.5)) * m_n_slabs);
            break;
            }
        slab_index %= m_n_slabs;

        if (slab_index != m_cold_slab && slab_index != m_hot_slab)
            continue;

        const Scalar energy = kineticEnergy(h_vel.data[j]);
        if (slab_index == m_cold_slab && energy > m_hottest_cold_particle.energy)
            {
            m_hottest_cold_particle.energy = energy;
            m_hottest_cold_particle.mass = h_vel.data[j].w;
            m_hottest_cold_particle.velocity = velocityVector(h_vel.data[j]);
            m_hottest_cold_particle.tag = h_tag.data[j];
            }
        else if (slab_index == m_hot_slab && energy < m_coldest_hot_particle.energy)
            {
            m_coldest_hot_particle.energy = energy;
            m_coldest_hot_particle.mass = h_vel.data[j].w;
            m_coldest_hot_particle.velocity = velocityVector(h_vel.data[j]);
            m_coldest_hot_particle.tag = h_tag.data[j];
            }
        }
    }

bool MuellerPlatheHeatFlow::exchangeVelocities(uint64_t timestep, Scalar& exchanged_energy)
    {
    if (m_hottest_cold_particle.tag == invalid_tag || m_coldest_hot_particle.tag == invalid_tag)
        {
        m_exec_conf->msg->warning() << "WARNING: at time " << timestep
                                    << " MuellerPlatheHeatFlow could not find a hot/cold pair."
                                    << std::endl;
        return false;
        }

    const Scalar mass_scale = std::max(std::fabs(m_hottest_cold_particle.mass),
                                       std::fabs(m_coldest_hot_particle.mass));
    const Scalar mass_tolerance = Scalar(100) * std::numeric_limits<Scalar>::epsilon()
                                  * std::max(Scalar(1), mass_scale);
    if (std::fabs(m_hottest_cold_particle.mass - m_coldest_hot_particle.mass) > mass_tolerance)
        {
        m_exec_conf->msg->warning()
            << "WARNING: at time " << timestep
            << " MuellerPlatheHeatFlow selected particles with different masses. "
            << "Use a filter containing equal-mass particles to conserve energy and momentum."
            << std::endl;
        return false;
        }

    exchanged_energy = m_hottest_cold_particle.energy - m_coldest_hot_particle.energy;
    if (exchanged_energy <= Scalar(0))
        {
        m_exec_conf->msg->notice(6)
            << "At time " << timestep
            << " MuellerPlatheHeatFlow found no positive kinetic-energy transfer."
            << std::endl;
        return false;
        }

    ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);
    const unsigned int cold_idx = h_rtag.data[m_hottest_cold_particle.tag];
    const unsigned int hot_idx = h_rtag.data[m_coldest_hot_particle.tag];
    const unsigned int n_total = m_pdata->getN() + m_pdata->getNGhosts();

    if (cold_idx >= n_total || hot_idx >= n_total)
        return false;

    ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(), access_location::host, access_mode::readwrite);
    h_vel.data[cold_idx].x = m_coldest_hot_particle.velocity.x;
    h_vel.data[cold_idx].y = m_coldest_hot_particle.velocity.y;
    h_vel.data[cold_idx].z = m_coldest_hot_particle.velocity.z;

    h_vel.data[hot_idx].x = m_hottest_cold_particle.velocity.x;
    h_vel.data[hot_idx].y = m_hottest_cold_particle.velocity.y;
    h_vel.data[hot_idx].z = m_hottest_cold_particle.velocity.z;

    return true;
    }

void MuellerPlatheHeatFlow::verifyOrthorhombicBox()
    {
    const BoxDim box = m_pdata->getGlobalBox();
    bool valid = true;
    valid &= std::fabs(box.getTiltFactorXY()) < 1e-7;
    valid &= std::fabs(box.getTiltFactorXZ()) < 1e-7;
    valid &= std::fabs(box.getTiltFactorYZ()) < 1e-7;
    if (!valid)
        {
        throw std::runtime_error("MuellerPlatheHeatFlow can only be used with orthorhombic boxes.");
        }
    m_needs_orthorhombic_check = false;
    }

Scalar MuellerPlatheHeatFlow::getSlabArea() const
    {
    const Scalar3 L = m_pdata->getGlobalBox().getL();
    switch (m_slab_direction)
        {
    case X:
        return L.y * L.z;
    case Y:
        return L.x * L.z;
    case Z:
        return L.x * L.y;
        }
    return Scalar(1);
    }

namespace detail
    {
void export_MuellerPlatheHeatFlow(pybind11::module& m)
    {
    pybind11::class_<MuellerPlatheHeatFlow, Updater, std::shared_ptr<MuellerPlatheHeatFlow>>(
        m,
        "MuellerPlatheHeatFlow")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            std::shared_ptr<Trigger>,
                            std::shared_ptr<ParticleGroup>,
                            std::shared_ptr<Variant>,
                            std::string,
                            unsigned int,
                            unsigned int,
                            unsigned int,
                            Scalar>())
        .def_property_readonly("n_slabs", &MuellerPlatheHeatFlow::getNSlabs)
        .def_property_readonly("cold_slab", &MuellerPlatheHeatFlow::getColdSlab)
        .def_property_readonly("hot_slab", &MuellerPlatheHeatFlow::getHotSlab)
        .def_property_readonly("heat_flux_target", &MuellerPlatheHeatFlow::getHeatFluxTarget)
        .def_property_readonly("slab_direction", &MuellerPlatheHeatFlow::getSlabDirectionPython)
        .def_property_readonly("heat_flux_epsilon", &MuellerPlatheHeatFlow::getHeatFluxEpsilon)
        .def_property_readonly("summed_exchanged_energy",
                               &MuellerPlatheHeatFlow::getSummedExchangedEnergy);
    }
    } // namespace detail
    } // namespace hoomd
