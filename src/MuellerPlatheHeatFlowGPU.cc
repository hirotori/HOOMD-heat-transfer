#include "MuellerPlatheHeatFlowGPU.h"

#ifdef ENABLE_HIP

#include "MuellerPlatheHeatFlowGPU.cuh"
#include "hoomd/GlobalArray.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace hoomd
    {
namespace
    {
    constexpr unsigned int invalid_tag = std::numeric_limits<unsigned int>::max();
    } // namespace

MuellerPlatheHeatFlowGPU::MuellerPlatheHeatFlowGPU(
    std::shared_ptr<SystemDefinition> sysdef,
    std::shared_ptr<Trigger> trigger,
    std::shared_ptr<ParticleGroup> group,
    std::shared_ptr<Variant> heat_flux_target,
    std::string slab_direction_str,
    unsigned int n_slabs,
    unsigned int cold_slab,
    unsigned int hot_slab,
    Scalar heat_flux_epsilon)
    : MuellerPlatheHeatFlow(sysdef,
                            trigger,
                            group,
                            heat_flux_target,
                            slab_direction_str,
                            n_slabs,
                            cold_slab,
                            hot_slab,
                            heat_flux_epsilon)
    {
    m_exec_conf->msg->notice(5) << "Constructing MuellerPlatheHeatFlowGPU" << std::endl;
    if (!m_exec_conf->isCUDAEnabled())
        {
        m_exec_conf->msg->error()
            << "MuellerPlatheHeatFlowGPU cannot run without a GPU execution configuration"
            << std::endl;
        throw std::runtime_error("Error initializing MuellerPlatheHeatFlowGPU");
        }

    m_tuner.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(m_exec_conf)},
                                   m_exec_conf,
                                   "muellerplatheheatflow",
                                   5,
                                   true));
    m_autotuners.push_back(m_tuner);
    }

MuellerPlatheHeatFlowGPU::~MuellerPlatheHeatFlowGPU()
    {
    m_exec_conf->msg->notice(5) << "Destroying MuellerPlatheHeatFlowGPU" << std::endl;
    }

void MuellerPlatheHeatFlowGPU::searchHotColdParticles()
    {
    const unsigned int group_size = m_group->getNumMembers();
    if (group_size == 0)
        return;

    const ArrayHandle<Scalar4> d_vel(m_pdata->getVelocities(),
                                     access_location::device,
                                     access_mode::read);
    const ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(),
                                     access_location::device,
                                     access_mode::read);
    const ArrayHandle<unsigned int> d_tag(m_pdata->getTags(),
                                          access_location::device,
                                          access_mode::read);
    const GlobalArray<unsigned int>& group_members = m_group->getIndexArray();
    const ArrayHandle<unsigned int> d_group_members(group_members,
                                                    access_location::device,
                                                    access_mode::read);

    kernel::HeatFlowCandidate hottest_cold;
    kernel::HeatFlowCandidate coldest_hot;

    m_tuner->begin();
    const hipError_t error = kernel::gpu_search_heat_flow_particles(
        group_size,
        d_vel.data,
        d_pos.data,
        d_tag.data,
        d_group_members.data,
        m_pdata->getGlobalBox(),
        getNSlabs(),
        getColdSlab(),
        getHotSlab(),
        static_cast<unsigned int>(m_slab_direction),
        &hottest_cold,
        &coldest_hot);
    m_tuner->end();

    if (error != hipSuccess)
        {
        m_exec_conf->msg->error() << "Error launching gpu_search_heat_flow_particles: "
                                  << hipGetErrorString(error) << std::endl;
        throw std::runtime_error("Error searching heat-flow particles on GPU");
        }
    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

    m_hottest_cold_particle.energy = hottest_cold.energy;
    m_hottest_cold_particle.mass = hottest_cold.mass;
    m_hottest_cold_particle.velocity = hottest_cold.velocity;
    m_hottest_cold_particle.tag = hottest_cold.tag;

    m_coldest_hot_particle.energy = coldest_hot.energy;
    m_coldest_hot_particle.mass = coldest_hot.mass;
    m_coldest_hot_particle.velocity = coldest_hot.velocity;
    m_coldest_hot_particle.tag = coldest_hot.tag;
    }

bool MuellerPlatheHeatFlowGPU::exchangeVelocities(uint64_t timestep, Scalar& exchanged_energy)
    {
    if (m_hottest_cold_particle.tag == invalid_tag || m_coldest_hot_particle.tag == invalid_tag)
        {
        m_exec_conf->msg->warning() << "WARNING: at time " << timestep
                                    << " MuellerPlatheHeatFlowGPU could not find a hot/cold pair."
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
            << " MuellerPlatheHeatFlowGPU selected particles with different masses. "
            << "Use a filter containing equal-mass particles to conserve energy and momentum."
            << std::endl;
        return false;
        }

    exchanged_energy = m_hottest_cold_particle.energy - m_coldest_hot_particle.energy;
    if (exchanged_energy <= Scalar(0))
        {
        m_exec_conf->msg->notice(6)
            << "At time " << timestep
            << " MuellerPlatheHeatFlowGPU found no positive kinetic-energy transfer."
            << std::endl;
        return false;
        }

    const uint64_t n_global_64 = m_pdata->getNGlobal();
    if (n_global_64 > static_cast<uint64_t>(std::numeric_limits<unsigned int>::max()))
        {
        throw std::runtime_error("MuellerPlatheHeatFlowGPU requires particle tags to fit in unsigned int.");
        }
    const unsigned int n_global = static_cast<unsigned int>(n_global_64);
    if (m_hottest_cold_particle.tag >= n_global || m_coldest_hot_particle.tag >= n_global)
        {
        m_exec_conf->msg->warning()
            << "WARNING: at time " << timestep
            << " MuellerPlatheHeatFlowGPU selected a tag outside the global particle range."
            << std::endl;
        return false;
        }

    const ArrayHandle<unsigned int> d_rtag(m_pdata->getRTags(),
                                           access_location::device,
                                           access_mode::read);
    ArrayHandle<Scalar4> d_vel(m_pdata->getVelocities(),
                               access_location::device,
                               access_mode::readwrite);
    const unsigned int n_total = m_pdata->getN() + m_pdata->getNGhosts();

    const kernel::HeatFlowCandidate hottest_cold{m_hottest_cold_particle.energy,
                                                 m_hottest_cold_particle.mass,
                                                 m_hottest_cold_particle.velocity,
                                                 m_hottest_cold_particle.tag};
    const kernel::HeatFlowCandidate coldest_hot{m_coldest_hot_particle.energy,
                                                m_coldest_hot_particle.mass,
                                                m_coldest_hot_particle.velocity,
                                                m_coldest_hot_particle.tag};

    const hipError_t error = kernel::gpu_exchange_heat_flow_velocities(d_rtag.data,
                                                                       d_vel.data,
                                                                       n_total,
                                                                       n_global,
                                                                       hottest_cold,
                                                                       coldest_hot);
    if (error != hipSuccess)
        {
        m_exec_conf->msg->error() << "Error launching gpu_exchange_heat_flow_velocities: "
                                  << hipGetErrorString(error) << std::endl;
        throw std::runtime_error("Error exchanging heat-flow velocities on GPU");
        }
    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

    return true;
    }

namespace detail
    {
void export_MuellerPlatheHeatFlowGPU(pybind11::module& m)
    {
    pybind11::class_<MuellerPlatheHeatFlowGPU,
                     MuellerPlatheHeatFlow,
                     std::shared_ptr<MuellerPlatheHeatFlowGPU>>(m, "MuellerPlatheHeatFlowGPU")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            std::shared_ptr<Trigger>,
                            std::shared_ptr<ParticleGroup>,
                            std::shared_ptr<Variant>,
                            std::string,
                            unsigned int,
                            unsigned int,
                            unsigned int,
                            Scalar>());
    }
    } // namespace detail
    } // namespace hoomd

#endif // ENABLE_HIP
