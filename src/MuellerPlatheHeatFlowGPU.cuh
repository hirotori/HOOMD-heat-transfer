#pragma once

#ifdef ENABLE_HIP

#include "hoomd/BoxDim.h"
#include "hoomd/HOOMDMath.h"
#include "hip/hip_runtime.h"

namespace hoomd
    {
namespace kernel
    {

struct HeatFlowCandidate
    {
    Scalar energy;
    Scalar mass;
    Scalar3 velocity;
    unsigned int tag;
    };

hipError_t gpu_search_heat_flow_particles(const unsigned int group_size,
                                          const Scalar4* const d_vel,
                                          const Scalar4* const d_pos,
                                          const unsigned int* const d_tag,
                                          const unsigned int* const d_group_members,
                                          const BoxDim box,
                                          const unsigned int n_slabs,
                                          const unsigned int cold_slab,
                                          const unsigned int hot_slab,
                                          const unsigned int slab_direction,
                                          HeatFlowCandidate* const hottest_cold_particle,
                                          HeatFlowCandidate* const coldest_hot_particle);

hipError_t gpu_exchange_heat_flow_velocities(const unsigned int* const d_rtag,
                                             Scalar4* const d_vel,
                                             const unsigned int n_total,
                                             const unsigned int n_global,
                                             const HeatFlowCandidate hottest_cold_particle,
                                             const HeatFlowCandidate coldest_hot_particle);

    } // namespace kernel
    } // namespace hoomd

#endif // ENABLE_HIP
