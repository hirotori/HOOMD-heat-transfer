#include "hip/hip_runtime.h"

#include "MuellerPlatheHeatFlowGPU.cuh"

#ifdef ENABLE_HIP

#include <limits>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
#include <thrust/device_ptr.h>
#include <thrust/transform_reduce.h>
#pragma GCC diagnostic pop

namespace hoomd
    {
namespace kernel
    {
namespace
    {
    constexpr unsigned int invalid_tag = std::numeric_limits<unsigned int>::max();

    __host__ __device__ HeatFlowCandidate make_invalid_candidate()
        {
        HeatFlowCandidate candidate;
        candidate.energy = Scalar(0);
        candidate.mass = Scalar(0);
        candidate.velocity = make_scalar3(0, 0, 0);
        candidate.tag = invalid_tag;
        return candidate;
        }

    __host__ __device__ Scalar compute_kinetic_energy(const Scalar4& vel)
        {
        return Scalar(0.5) * vel.w * (vel.x * vel.x + vel.y * vel.y + vel.z * vel.z);
        }

    __host__ __device__ unsigned int compute_slab_index(const Scalar4& pos,
                                                        const BoxDim& box,
                                                        const unsigned int n_slabs,
                                                        const unsigned int slab_direction)
        {
        const Scalar3 L = box.getL();
        Scalar fraction = Scalar(0);
        if (slab_direction == 0)
            {
            fraction = pos.x / L.x + Scalar(0.5);
            }
        else if (slab_direction == 1)
            {
            fraction = pos.y / L.y + Scalar(0.5);
            }
        else
            {
            fraction = pos.z / L.z + Scalar(0.5);
            }

        unsigned int slab = static_cast<unsigned int>(fraction * static_cast<Scalar>(n_slabs));
        slab %= n_slabs;
        return slab;
        }

    struct make_heat_flow_candidate
        {
        make_heat_flow_candidate(const Scalar4* const d_vel,
                                 const Scalar4* const d_pos,
                                 const unsigned int* const d_tag,
                                 const BoxDim box,
                                 const unsigned int n_slabs,
                                 const unsigned int target_slab,
                                 const unsigned int slab_direction)
            : m_vel(d_vel), m_pos(d_pos), m_tag(d_tag), m_box(box), m_n_slabs(n_slabs),
              m_target_slab(target_slab), m_slab_direction(slab_direction)
            {
            }

        const Scalar4* const m_vel;
        const Scalar4* const m_pos;
        const unsigned int* const m_tag;
        const BoxDim m_box;
        const unsigned int m_n_slabs;
        const unsigned int m_target_slab;
        const unsigned int m_slab_direction;

        __host__ __device__ HeatFlowCandidate operator()(const unsigned int particle_idx) const
            {
            const unsigned int slab = compute_slab_index(m_pos[particle_idx],
                                                         m_box,
                                                         m_n_slabs,
                                                         m_slab_direction);
            if (slab != m_target_slab)
                {
                return make_invalid_candidate();
                }

            const Scalar4 vel = m_vel[particle_idx];
            HeatFlowCandidate candidate;
            candidate.energy = compute_kinetic_energy(vel);
            candidate.mass = vel.w;
            candidate.velocity = make_scalar3(vel.x, vel.y, vel.z);
            candidate.tag = m_tag[particle_idx];
            return candidate;
            }
        };

    struct choose_hotter_candidate
        {
        __host__ __device__ HeatFlowCandidate operator()(const HeatFlowCandidate& a,
                                                         const HeatFlowCandidate& b) const
            {
            if (a.tag == invalid_tag)
                return b;
            if (b.tag == invalid_tag)
                return a;
            if (a.energy > b.energy)
                return a;
            if (b.energy > a.energy)
                return b;
            return a.tag < b.tag ? a : b;
            }
        };

    struct choose_colder_candidate
        {
        __host__ __device__ HeatFlowCandidate operator()(const HeatFlowCandidate& a,
                                                         const HeatFlowCandidate& b) const
            {
            if (a.tag == invalid_tag)
                return b;
            if (b.tag == invalid_tag)
                return a;
            if (a.energy < b.energy)
                return a;
            if (b.energy < a.energy)
                return b;
            return a.tag < b.tag ? a : b;
            }
        };
    } // namespace

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
                                          HeatFlowCandidate* const coldest_hot_particle)
    {
    thrust::device_ptr<const unsigned int> member_ptr(d_group_members);
    const HeatFlowCandidate init = make_invalid_candidate();

    *hottest_cold_particle = thrust::transform_reduce(member_ptr,
                                                      member_ptr + group_size,
                                                      make_heat_flow_candidate(d_vel,
                                                                               d_pos,
                                                                               d_tag,
                                                                               box,
                                                                               n_slabs,
                                                                               cold_slab,
                                                                               slab_direction),
                                                      init,
                                                      choose_hotter_candidate());

    *coldest_hot_particle = thrust::transform_reduce(member_ptr,
                                                     member_ptr + group_size,
                                                     make_heat_flow_candidate(d_vel,
                                                                              d_pos,
                                                                              d_tag,
                                                                              box,
                                                                              n_slabs,
                                                                              hot_slab,
                                                                              slab_direction),
                                                     init,
                                                     choose_colder_candidate());

    return hipPeekAtLastError();
    }

__global__ void gpu_exchange_heat_flow_velocities_kernel(
    const unsigned int* const d_rtag,
    Scalar4* const d_vel,
    const unsigned int n_total,
    const unsigned int n_global,
    const HeatFlowCandidate hottest_cold_particle,
    const HeatFlowCandidate coldest_hot_particle)
    {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx != 0)
        return;

    if (hottest_cold_particle.tag == invalid_tag || coldest_hot_particle.tag == invalid_tag)
        return;
    if (hottest_cold_particle.tag >= n_global || coldest_hot_particle.tag >= n_global)
        return;

    const unsigned int cold_idx = d_rtag[hottest_cold_particle.tag];
    const unsigned int hot_idx = d_rtag[coldest_hot_particle.tag];
    if (cold_idx >= n_total || hot_idx >= n_total)
        return;

    d_vel[cold_idx].x = coldest_hot_particle.velocity.x;
    d_vel[cold_idx].y = coldest_hot_particle.velocity.y;
    d_vel[cold_idx].z = coldest_hot_particle.velocity.z;

    d_vel[hot_idx].x = hottest_cold_particle.velocity.x;
    d_vel[hot_idx].y = hottest_cold_particle.velocity.y;
    d_vel[hot_idx].z = hottest_cold_particle.velocity.z;
    }

hipError_t gpu_exchange_heat_flow_velocities(const unsigned int* const d_rtag,
                                             Scalar4* const d_vel,
                                             const unsigned int n_total,
                                             const unsigned int n_global,
                                             const HeatFlowCandidate hottest_cold_particle,
                                             const HeatFlowCandidate coldest_hot_particle)
    {
    hipLaunchKernelGGL(gpu_exchange_heat_flow_velocities_kernel,
                       dim3(1, 1, 1),
                       dim3(1, 1, 1),
                       0,
                       0,
                       d_rtag,
                       d_vel,
                       n_total,
                       n_global,
                       hottest_cold_particle,
                       coldest_hot_particle);

    return hipGetLastError();
    }

    } // namespace kernel
    } // namespace hoomd

#endif // ENABLE_HIP
