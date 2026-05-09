#include "hip/hip_runtime.h"

#include "ComputeHeatFluxGPU.cuh"

#ifdef ENABLE_HIP

namespace hoomd
    {
namespace kernel
    {

__global__ void gpu_compute_heat_flux_kernel(Scalar3* d_partial_kin,
                                             Scalar3* d_partial_vir,
                                             const Scalar4* d_vel,
                                             const Scalar4* d_net_force,
                                             const Scalar* d_virial,
                                             size_t virial_pitch,
                                             const unsigned int* d_group_members,
                                             unsigned int group_size)
    {
    extern __shared__ char s_data[];
    Scalar3* s_kin = reinterpret_cast<Scalar3*>(s_data);
    Scalar3* s_vir = s_kin + blockDim.x;

    const unsigned int tid = threadIdx.x;
    const unsigned int group_idx = blockIdx.x * blockDim.x + tid;

    Scalar3 J_kin = make_scalar3(0, 0, 0);
    Scalar3 J_vir = make_scalar3(0, 0, 0);

    if (group_idx < group_size)
        {
        const unsigned int i = d_group_members[group_idx];
        const Scalar4 vel = d_vel[i];
        const Scalar3 v = make_scalar3(vel.x, vel.y, vel.z);
        const Scalar m = vel.w;

        const Scalar ke = Scalar(0.5) * m * (v.x * v.x + v.y * v.y + v.z * v.z);
        const Scalar ei = ke + d_net_force[i].w;
        J_kin = ei * v;

        const Scalar Wxx = d_virial[i + 0 * virial_pitch];
        const Scalar Wxy = d_virial[i + 1 * virial_pitch];
        const Scalar Wxz = d_virial[i + 2 * virial_pitch];
        const Scalar Wyy = d_virial[i + 3 * virial_pitch];
        const Scalar Wyz = d_virial[i + 4 * virial_pitch];
        const Scalar Wzz = d_virial[i + 5 * virial_pitch];

        J_vir.x = Wxx * v.x + Wxy * v.y + Wxz * v.z;
        J_vir.y = Wxy * v.x + Wyy * v.y + Wyz * v.z;
        J_vir.z = Wxz * v.x + Wyz * v.y + Wzz * v.z;
        }

    s_kin[tid] = J_kin;
    s_vir[tid] = J_vir;
    __syncthreads();

    for (unsigned int offset = blockDim.x / 2; offset > 0; offset >>= 1)
        {
        if (tid < offset)
            {
            s_kin[tid] += s_kin[tid + offset];
            s_vir[tid] += s_vir[tid + offset];
            }
        __syncthreads();
        }

    if (tid == 0)
        {
        d_partial_kin[blockIdx.x] = s_kin[0];
        d_partial_vir[blockIdx.x] = s_vir[0];
        }
    }

hipError_t gpu_compute_heat_flux(Scalar3* d_partial_kin,
                                 Scalar3* d_partial_vir,
                                 const Scalar4* d_vel,
                                 const Scalar4* d_net_force,
                                 const Scalar* d_virial,
                                 size_t virial_pitch,
                                 const unsigned int* d_group_members,
                                 unsigned int group_size,
                                 unsigned int block_size,
                                 unsigned int num_blocks)
    {
    const dim3 grid(num_blocks, 1, 1);
    const dim3 threads(block_size, 1, 1);
    const size_t shared_bytes = 2 * block_size * sizeof(Scalar3);

    hipLaunchKernelGGL(gpu_compute_heat_flux_kernel,
                       grid,
                       threads,
                       shared_bytes,
                       0,
                       d_partial_kin,
                       d_partial_vir,
                       d_vel,
                       d_net_force,
                       d_virial,
                       virial_pitch,
                       d_group_members,
                       group_size);

    return hipGetLastError();
    }

__global__ void gpu_compute_enthalpy_flux_partial_kernel(Scalar* d_partial_count,
                                                         Scalar* d_partial_mvsq,
                                                         Scalar* d_partial_uesum,
                                                         Scalar* d_partial_ptrace,
                                                         Scalar3* d_partial_vsum,
                                                         const Scalar4* d_pos,
                                                         const Scalar4* d_vel,
                                                         const Scalar4* d_net_force,
                                                         const Scalar* d_virial,
                                                         size_t virial_pitch,
                                                         unsigned int N,
                                                         unsigned int ntypes)
    {
    extern __shared__ char s_data[];
    Scalar* s_count = reinterpret_cast<Scalar*>(s_data);
    Scalar* s_mvsq = s_count + blockDim.x;
    Scalar* s_uesum = s_mvsq + blockDim.x;
    Scalar* s_ptrace = s_uesum + blockDim.x;
    Scalar3* s_vsum = reinterpret_cast<Scalar3*>(s_ptrace + blockDim.x);

    const unsigned int tid = threadIdx.x;
    const unsigned int i = blockIdx.x * blockDim.x + tid;

    for (unsigned int type = 0; type < ntypes; ++type)
        {
        Scalar count = 0;
        Scalar mvsq = 0;
        Scalar uesum = 0;
        Scalar ptrace = 0;
        Scalar3 vsum = make_scalar3(0, 0, 0);

        if (i < N && __scalar_as_int(d_pos[i].w) == type)
            {
            const Scalar4 vel = d_vel[i];
            const Scalar3 v = make_scalar3(vel.x, vel.y, vel.z);
            const Scalar vsq = v.x * v.x + v.y * v.y + v.z * v.z;

            count = 1;
            mvsq = vel.w * vsq;
            uesum = d_net_force[i].w;
            ptrace = d_virial[i + 0 * virial_pitch] + d_virial[i + 3 * virial_pitch]
                     + d_virial[i + 5 * virial_pitch];
            vsum = v;
            }

        s_count[tid] = count;
        s_mvsq[tid] = mvsq;
        s_uesum[tid] = uesum;
        s_ptrace[tid] = ptrace;
        s_vsum[tid] = vsum;
        __syncthreads();

        for (unsigned int offset = blockDim.x / 2; offset > 0; offset >>= 1)
            {
            if (tid < offset)
                {
                s_count[tid] += s_count[tid + offset];
                s_mvsq[tid] += s_mvsq[tid + offset];
                s_uesum[tid] += s_uesum[tid + offset];
                s_ptrace[tid] += s_ptrace[tid + offset];
                s_vsum[tid] += s_vsum[tid + offset];
                }
            __syncthreads();
            }

        if (tid == 0)
            {
            const unsigned int out = type * gridDim.x + blockIdx.x;
            d_partial_count[out] = s_count[0];
            d_partial_mvsq[out] = s_mvsq[0];
            d_partial_uesum[out] = s_uesum[0];
            d_partial_ptrace[out] = s_ptrace[0];
            d_partial_vsum[out] = s_vsum[0];
            }
        __syncthreads();
        }
    }

hipError_t gpu_compute_enthalpy_flux_partial(Scalar* d_partial_count,
                                             Scalar* d_partial_mvsq,
                                             Scalar* d_partial_uesum,
                                             Scalar* d_partial_ptrace,
                                             Scalar3* d_partial_vsum,
                                             const Scalar4* d_pos,
                                             const Scalar4* d_vel,
                                             const Scalar4* d_net_force,
                                             const Scalar* d_virial,
                                             size_t virial_pitch,
                                             unsigned int N,
                                             unsigned int ntypes,
                                             unsigned int block_size,
                                             unsigned int num_blocks)
    {
    const dim3 grid(num_blocks, 1, 1);
    const dim3 threads(block_size, 1, 1);
    const size_t shared_bytes = 4 * block_size * sizeof(Scalar)
                                + block_size * sizeof(Scalar3);

    hipLaunchKernelGGL(gpu_compute_enthalpy_flux_partial_kernel,
                       grid,
                       threads,
                       shared_bytes,
                       0,
                       d_partial_count,
                       d_partial_mvsq,
                       d_partial_uesum,
                       d_partial_ptrace,
                       d_partial_vsum,
                       d_pos,
                       d_vel,
                       d_net_force,
                       d_virial,
                       virial_pitch,
                       N,
                       ntypes);

    return hipGetLastError();
    }

    } // end namespace kernel
    } // end namespace hoomd

#endif // ENABLE_HIP
