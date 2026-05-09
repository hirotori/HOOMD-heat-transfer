#pragma once

#ifdef ENABLE_HIP

#include "hoomd/HOOMDMath.h"
#include "hip/hip_runtime.h"

namespace hoomd
    {
namespace kernel
    {

hipError_t gpu_compute_heat_flux(Scalar3* d_partial_kin,
                                 Scalar3* d_partial_vir,
                                 const Scalar4* d_vel,
                                 const Scalar4* d_net_force,
                                 const Scalar* d_virial,
                                 size_t virial_pitch,
                                 const unsigned int* d_group_members,
                                 unsigned int group_size,
                                 unsigned int block_size,
                                 unsigned int num_blocks);

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
                                             unsigned int num_blocks);

    } // end namespace kernel
    } // end namespace hoomd

#endif // ENABLE_HIP
