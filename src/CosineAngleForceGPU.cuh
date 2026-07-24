// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef __COSINEANGLEFORCEGPU_CUH__
#define __COSINEANGLEFORCEGPU_CUH__

#ifdef ENABLE_HIP

#include "hoomd/BondedGroupData.cuh"
#include "hoomd/HOOMDMath.h"
#include "hoomd/ParticleData.cuh"

namespace hoomd
    {
namespace md
    {
namespace kernel
    {
hipError_t gpu_compute_cosine_angle_forces(Scalar4* d_force,
                                           Scalar* d_virial,
                                           const size_t virial_pitch,
                                           const unsigned int N,
                                           const Scalar4* d_pos,
                                           const BoxDim& box,
                                           const group_storage<3>* atable,
                                           const unsigned int* apos_list,
                                           const unsigned int pitch,
                                           const unsigned int* n_angles_list,
                                           Scalar* d_params,
                                           unsigned int n_angle_types,
                                           int block_size);

    } // namespace kernel
    } // namespace md
    } // namespace hoomd

#endif // ENABLE_HIP

#endif // __COSINEANGLEFORCEGPU_CUH__
