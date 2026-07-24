// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hip/hip_runtime.h"

#include "CosineAngleForceGPU.cuh"
#include "hoomd/TextureTools.h"

#include <assert.h>

namespace hoomd
    {
namespace md
    {
namespace kernel
    {
__global__ void gpu_compute_cosine_angle_forces_kernel(Scalar4* d_force,
                                                       Scalar* d_virial,
                                                       const size_t virial_pitch,
                                                       const unsigned int N,
                                                       const Scalar4* d_pos,
                                                       const Scalar* d_params,
                                                       BoxDim box,
                                                       const group_storage<3>* alist,
                                                       const unsigned int* apos_list,
                                                       const unsigned int pitch,
                                                       const unsigned int* n_angles_list)
    {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N)
        return;

    int n_angles = n_angles_list[idx];

    Scalar4 idx_postype = d_pos[idx];
    Scalar3 idx_pos = make_scalar3(idx_postype.x, idx_postype.y, idx_postype.z);
    Scalar3 a_pos, b_pos, c_pos;

    Scalar4 force_idx = make_scalar4(Scalar(0.0), Scalar(0.0), Scalar(0.0), Scalar(0.0));

    Scalar virial[6];
    for (int i = 0; i < 6; i++)
        virial[i] = Scalar(0.0);

    for (int angle_idx = 0; angle_idx < n_angles; angle_idx++)
        {
        group_storage<3> cur_angle = alist[pitch * angle_idx + idx];

        int cur_angle_x_idx = cur_angle.idx[0];
        int cur_angle_y_idx = cur_angle.idx[1];
        int cur_angle_type = cur_angle.idx[2];

        int cur_angle_abc = apos_list[pitch * angle_idx + idx];

        Scalar4 x_postype = d_pos[cur_angle_x_idx];
        Scalar3 x_pos = make_scalar3(x_postype.x, x_postype.y, x_postype.z);

        Scalar4 y_postype = d_pos[cur_angle_y_idx];
        Scalar3 y_pos = make_scalar3(y_postype.x, y_postype.y, y_postype.z);

        if (cur_angle_abc == 0)
            {
            a_pos = idx_pos;
            b_pos = x_pos;
            c_pos = y_pos;
            }
        if (cur_angle_abc == 1)
            {
            b_pos = idx_pos;
            a_pos = x_pos;
            c_pos = y_pos;
            }
        if (cur_angle_abc == 2)
            {
            c_pos = idx_pos;
            a_pos = x_pos;
            b_pos = y_pos;
            }

        Scalar3 dab = a_pos - b_pos;
        Scalar3 dcb = c_pos - b_pos;

        dab = box.minImage(dab);
        dcb = box.minImage(dcb);

        Scalar K = __ldg(d_params + cur_angle_type);

        Scalar rsqab = dot(dab, dab);
        Scalar rab = fast::sqrt(rsqab);
        Scalar rsqcb = dot(dcb, dcb);
        Scalar rcb = fast::sqrt(rsqcb);

        Scalar c_abbc = dot(dab, dcb);
        c_abbc /= rab * rcb;

        if (c_abbc > Scalar(1.0))
            c_abbc = Scalar(1.0);
        if (c_abbc < -Scalar(1.0))
            c_abbc = -Scalar(1.0);

        Scalar a11 = K * c_abbc / rsqab;
        Scalar a12 = -K / (rab * rcb);
        Scalar a22 = K * c_abbc / rsqcb;

        Scalar fab[3], fcb[3];

        fab[0] = a11 * dab.x + a12 * dcb.x;
        fab[1] = a11 * dab.y + a12 * dcb.y;
        fab[2] = a11 * dab.z + a12 * dcb.z;

        fcb[0] = a22 * dcb.x + a12 * dab.x;
        fcb[1] = a22 * dcb.y + a12 * dab.y;
        fcb[2] = a22 * dcb.z + a12 * dab.z;

        Scalar angle_eng = K * (Scalar(1.0) + c_abbc) * Scalar(1.0 / 3.0);

        Scalar angle_virial[6];
        angle_virial[0] = Scalar(1. / 3.) * (dab.x * fab[0] + dcb.x * fcb[0]);
        angle_virial[1] = Scalar(1. / 3.) * (dab.y * fab[0] + dcb.y * fcb[0]);
        angle_virial[2] = Scalar(1. / 3.) * (dab.z * fab[0] + dcb.z * fcb[0]);
        angle_virial[3] = Scalar(1. / 3.) * (dab.y * fab[1] + dcb.y * fcb[1]);
        angle_virial[4] = Scalar(1. / 3.) * (dab.z * fab[1] + dcb.z * fcb[1]);
        angle_virial[5] = Scalar(1. / 3.) * (dab.z * fab[2] + dcb.z * fcb[2]);

        if (cur_angle_abc == 0)
            {
            force_idx.x += fab[0];
            force_idx.y += fab[1];
            force_idx.z += fab[2];
            }
        if (cur_angle_abc == 1)
            {
            force_idx.x -= fab[0] + fcb[0];
            force_idx.y -= fab[1] + fcb[1];
            force_idx.z -= fab[2] + fcb[2];
            }
        if (cur_angle_abc == 2)
            {
            force_idx.x += fcb[0];
            force_idx.y += fcb[1];
            force_idx.z += fcb[2];
            }

        force_idx.w += angle_eng;

        for (int i = 0; i < 6; i++)
            virial[i] += angle_virial[i];
        }

    d_force[idx] = force_idx;
    for (int i = 0; i < 6; i++)
        d_virial[i * virial_pitch + idx] = virial[i];
    }

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
                                           int block_size)
    {
    assert(d_params);

    unsigned int max_block_size;
    hipFuncAttributes attr;
    hipFuncGetAttributes(&attr, (const void*)gpu_compute_cosine_angle_forces_kernel);
    max_block_size = attr.maxThreadsPerBlock;

    unsigned int run_block_size = min(block_size, max_block_size);

    dim3 grid(N / run_block_size + 1, 1, 1);
    dim3 threads(run_block_size, 1, 1);

    hipLaunchKernelGGL((gpu_compute_cosine_angle_forces_kernel),
                       dim3(grid),
                       dim3(threads),
                       0,
                       0,
                       d_force,
                       d_virial,
                       virial_pitch,
                       N,
                       d_pos,
                       d_params,
                       box,
                       atable,
                       apos_list,
                       pitch,
                       n_angles_list);

    return hipSuccess;
    }

    } // namespace kernel
    } // namespace md
    } // namespace hoomd
