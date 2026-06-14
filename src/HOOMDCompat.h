#pragma once

#include "HOOMDVersion.h"

#if HOOMD_VERSION_MAJOR >= 5
#include "hoomd/GPUArray.h"
#else
#include "hoomd/GlobalArray.h"
#endif

namespace hoomd
    {

#if HOOMD_VERSION_MAJOR >= 5
template<class T> using HeatTransferArray = GPUArray<T>;
#else
template<class T> using HeatTransferArray = GlobalArray<T>;
#endif

    } // namespace hoomd
