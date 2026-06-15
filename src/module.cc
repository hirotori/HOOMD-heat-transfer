// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

// TODO: Include the header files of classes that will be exported to Python.

#include <pybind11/pybind11.h>
#include "ComputeheatFlux.h"
#include "ComputeHeatFluxGPU.h"
#include "Correlator.h"
#include "MuellerPlatheHeatFlow.h"
#include "MuellerPlatheHeatFlowGPU.h"

namespace hoomd
    {
namespace md
    {
    using namespace detail;
// TODO: Set the name of the python module to match ${COMPONENT_NAME} (set in
// CMakeLists.txt), prefixed with an underscore.
PYBIND11_MODULE(_heat_transfer, m)
    {
        // TODO: Call export_Class(m) for each C++ class to be exported to Python.
        export_ComputeHeatFlux(m);
        export_MuellerPlatheHeatFlow(m);
        
#ifdef ENABLE_HIP
        // TODO: Call export_ClassGPU(m) for each GPU enabled C++ class to be exported
        // to Python.
        export_ComputeHeatFluxGPU(m);
        export_MuellerPlatheHeatFlowGPU(m);
#endif
        export_Correlator(m);

    }

    } // end namespace md
    } // end namespace hoomd
