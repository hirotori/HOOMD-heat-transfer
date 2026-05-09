"""compute heat flux"""
import hoomd
from . import _heat_transfer
from hoomd import operation

class ComputeheatFlux(operation.Compute):
    def __init__(self, filter, include_enthalpy:bool=False) -> None:
        super().__init__()
        self._filter = filter
        self._include_enthalpy = include_enthalpy
    
    def _attach_hook(self):
        sim = self._simulation
        group = self._simulation.state._get_group(self._filter)

        if isinstance(sim.device, hoomd.device.CPU):
            cpp_class = _heat_transfer.ComputeHeatFlux
        else:
            cpp_class = getattr(
                _heat_transfer, "ComputeHeatFluxGPU", _heat_transfer.ComputeHeatFlux
            )

        self._cpp_obj = cpp_class(
            sim.state._cpp_sys_def,
            group,
            self._include_enthalpy
        )

        super()._attach_hook()

    @hoomd.logging.log(category="sequence", requires_run=True)
    def heatflux(self):
        self._cpp_obj.compute(self._simulation.timestep)
        return self._cpp_obj.heatflux

    @hoomd.logging.log(category="sequence", requires_run=True)
    def kinetic_heatflux(self):
        self._cpp_obj.compute(self._simulation.timestep)
        return self._cpp_obj.kinetic_heatflux

    @hoomd.logging.log(category="sequence", requires_run=True)
    def virial_heatflux(self):
        self._cpp_obj.compute(self._simulation.timestep)
        return self._cpp_obj.virial_heatflux
    
    @hoomd.logging.log(category="sequence", requires_run=True)
    def enthalpy_flux(self):
        self._cpp_obj.compute(self._simulation.timestep)
        return self._cpp_obj.enthalpy_flux
