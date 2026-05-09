"""Heat-flux compute for HOOMD-blue simulations."""

import hoomd
from . import _heat_transfer
from hoomd import operation

class ComputeheatFlux(operation.Compute):
    """Compute microscopic heat flux.

    Args:
        filter (hoomd.filter.ParticleFilter): Particle group used for the
            kinetic and virial heat-flux contributions.
        include_enthalpy (bool): When ``True``, add the partial-enthalpy flux
            contribution. The partial enthalpy is accumulated by particle type
            over all local particles, matching the C++ implementation.

    Logged sequence quantities:
        `heatflux`
            Total heat flux, `kinetic_heatflux + virial_heatflux + enthalpy_flux`.
        `kinetic_heatflux`
            Convective energy flux.
        `virial_heatflux`
            Virial contribution to the heat flux.
        `enthalpy_flux`
            Partial-enthalpy contribution. This is zero unless
            `include_enthalpy=True`.
    """

    def __init__(self, filter, include_enthalpy: bool = False) -> None:
        """Initialize the heat-flux compute."""
        super().__init__()
        self._filter = filter
        self._include_enthalpy = include_enthalpy
    
    def _attach_hook(self):
        """Create the matching C++ compute for the active HOOMD device."""
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
        """list[float]: Total heat-flux vector."""
        self._cpp_obj.compute(self._simulation.timestep)
        return self._cpp_obj.heatflux

    @hoomd.logging.log(category="sequence", requires_run=True)
    def kinetic_heatflux(self):
        """list[float]: Kinetic contribution to the heat-flux vector."""
        self._cpp_obj.compute(self._simulation.timestep)
        return self._cpp_obj.kinetic_heatflux

    @hoomd.logging.log(category="sequence", requires_run=True)
    def virial_heatflux(self):
        """list[float]: Virial contribution to the heat-flux vector."""
        self._cpp_obj.compute(self._simulation.timestep)
        return self._cpp_obj.virial_heatflux
    
    @hoomd.logging.log(category="sequence", requires_run=True)
    def enthalpy_flux(self):
        """list[float]: Partial-enthalpy contribution to the heat-flux vector."""
        self._cpp_obj.compute(self._simulation.timestep)
        return self._cpp_obj.enthalpy_flux
