"""Reference partial-enthalpy flux action used by the binary sample."""

import numpy as np
from hoomd.custom import Action
from hoomd import logging

class ComputeEnthalpyFlux(Action):
    """Pure-Python reference implementation of partial-enthalpy flux.

    This action reads the CPU local snapshot, groups particles by type, and
    computes the same partial-enthalpy expression used by the compiled
    ``ComputeheatFlux`` implementation. It is intended for validation and
    inspection in the sample, not for production performance.
    """

    flags = [Action.Flags.PRESSURE_TENSOR]
    def __init__(self):
        """Initialize accumulators for instantaneous and averaged enthalpy."""
        super().__init__()
        self.h = np.zeros(3)
        self.jh = np.zeros(3)
        self.h_avg = np.zeros(3)
        self._count = 0

    def act(self, timestep):
        """Compute partial enthalpy and enthalpy flux at one timestep."""
        with self._state.cpu_local_snapshot as data:
            typeid = data.particles.typeid
            v = data.particles.velocity
            m = data.particles.mass
            ue = data.particles.net_energy
            virial = data.particles.net_virial

            ntypes = len(self._state.particle_types)

            vsq = np.sum(v*v, axis=1)
            ptrace = virial[:,0] + virial[:,3] + virial[:,5]

            count = np.bincount(typeid, minlength=ntypes)

            vsum = np.vstack([
                np.bincount(typeid, weights=v[:,0], minlength=ntypes),
                np.bincount(typeid, weights=v[:,1], minlength=ntypes),
                np.bincount(typeid, weights=v[:,2], minlength=ntypes)
            ]).T

            mvsq = np.bincount(typeid, weights=m*vsq, minlength=ntypes)
            uesum = np.bincount(typeid, weights=ue, minlength=ntypes)
            ptrace_sum = np.bincount(typeid, weights=ptrace, minlength=ntypes)

            self.h = (5.0/6.0*mvsq + uesum + ptrace_sum/3.0) / count
            self.jh = np.sum(self.h[:,None] * vsum, axis=0)

            self.h_avg += self.h
            self._count += 1
    
    @logging.log(category="sequence")
    def enthalpy_flux(self):
        """numpy.ndarray: Instantaneous partial-enthalpy flux vector."""
        return self.jh
    
    @logging.log(category="sequence")
    def partial_enthalpy(self):
        """numpy.ndarray: Instantaneous partial enthalpy for each type."""
        return self.h
    
    @logging.log(category="sequence")
    def averaged_partial_enthalpy(self):
        """numpy.ndarray: Running average of partial enthalpy by type."""
        return self.h_avg/self._count
