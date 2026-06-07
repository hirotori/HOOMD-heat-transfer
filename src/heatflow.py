"""Reverse perturbation updater for imposing heat flow."""

import hoomd
from hoomd.data.parameterdicts import ParameterDict
from hoomd.data.typeconverter import OnlyTypes
from hoomd.operation import Updater
from . import _heat_transfer


class ReversePerturbationHeatFlow(Updater):
    """Muller-Plathe reverse perturbation method for heat transport.

    The updater divides the simulation box into slabs perpendicular to
    ``slab_direction``. It selects the hottest particle in ``cold_slab`` and the
    coldest equal-mass particle in ``hot_slab`` and exchanges their full velocity
    vectors. This transfers kinetic energy from ``cold_slab`` to ``hot_slab``
    while conserving total kinetic energy and linear momentum for equal masses.

    Args:
        filter (hoomd.filter.filter_like): Particles eligible for exchange.
        heat_flux_target (hoomd.variant.variant_like): Target value for the
            time-integrated exchanged kinetic energy per slab area.
        slab_direction (str): Direction perpendicular to the slabs: ``"x"``,
            ``"y"``, or ``"z"``.
        n_slabs (int): Number of slabs along ``slab_direction``.
        hot_slab (int): Slab receiving kinetic energy. Defaults to
            ``n_slabs // 2``.
        cold_slab (int): Slab donating kinetic energy. Defaults to ``0``.
        heat_flux_epsilon (float): Tolerance for matching ``heat_flux_target``.

    Note:
        For strict energy and momentum conservation, use a filter that contains
        particles of a single mass.
    """

    def __init__(self,
                 filter,
                 heat_flux_target,
                 slab_direction,
                 n_slabs,
                 hot_slab=-1,
                 cold_slab=-1,
                 heat_flux_epsilon=1e-2):
        params = ParameterDict(
            filter=hoomd.filter.ParticleFilter,
            heat_flux_target=hoomd.variant.Variant,
            slab_direction=OnlyTypes(str,
                                     strict=True,
                                     postprocess=self._to_lowercase),
            n_slabs=OnlyTypes(int, preprocess=self._preprocess_n_slabs),
            hot_slab=OnlyTypes(int, preprocess=self._preprocess_hot_slab),
            cold_slab=OnlyTypes(int, preprocess=self._preprocess_cold_slab),
            heat_flux_epsilon=float(heat_flux_epsilon),
        )
        params.update(
            dict(filter=filter,
                 heat_flux_target=heat_flux_target,
                 slab_direction=slab_direction,
                 n_slabs=n_slabs))
        self._param_dict.update(params)
        self._param_dict.update(dict(hot_slab=hot_slab))
        self._param_dict.update(dict(cold_slab=cold_slab))

        super().__init__(hoomd.trigger.Periodic(1))

    def _to_lowercase(self, value):
        value = value.lower()
        if value not in ("x", "y", "z"):
            raise ValueError("slab_direction must be 'x', 'y', or 'z'.")
        return value

    def _preprocess_n_slabs(self, n_slabs):
        if n_slabs <= 0:
            raise ValueError(f"n_slabs must be positive, got {n_slabs}.")
        return n_slabs

    def _preprocess_hot_slab(self, hot_slab):
        if hot_slab < 0:
            hot_slab = self.n_slabs // 2
        if hot_slab < 0 or hot_slab >= self.n_slabs:
            raise ValueError(f"Invalid hot_slab of {hot_slab}.")
        return hot_slab

    def _preprocess_cold_slab(self, cold_slab):
        if cold_slab < 0:
            cold_slab = 0
        if cold_slab < 0 or cold_slab >= self.n_slabs:
            raise ValueError(f"Invalid cold_slab of {cold_slab}.")
        if cold_slab == self.hot_slab:
            raise ValueError(
                f"cold_slab and hot_slab must differ, both are {cold_slab}.")
        return cold_slab

    def _attach_hook(self):
        group = self._simulation.state._get_group(self.filter)
        self._cpp_obj = _heat_transfer.MuellerPlatheHeatFlow(
            self._simulation.state._cpp_sys_def,
            self.trigger,
            group,
            self.heat_flux_target,
            self.slab_direction,
            self.n_slabs,
            self.cold_slab,
            self.hot_slab,
            self.heat_flux_epsilon,
        )

    @hoomd.logging.log(category="scalar", requires_run=True)
    def summed_exchanged_energy(self):
        """float: Total exchanged kinetic energy accumulated by the updater."""
        return self._cpp_obj.summed_exchanged_energy
