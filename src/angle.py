"""Angle force potentials for HOOMD-blue simulations."""

import hoomd
from hoomd.data.parameterdicts import TypeParameterDict
from hoomd.data.typeparam import TypeParameter
from hoomd.md.angle import Angle

from . import _heat_transfer


class Cosine(Angle):
    r"""Cosine angle force.

    `Cosine` computes forces, virials, and energies on all angles in the
    simulation state with:

    .. math::

        U(\theta) = k \left(1 + \cos\theta \right)

    Attributes:
        params (TypeParameter[``angle type``, dict]):
            The potential parameters for each angle type. The dictionary has
            the following key:

            * ``k`` (`float`, **required**) - potential constant
              :math:`k` :math:`[\mathrm{energy}]`
    """

    _ext_module = _heat_transfer
    _cpp_class_name = "CosineAngleForceCompute"

    def __init__(self):
        super().__init__()
        params = TypeParameter(
            "params", "angle_types", TypeParameterDict(k=float, len_keys=1)
        )
        self._add_typeparam(params)


__all__ = ["Cosine"]
