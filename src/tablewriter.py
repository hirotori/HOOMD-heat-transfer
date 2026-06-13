"""Table writer for scalar and sequence logger quantities."""

from numbers import Number
from sys import stdout

import hoomd
import numpy as np
from hoomd.custom import Action
from hoomd.util import _dict_flatten


class SequenceTable(hoomd.write.CustomWriter):
    """Write scalar and one-dimensional sequence logger values as columns.

    ``SequenceTable`` is similar to ``hoomd.write.Table``, but it accepts
    sequence quantities. One-dimensional sequences are expanded into separate
    columns with ``_0``, ``_1``, ... suffixes.

    Args:
        logger (hoomd.logging.Logger): Logger with numeric scalar or sequence
            quantities.
        trigger (hoomd.trigger.Trigger): Write trigger.
        output: File-like object with ``write`` and ``flush`` methods.
        delimiter (str): Column delimiter.
        include_timestep (bool): When ``True``, write the current timestep as
            the first column.
        flush (bool): When ``True``, flush after each row.
    """

    def __init__(self,
                 logger,
                 trigger,
                 output=stdout,
                 delimiter=" ",
                 include_timestep=True,
                 flush=True):
        action = _SequenceTableAction(
            logger=logger,
            output=output,
            delimiter=delimiter,
            include_timestep=include_timestep,
            flush=flush,
        )
        super().__init__(trigger=trigger, action=action)


class _SequenceTableAction(Action):
    """Sample a logger and write flattened numeric values."""

    flags = [
        Action.Flags.ROTATIONAL_KINETIC_ENERGY,
        Action.Flags.PRESSURE_TENSOR,
        Action.Flags.EXTERNAL_FIELD_VIRIAL,
    ]

    def __init__(self,
                 logger,
                 output,
                 delimiter,
                 include_timestep,
                 flush):
        super().__init__()
        self._logger = logger
        self._output = output
        self._delimiter = delimiter
        self._include_timestep = include_timestep
        self._flush = flush
        self._columns = None

    def act(self, timestep):
        columns, values = self._get_row()

        if self._include_timestep:
            columns = ["timestep"] + columns
            values = [timestep] + values

        if self._columns is None:
            self._columns = columns
            self._output.write(self._delimiter.join(self._columns) + "\n")
        elif columns != self._columns:
            raise RuntimeError("SequenceTable: logged columns changed.")

        self._output.write(
            self._delimiter.join(self._format_value(value) for value in values)
            + "\n"
        )
        if self._flush:
            self._output.flush()

    def _get_row(self):
        raw = self._logger.log()
        flat = _dict_flatten(raw)

        columns = []
        values = []
        for key, logged_value in flat.items():
            value = logged_value[0]
            key_name = self._format_key(key)

            if value is None:
                raise TypeError(f"SequenceTable: {key} is None")

            if isinstance(value, str):
                raise TypeError("SequenceTable accepts only numeric values.")

            if isinstance(value, Number):
                columns.append(key_name)
                values.append(value)
                continue

            if isinstance(value, np.generic):
                columns.append(key_name)
                values.append(value.item())
                continue

            if isinstance(value, (list, tuple, np.ndarray)):
                arr = np.asarray(value)
                if not np.issubdtype(arr.dtype, np.number):
                    raise TypeError(
                        f"SequenceTable: {key} contains non-numeric values."
                    )
                if arr.ndim != 1:
                    raise TypeError(f"SequenceTable: {key} must be 1D.")
                for i, item in enumerate(arr):
                    columns.append(f"{key_name}_{i}")
                    values.append(item)
                continue

            raise TypeError(
                f"SequenceTable: unsupported type {type(value)} for {key}."
            )

        return columns, values

    @staticmethod
    def _format_key(key):
        if isinstance(key, tuple):
            return ".".join(str(part) for part in key)
        return str(key)

    @staticmethod
    def _format_value(value):
        if isinstance(value, (np.integer, int)):
            return str(int(value))
        if isinstance(value, (np.floating, float)):
            return f"{float(value):.16g}"
        return f"{float(value):.16g}"
