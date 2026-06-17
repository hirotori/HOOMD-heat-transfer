"""Autocorrelation writer for HOOMD-blue logged quantities."""

import hoomd
from numbers import Number
import numpy as np
from . import _heat_transfer
from hoomd.custom import Action
from hoomd.util import _dict_flatten

class Correlator(hoomd.write.CustomWriter):
    """Accumulate autocorrelation functions from a HOOMD logger.

    ``Correlator`` samples numeric values from ``logger.log()`` when its
    trigger fires. Scalars are used directly. One-dimensional numeric
    sequences are expanded into separate scalar channels. The Python writer
    reads the logger and passes numeric values to the C++ backend, which
    accumulates the unnormalized time-lag products and writes normalized
    autocorrelation data every ``output_interval`` timesteps.

    Args:
        logger (hoomd.logging.Logger): Logger containing numeric scalar or
            one-dimensional sequence quantities.
        trigger (hoomd.trigger.Trigger): Sampling trigger.
        output_interval (int): Timestep interval for writing
            ``correlation_<timestep>.dat`` files.
        max_lag (int): Number of lag points to keep in the ring buffer.
        sample_interval (int): Timestep interval between samples. When
            ``None``, this is inferred from ``trigger.period`` when present
            and otherwise defaults to 1.
    """
    
    def __init__(self,
                 logger,
                 trigger,
                 output_interval:int,
                 max_lag:int,
                 sample_interval=None):
        """Initialize the correlator writer."""

        action = _CorrelatorAction(
            logger=logger,
            output_interval=output_interval,
            max_lag=max_lag,
            sample_interval=_infer_sample_interval(trigger, sample_interval),
        )
        super().__init__(trigger=trigger, action=action)

    def correlation(self):
        """numpy.ndarray: Current normalized autocorrelation array.

        The array shape is ``(max_lag, n_values)``. Columns follow the order
        produced by the flattened logger output.
        """
        return self._action.correlation()


class _CorrelatorAction(Action):
    """Python action that samples a logger and updates a C++ correlator core."""

    flags = [
        Action.Flags.ROTATIONAL_KINETIC_ENERGY,
        Action.Flags.PRESSURE_TENSOR,
        Action.Flags.EXTERNAL_FIELD_VIRIAL,
    ]

    def __init__(self, logger, output_interval: int, max_lag: int, sample_interval: int):
        super().__init__()
        self._log_wrapper = _CorrelatorLogWrapper(logger)
        self._core = _heat_transfer.Correlator(
            output_interval,
            max_lag,
            sample_interval,
        )
        self._nvalues = None
        self._column_names = None

    def act(self, timestep):
        """Sample the logger and accumulate autocorrelation values."""
        logged_values = self._log_wrapper.log()
        values = np.asarray(
            list(logged_values.values()),
            dtype=np.float64,
            order="C",
        )
        column_names = list(logged_values.keys())

        if self._nvalues is None:
            self._nvalues = values.size
            self._column_names = column_names
            self._core.set_column_names(column_names)
        elif values.size != self._nvalues:
            raise RuntimeError("Correlator: logged value count changed.")
        elif column_names != self._column_names:
            raise RuntimeError("Correlator: logged value names changed.")

        if not np.all(np.isfinite(values)):
            raise RuntimeError(
                f"Correlator: non-finite values at timestep {timestep}: {values}"
            )

        self._core.accumulate(values, timestep)

    def correlation(self):
        """numpy.ndarray: Current normalized autocorrelation array."""
        return self._core.correlation
    
class _CorrelatorLogWrapper:
    """Validate and flatten HOOMD logger output for the C++ correlator."""

    def __init__(self, logger):
        """Store the HOOMD logger to sample."""
        self._logger = logger
    
    def log(self):
        """Return flattened numeric log values as a Python dictionary.

        Raises:
            TypeError: If any logged value is ``None``, a string,
                non-numeric, or a multidimensional sequence.
        """
        raw = self._logger.log()
        flat = _dict_flatten(raw)

        result = {}
        for key, values in flat.items():
            value = values[0]
            
            if value is None:
                raise TypeError(f"Correlator: {key} is None")

            if isinstance(value, str):
                raise TypeError(
                    f"Correlator: {key} is string. "
                    "Correlator accepts only numeric values."
                )

            if isinstance(value, Number):
                result[_format_log_key(key)] = float(value)
                continue

            if isinstance(value, np.generic):
                result[_format_log_key(key)] = float(value)
                continue

            if isinstance(value, (list, tuple, np.ndarray)):
                arr = np.asarray(value)

                if not np.issubdtype(arr.dtype, np.number):
                    raise TypeError(
                        f"Correlator: {key} contains non-numeric sequence."
                    )

                if arr.ndim != 1:
                    raise TypeError(
                        f"Correlator: {key} must be 1D sequence."
                    )

                base_name = _format_log_key(key)
                for i, v in enumerate(arr):
                    result[f"{base_name}_{i}"] = float(v)
                continue

            raise TypeError(
                f"Correlator: Unsupported type {type(value)} for key {key}"
            )

        return result


def _infer_sample_interval(trigger, sample_interval):
    """Return a positive integer timestep interval between samples."""
    if sample_interval is None:
        sample_interval = getattr(trigger, "period", 1)

    sample_interval = int(sample_interval)
    if sample_interval <= 0:
        raise ValueError("Correlator: sample_interval must be positive.")

    return sample_interval


def _format_log_key(key):
    """Convert flattened HOOMD logger keys into file-safe column names."""
    if isinstance(key, tuple):
        parts = key
    else:
        parts = (key,)

    return ".".join(str(part).replace(" ", "_") for part in parts)
