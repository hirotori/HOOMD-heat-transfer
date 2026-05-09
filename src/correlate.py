"""Autocorrelation writer for HOOMD-blue logged quantities."""

import hoomd
from numbers import Number
import numpy as np
from . import _heat_transfer
from hoomd import operation
from hoomd.util import _dict_flatten

class Correlator(hoomd.operation.Writer):
    """Accumulate autocorrelation functions from a HOOMD logger.

    ``Correlator`` samples numeric values from ``logger.log()`` when its
    trigger fires. Scalars are used directly. One-dimensional numeric
    sequences are expanded into separate scalar channels. The C++ backend
    accumulates the unnormalized time-lag products and writes normalized
    autocorrelation data every ``output_interval`` timesteps.

    Args:
        logger (hoomd.logging.Logger): Logger containing numeric scalar or
            one-dimensional sequence quantities.
        trigger (hoomd.trigger.Trigger): Sampling trigger.
        output_interval (int): Timestep interval for writing
            ``correlation_<timestep>.dat`` files.
        max_lag (int): Number of lag points to keep in the ring buffer.
    """
    
    def __init__(self,
                 logger,
                 trigger,
                 output_interval:int,
                 max_lag:int):
        """Initialize the correlator writer."""
        
        super().__init__(trigger)
        
        self._log_wrapper = _CorrelatorLogWrapper(logger)
        self._output_interval = output_interval
        self._max_lag = max_lag
 
    def _attach_hook(self):
        """Create the C++ correlator and attach the Python log wrapper."""
        self._cpp_obj = _heat_transfer.Correlator(
            self._simulation.state._cpp_sys_def,
            self.trigger,
            self._output_interval,
            self._max_lag
        )
        self._cpp_obj.log_writer = self._log_wrapper
        super()._attach_hook()

    def correlation(self):
        """numpy.ndarray: Current normalized autocorrelation array.

        The array shape is ``(max_lag, n_values)``. Columns follow the order
        produced by the flattened logger output.
        """
        return self._cpp_obj.correlation
    
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
                result[key] = float(value)
                continue

            if isinstance(value, np.generic):
                result[key] = float(value)
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

                for i, v in enumerate(arr):
                    result[key[-1]+"_"+str(i)] = float(v)
                continue

            raise TypeError(
                f"Correlator: Unsupported type {type(value)} for key {key}"
            )

        return result
