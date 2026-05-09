"""compute auto-correlation function"""
import hoomd
from numbers import Number
import numpy as np
from . import _heat_transfer
from hoomd import operation
from hoomd.util import _dict_flatten

class Correlator(hoomd.operation.Writer):
    
    def __init__(self,
                 logger,
                 trigger,
                 output_interval:int,
                 max_lag:int):
        
        super().__init__(trigger)
        
        self._log_wrapper = _CorrelatorLogWrapper(logger)
        self._output_interval = output_interval
        self._max_lag = max_lag
 
    def _attach_hook(self):
        self._cpp_obj = _heat_transfer.Correlator(
            self._simulation.state._cpp_sys_def,
            self.trigger,
            self._output_interval,
            self._max_lag
        )
        self._cpp_obj.log_writer = self._log_wrapper
        super()._attach_hook()

    def correlation(self):
        return self._cpp_obj.correlation
    
class _CorrelatorLogWrapper:
    def __init__(self, logger):
        self._logger = logger
    
    def log(self):
        raw = self._logger.log()
        flat = _dict_flatten(raw)

        result = {}
        for key, values in flat.items():
            value = values[0]
            
            # --- None を拒否 ---
            if value is None:
                raise TypeError(f"Correlator: {key} is None")

            # --- string を拒否 ---
            if isinstance(value, str):
                raise TypeError(
                    f"Correlator: {key} is string. "
                    "Correlator accepts only numeric values."
                )

            # --- scalar ---
            if isinstance(value, Number):
                result[key] = float(value)
                continue

            # --- numpy scalar ---
            if isinstance(value, np.generic):
                result[key] = float(value)
                continue

            # --- sequence ---
            if isinstance(value, (list, tuple, np.ndarray)):
                arr = np.asarray(value)

                if not np.issubdtype(arr.dtype, np.number):
                    raise TypeError(
                        f"Correlator: {key} contains non-numeric sequence."
                    )

                # 1D に限定
                if arr.ndim != 1:
                    raise TypeError(
                        f"Correlator: {key} must be 1D sequence."
                    )

                for i, v in enumerate(arr):
                    result[key[-1]+"_"+str(i)] = float(v)
                continue

            # --- その他は拒否 ---
            raise TypeError(
                f"Correlator: Unsupported type {type(value)} for key {key}"
            )

        return result
