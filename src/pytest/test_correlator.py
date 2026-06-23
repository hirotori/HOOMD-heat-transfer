import hoomd
from hoomd import conftest
from hoomd.heat_transfer import correlate
import pytest
import numpy as np
from pathlib import Path

class DummyLogWriter:
    def __init__(self, sequence):
        self.sequence = sequence
        self.i = 0

    def log(self):
        value = self.sequence[self.i]
        self.i += 1
        return {"value": (value, "dummy")}


def test_create(simulation_factory, one_particle_snapshot_factory):

    sim = simulation_factory(one_particle_snapshot_factory())

    logger = DummyLogWriter(sequence=[0.0])

    corr = correlate.Correlator(
        logger,
        sample_interval=1,
        output_interval=10,
        max_lag=5,
    )

    assert corr is not None
    assert corr.trigger == hoomd.trigger.Periodic(1)


def test_deprecated_trigger():
    logger = DummyLogWriter(sequence=[0.0])

    with pytest.warns(FutureWarning, match="trigger is deprecated"):
        corr = correlate.Correlator(
            logger,
            trigger=hoomd.trigger.Periodic(2),
            output_interval=10,
            max_lag=5,
        )

    assert corr.trigger == hoomd.trigger.Periodic(2)

def test_trigger_runs(simulation_factory, one_particle_snapshot_factory):
    sim = simulation_factory(one_particle_snapshot_factory())

    writer = DummyLogWriter(np.arange(5, dtype=np.float64))

    corr = correlate.Correlator(
        writer,
        sample_interval=1,
        output_interval=100,
        max_lag=10,
    )

    sim.operations += corr

    sim.run(5)

def test_autocorrelation(simulation_factory, one_particle_snapshot_factory):
    sim = simulation_factory(one_particle_snapshot_factory())

    seq = np.arange(4, dtype=np.float64) + 1.0
    writer = DummyLogWriter(seq)

    corr = correlate.Correlator(
        writer,
        sample_interval=1,
        output_interval=4,
        max_lag=4,
    )

    sim.operations += corr

    sim.run(4)

    result = corr.correlation()

    expected = np.array([
        (1*1+2*2+3*3+4*4)/4, # = 7.5
        (1*2+2*3+3*4)/3,     # = 6.67
        (1*3+2*4)/2,         # = 5.5
        (1*4)/1              # = 4
    ])

    assert np.allclose(result[:,0], expected)

def test_autocorrelation_long(simulation_factory, one_particle_snapshot_factory):
    sim = simulation_factory(one_particle_snapshot_factory())

    seq = np.ones(10, dtype=np.float64)
    writer = DummyLogWriter(seq)

    corr = correlate.Correlator(
        writer,
        sample_interval=1,
        output_interval=10,
        max_lag=5,
    )

    sim.operations += corr

    sim.run(10)

    result = corr.correlation()

    expected = np.ones(5)

    assert np.allclose(result[:,0], expected)

def test_autocorrelation_linear_long(simulation_factory, one_particle_snapshot_factory):
    sim = simulation_factory(one_particle_snapshot_factory())

    N = 100
    max_lag = 10

    seq = np.arange(N, dtype=np.float64) + 1.0
    writer = DummyLogWriter(seq)
    corr = correlate.Correlator(
        writer,
        sample_interval=1,
        output_interval=N,
        max_lag=max_lag,
    )

    sim.operations += corr

    sim.run(N)

    result = corr.correlation()

    def autocorr_linear(N, tau):
        return (N - tau - 1)*(2*(N - tau) - 1)/6 + (2 + tau)*(N - tau - 1)/2+ 1 + tau
            
    expected = np.array([autocorr_linear(N, tau) for tau in range(max_lag)])

    assert np.allclose(result[:,0], expected)


def test_output_timelag_and_column_names(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    core = correlate._heat_transfer.Correlator(4, 3, 2)
    core.set_column_names(["heat_flux_0", "heat_flux_1"])

    core.accumulate(np.array([1.0, 2.0], dtype=np.float64), 0)
    core.accumulate(np.array([3.0, 4.0], dtype=np.float64), 2)
    core.accumulate(np.array([5.0, 6.0], dtype=np.float64), 4)

    lines = Path("correlation_4.dat").read_text().splitlines()

    assert (
        lines[2]
        == "# Index TimeLag Count heat_flux_0*heat_flux_0 heat_flux_1*heat_flux_1"
    )
    assert lines[3].split()[:3] == ["1", "0", "3"]
    assert lines[4].split()[:3] == ["2", "2", "2"]
    assert lines[5].split()[:3] == ["3", "4", "1"]


def test_log_wrapper_sequence_column_names():
    class VectorLogWriter:
        def log(self):
            return {"heat_flux": (np.array([1.0, 2.0]), "dummy")}

    logged = correlate._CorrelatorLogWrapper(VectorLogWriter()).log()

    assert list(logged.keys()) == ["heat_flux_0", "heat_flux_1"]
    assert list(logged.values()) == [1.0, 2.0]
