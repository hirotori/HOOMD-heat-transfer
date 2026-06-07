import hoomd
import numpy as np
import pytest


@pytest.fixture
def simulation():
    device = hoomd.device.CPU()
    sim = hoomd.Simulation(device=device, seed=42)

    snap = hoomd.Snapshot()
    snap.particles.N = 10
    snap.particles.position[:] = np.random.rand(10,3)
    snap.particles.velocity[:] = 0.0
    snap.particles.types = ['A']

    sim.create_state_from_snapshot(snap)

    return sim


def test_zero_heatflux(simulation):
    from hoomd.heat_transfer.heatflux import ComputeheatFlux

    compute = ComputeheatFlux(filter=hoomd.filter.All())
    simulation.operations.computes.append(compute)

    simulation.run(0)

    J = compute.heatflux

    assert np.allclose(J, np.zeros(3))

def test_zero_enthalpy_flux(simulation):
    from hoomd.heat_transfer.heatflux import ComputeheatFlux

    compute = ComputeheatFlux(filter=hoomd.filter.All())
    simulation.operations.computes.append(compute)

    simulation.run(0)

    J = compute.enthalpy_flux

    assert np.allclose(J, np.zeros(3))

def test_uniform_velocity(simulation):
    from hoomd.heat_transfer.heatflux import ComputeheatFlux

    v = np.array([1.0, 0.0, 0.0])
    snapshot = simulation.state.get_snapshot()
    snapshot.particles.velocity[:] = v
    N = snapshot.particles.N
    simulation.state.set_snapshot(snapshot)

    compute = ComputeheatFlux(hoomd.filter.All())
    simulation.operations.computes.append(compute)

    simulation.run(0)

    J = np.array(compute.heatflux)

    m = 1.0

    expected = N * 0.5 * m * np.dot(v,v) * v

    assert np.allclose(J, expected)

def test_uniform_velocity_with_nonunit_mass(simulation):
    from hoomd.heat_transfer.heatflux import ComputeheatFlux

    v = np.array([0.0, 2.0, 3.0])
    m = 2.0
    snapshot = simulation.state.get_snapshot()
    snapshot.particles.velocity[:] = v
    snapshot.particles.mass[:] = m
    N = snapshot.particles.N
    simulation.state.set_snapshot(snapshot)

    compute = ComputeheatFlux(hoomd.filter.All())
    simulation.operations.computes.append(compute)

    simulation.run(0)

    J = np.array(compute.heatflux)
    expected = N * 0.5 * m * np.dot(v, v) * v

    assert np.allclose(J, expected)

def test_uniform_velocity_with_enthalpy_subtraction(simulation):
    from hoomd.heat_transfer.heatflux import ComputeheatFlux

    v = np.array([1.0, 0.0, 0.0])
    m = 1.0
    snapshot = simulation.state.get_snapshot()
    snapshot.particles.velocity[:] = v
    snapshot.particles.mass[:] = m
    N = snapshot.particles.N
    simulation.state.set_snapshot(snapshot)

    compute = ComputeheatFlux(hoomd.filter.All(), include_enthalpy=True)
    simulation.operations.computes.append(compute)

    simulation.run(0)

    J = np.array(compute.heatflux)
    J_kin = np.array(compute.kinetic_heatflux)
    J_vir = np.array(compute.virial_heatflux)
    J_h = np.array(compute.enthalpy_flux)

    expected_h = N * (5.0 / 6.0) * m * np.dot(v, v) * v

    assert np.allclose(J_h, expected_h)
    assert np.allclose(J, J_kin + J_vir - J_h)
