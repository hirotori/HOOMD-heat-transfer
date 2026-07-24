import hoomd
import numpy as np
import pytest


@pytest.fixture(scope="session")
def triplet_snapshot_factory():
    def make_snapshot(d=1.0, theta_deg=60, dimensions=3, L=20):
        theta_rad = np.deg2rad(theta_deg)
        snapshot = hoomd.Snapshot()
        box = [L, L, L, 0, 0, 0]
        if dimensions == 2:
            box[2] = 0
        snapshot.configuration.box = box
        snapshot.particles.N = 3
        snapshot.particles.types = ["A"]
        snapshot.particles.position[:] = [
            [-d * np.sin(theta_rad / 2), d * np.cos(theta_rad / 2), 0.0],
            [0.0, 0.0, 0.0],
            [d * np.sin(theta_rad / 2), d * np.cos(theta_rad / 2), 0.0],
        ]
        snapshot.angles.N = 1
        snapshot.angles.types = ["A-A-A"]
        snapshot.angles.typeid[0] = 0
        snapshot.angles.group[0] = (0, 1, 2)
        return snapshot

    return make_snapshot


def _make_simulation(snapshot, device=None):
    if device is None:
        device = hoomd.device.CPU()
    sim = hoomd.Simulation(device=device, seed=42)
    sim.create_state_from_snapshot(snapshot)
    return sim


def _make_gpu_device():
    try:
        return hoomd.device.GPU()
    except RuntimeError as err:
        pytest.skip(f"GPU device is not available: {err}")


def _assert_forces_and_energies(triplet_snapshot_factory, device):
    from hoomd.heat_transfer.angle import Cosine

    theta_deg = 60
    theta_rad = np.deg2rad(theta_deg)
    k = 3.0
    sim = _make_simulation(
        triplet_snapshot_factory(theta_deg=theta_deg), device=device
    )

    potential = Cosine()
    potential.params["A-A-A"] = dict(k=k)
    sim.operations.integrator = hoomd.md.Integrator(dt=0.005, forces=[potential])

    sim.run(0)

    force = -k * np.sin(theta_rad)
    force_array = force * np.asarray([np.cos(theta_rad / 2), np.sin(theta_rad / 2), 0])
    energy = k * (1 + np.cos(theta_rad))

    if sim.device.communicator.rank == 0:
        assert potential.energy == pytest.approx(energy, rel=1e-5)
        np.testing.assert_allclose(potential.forces[0], force_array, rtol=1e-5, atol=1e-7)
        np.testing.assert_allclose(potential.forces[1], [0, -force, 0], rtol=1e-5, atol=1e-7)
        np.testing.assert_allclose(
            potential.forces[2],
            [-force_array[0], force_array[1], force_array[2]],
            rtol=1e-5,
            atol=1e-7,
        )


def test_before_attaching():
    from hoomd.heat_transfer.angle import Cosine

    potential = Cosine()
    potential.params["A-A-A"] = dict(k=3.0)

    assert potential.params["A-A-A"]["k"] == pytest.approx(3.0)


def test_after_attaching(triplet_snapshot_factory):
    from hoomd.heat_transfer.angle import Cosine

    sim = _make_simulation(triplet_snapshot_factory())

    potential = Cosine()
    potential.params["A-A-A"] = dict(k=3.0)
    sim.operations.integrator = hoomd.md.Integrator(dt=0.005, forces=[potential])

    sim.run(0)

    assert potential.params["A-A-A"]["k"] == pytest.approx(3.0)


def test_forces_and_energies(triplet_snapshot_factory):
    _assert_forces_and_energies(triplet_snapshot_factory, hoomd.device.CPU())


def test_forces_and_energies_gpu(triplet_snapshot_factory):
    _assert_forces_and_energies(triplet_snapshot_factory, _make_gpu_device())
