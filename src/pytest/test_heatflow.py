import hoomd
import numpy as np
import pytest


def _make_two_particle_simulation():
    sim = hoomd.Simulation(device=hoomd.device.CPU(), seed=42)
    snap = hoomd.Snapshot()
    snap.configuration.box = [10, 10, 10, 0, 0, 0]
    snap.particles.N = 2
    snap.particles.types = ["A"]
    snap.particles.position[:] = [[0, 0, -4.9], [0, 0, 0.1]]
    snap.particles.velocity[:] = [[3, 0, 0], [1, 0, 0]]
    snap.particles.mass[:] = [1, 1]
    sim.create_state_from_snapshot(snap)
    sim.operations.integrator = hoomd.md.Integrator(
        0.001, methods=[hoomd.md.methods.ConstantVolume(hoomd.filter.All())])
    return sim


def test_before_attaching():
    from hoomd.heat_transfer.heatflow import ReversePerturbationHeatFlow

    filt = hoomd.filter.All()
    target = hoomd.variant.Constant(1.0)
    updater = ReversePerturbationHeatFlow(
        filt, target, "Z", 2, heat_flux_epsilon=1e-6)

    assert updater.filter == filt
    assert updater.heat_flux_target == target
    assert updater.slab_direction == "z"
    assert updater.n_slabs == 2
    assert updater.hot_slab == 1
    assert updater.cold_slab == 0
    assert updater.trigger == hoomd.trigger.Periodic(1)
    assert updater.heat_flux_epsilon == 1e-6
    with pytest.raises(hoomd.error.DataAccessError):
        updater.summed_exchanged_energy


def test_velocity_exchange_conserves_total_kinetic_energy():
    from hoomd.heat_transfer.heatflow import ReversePerturbationHeatFlow

    sim = _make_two_particle_simulation()
    updater = ReversePerturbationHeatFlow(
        hoomd.filter.All(), hoomd.variant.Constant(0.04), "z", 2,
        heat_flux_epsilon=1e-6)
    sim.operations.add(updater)

    before = sim.state.get_snapshot()
    ke_before = 0.5 * np.sum(before.particles.mass
                             * np.sum(before.particles.velocity**2, axis=1))

    sim.run(1)

    after = sim.state.get_snapshot()
    ke_after = 0.5 * np.sum(after.particles.mass
                            * np.sum(after.particles.velocity**2, axis=1))

    assert np.allclose(ke_after, ke_before)
    assert np.allclose(after.particles.velocity,
                       [[1, 0, 0], [3, 0, 0]])
    assert updater.summed_exchanged_energy == pytest.approx(4.0)
