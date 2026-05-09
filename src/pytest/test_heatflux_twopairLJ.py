import hoomd
import hoomd.md
from hoomd.heat_transfer import heatflux
import numpy as np
import pytest


def setup_two_particle_sim(r=1.5, v=0.0):
    device = hoomd.device.CPU()
    sim = hoomd.Simulation(device=device, seed=42)

    snap = hoomd.Snapshot()
    snap.particles.N = 2
    snap.particles.types = ['A']

    snap.configuration.box = [10,10,10,0,0,0]

    snap.particles.mass[:] = 1.0
    snap.particles.position[0] = [+r/2, 0, 0]
    snap.particles.position[1] = [-r/2, 0, 0]

    snap.particles.velocity[0] = [v, 0, 0]
    snap.particles.velocity[1] = [-v, 0, 0]

    sim.create_state_from_snapshot(snap)

    nl = hoomd.md.nlist.Cell(buffer=0.4)
    lj = hoomd.md.pair.LJ(nlist=nl)
    lj.params[('A','A')] = dict(epsilon=1.0, sigma=1.0)
    lj.r_cut[('A','A')] = 3.0

    integrator = hoomd.md.Integrator(dt=0.0)
    integrator.forces.append(lj)

    sim.operations.integrator = integrator

    sim.always_compute_pressure = True
    return sim


def test_symmetric_velocity_zero_heatflux():

    sim = setup_two_particle_sim(r=1.5, v=1.0)

    compute = heatflux.ComputeheatFlux(hoomd.filter.All())
    sim.operations.computes.append(compute)

    sim.run(0)

    J = np.array(compute.heatflux)

    assert np.allclose(J, np.zeros(3), atol=1e-10)

def lj_force(r):
    epsilon = 1.0
    sigma = 1.0
    return 24*epsilon*(2*(sigma/r)**12 - (sigma/r)**6)/r


def test_asymmetric_velocity():

    r = 1.5
    v = 1.0
    m = 1.0

    sim = setup_two_particle_sim(r=r, v=0.0)

    # 粒子1のみ速度
    snap = sim.state.get_snapshot()
    snap.particles.velocity[0] = [v,0,0]
    snap.particles.velocity[1] = [0,0,0]
    sim.state.set_snapshot(snap)

    compute = heatflux.ComputeheatFlux(hoomd.filter.All())
    sim.operations.computes.append(compute)

    sim.run(0)

    J = np.array(compute.heatflux)

    U = 4*((1/r)**12 - (1/r)**6)
    ke = 0.5*m*v*v
    F = lj_force(r)

    expected = (ke + 0.5*U)*v + 0.5*r*F*v

    # print(" ** kinetic/virial heatflux ** ")
    # print(f"kinetic: {compute.kinetic_heatflux[0]} | {(ke + 0.5*U)*v}")
    # print(f"virial : {compute.virial_heatflux[0]} | {0.5*r*F*v}")

    assert np.allclose(J[0], expected, atol=1e-8)