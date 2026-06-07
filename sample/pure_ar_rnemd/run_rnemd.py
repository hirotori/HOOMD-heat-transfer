"""Run a small pure-Ar RNEMD thermal-conductivity sample.

The run writes:

- ``rnemd_state.gsd``: final trajectory frames for inspection.
- ``temperature_profile.dat``: averaged slab temperatures.
- ``rnemd_summary.dat``: heat flux, gradient, and thermal conductivity estimate.

For a quick smoke test, reduce ``N_RNEMD_STEPS`` to 20_000. The default is still
small enough for a MacBook CPU, but long enough to show a visible gradient.
"""

from pathlib import Path

import hoomd
import numpy as np
from hoomd.custom import Action
from hoomd.heat_transfer import heatflow

DT = 0.002
KT = 0.71
RHO = 0.84
N_SLABS = 20
NVT_STEPS = 20_000
NVE_RELAX_STEPS = 10_000
N_RNEMD_STEPS = 100_000
PROFILE_PERIOD = 100
DUMP_PERIOD = 10_000

# Target slope of integrated exchanged energy per area per timestep.
# The resulting imposed heat flux is TARGET_SLOPE / (2 * DT).
TARGET_SLOPE = 2.0e-5
HEAT_EPSILON = 5.0e-3


class SlabTemperatureProfile(Action):
    """Accumulate kinetic-temperature and density profiles along z."""

    def __init__(self, n_slabs, start_timestep=0):
        super().__init__()
        self.n_slabs = n_slabs
        self.start_timestep = start_timestep
        self.temperature_sum = np.zeros(n_slabs)
        self.count_sum = np.zeros(n_slabs)
        self.samples = 0

    def act(self, timestep):
        if timestep < self.start_timestep:
            return

        snap = self._state.get_snapshot()
        if snap.communicator.rank != 0:
            return

        box = snap.configuration.box
        Lz = box[2]
        z = snap.particles.position[:, 2]
        v = snap.particles.velocity
        m = snap.particles.mass

        slab = ((z / Lz + 0.5) * self.n_slabs).astype(np.int64) % self.n_slabs
        ke = 0.5 * m * np.sum(v * v, axis=1)

        count = np.bincount(slab, minlength=self.n_slabs).astype(float)
        ke_sum = np.bincount(slab, weights=ke, minlength=self.n_slabs)

        temperature = np.zeros(self.n_slabs)
        valid = count > 0
        temperature[valid] = 2.0 * ke_sum[valid] / (3.0 * count[valid])

        self.temperature_sum += temperature
        self.count_sum += count
        self.samples += 1

    @property
    def average_temperature(self):
        if self.samples == 0:
            return np.zeros(self.n_slabs)
        return self.temperature_sum / self.samples

    @property
    def average_count(self):
        if self.samples == 0:
            return np.zeros(self.n_slabs)
        return self.count_sum / self.samples


def fit_temperature_gradient(z_centers, temperature, cold_slab, hot_slab):
    """Fit both linear branches and return the mean dT/dz magnitude."""
    n_slabs = len(temperature)
    left = np.arange(cold_slab + 2, hot_slab - 1)
    right = np.arange(hot_slab + 2, n_slabs - 1)

    slope_left = np.polyfit(z_centers[left], temperature[left], 1)[0]
    slope_right = np.polyfit(z_centers[right], temperature[right], 1)[0]

    # Periodicity makes the two branches have opposite signs.
    gradient = 0.5 * (abs(slope_left) + abs(slope_right))
    return gradient, slope_left, slope_right


def main():
    if not Path("init.gsd").exists():
        raise RuntimeError("init.gsd not found. Run `python init_cond.py` first.")

    sim = hoomd.Simulation(device=hoomd.device.CPU(), seed=605)
    sim.create_state_from_gsd("init.gsd")

    integrator = hoomd.md.Integrator(dt=DT)
    nlist = hoomd.md.nlist.Cell(buffer=0.4)
    lj = hoomd.md.pair.LJ(nlist=nlist, default_r_cut=2.5)
    lj.params[("Ar", "Ar")] = dict(epsilon=1.0, sigma=1.0)
    integrator.forces.append(lj)

    nvt = hoomd.md.methods.ConstantVolume(
        filter=hoomd.filter.All(),
        thermostat=hoomd.md.methods.thermostats.MTTK(kT=KT, tau=100 * DT),
    )
    integrator.methods.append(nvt)
    sim.operations.integrator = integrator
    sim.state.thermalize_particle_momenta(hoomd.filter.All(), kT=KT)

    print(f"NVT equilibration: {NVT_STEPS} steps")
    sim.run(NVT_STEPS)

    integrator.methods.clear()
    integrator.methods.append(hoomd.md.methods.ConstantVolume(hoomd.filter.All()))

    print(f"NVE relaxation: {NVE_RELAX_STEPS} steps")
    sim.run(NVE_RELAX_STEPS)

    ramp_start = sim.timestep
    target = hoomd.variant.Ramp(0.0, TARGET_SLOPE * N_RNEMD_STEPS,
                                ramp_start, N_RNEMD_STEPS)
    rnemd = heatflow.ReversePerturbationHeatFlow(
        filter=hoomd.filter.All(),
        heat_flux_target=target,
        slab_direction="z",
        n_slabs=N_SLABS,
        cold_slab=0,
        hot_slab=N_SLABS // 2,
        heat_flux_epsilon=HEAT_EPSILON,
    )
    sim.operations.add(rnemd)

    profile = SlabTemperatureProfile(N_SLABS, start_timestep=ramp_start + N_RNEMD_STEPS // 2)
    sim.operations.writers.append(
        hoomd.write.CustomWriter(action=profile, trigger=hoomd.trigger.Periodic(PROFILE_PERIOD)))

    sim.operations.writers.append(
        hoomd.write.GSD(filename="rnemd_state.gsd",
                        trigger=hoomd.trigger.Periodic(DUMP_PERIOD),
                        mode="wb"))

    print(f"RNEMD production: {N_RNEMD_STEPS} steps")
    sim.run(N_RNEMD_STEPS)

    snap = sim.state.get_snapshot()
    if snap.communicator.rank != 0:
        return

    box = snap.configuration.box
    area = box[0] * box[1]
    elapsed_time = N_RNEMD_STEPS * DT
    exchanged_energy = rnemd.summed_exchanged_energy
    imposed_flux = exchanged_energy / (2.0 * area * elapsed_time)

    dz = box[2] / N_SLABS
    z_centers = (np.arange(N_SLABS) + 0.5) * dz - 0.5 * box[2]
    temperature = profile.average_temperature
    count = profile.average_count
    gradient, slope_left, slope_right = fit_temperature_gradient(
        z_centers, temperature, 0, N_SLABS // 2)
    conductivity = imposed_flux / gradient

    np.savetxt(
        "temperature_profile.dat",
        np.column_stack([np.arange(N_SLABS), z_centers, temperature, count]),
        header="slab z_center T average_particle_count",
    )

    with open("rnemd_summary.dat", "w", encoding="utf-8") as f:
        f.write("# Pure Ar RNEMD sample in Lennard-Jones reduced units\n")
        f.write(f"N {snap.particles.N}\n")
        f.write(f"rho {RHO}\n")
        f.write(f"kT_initial {KT}\n")
        f.write(f"dt {DT}\n")
        f.write(f"n_rnemd_steps {N_RNEMD_STEPS}\n")
        f.write(f"n_profile_samples {profile.samples}\n")
        f.write(f"area {area:.12g}\n")
        f.write(f"elapsed_time {elapsed_time:.12g}\n")
        f.write(f"summed_exchanged_energy {exchanged_energy:.12g}\n")
        f.write(f"imposed_heat_flux {imposed_flux:.12g}\n")
        f.write(f"temperature_gradient_abs {gradient:.12g}\n")
        f.write(f"temperature_slope_left {slope_left:.12g}\n")
        f.write(f"temperature_slope_right {slope_right:.12g}\n")
        f.write(f"thermal_conductivity {conductivity:.12g}\n")

    print("Wrote temperature_profile.dat and rnemd_summary.dat")
    print(f"imposed heat flux       = {imposed_flux:.6g}")
    print(f"|dT/dz|                 = {gradient:.6g}")
    print(f"thermal conductivity k  = {conductivity:.6g}")


if __name__ == "__main__":
    main()
