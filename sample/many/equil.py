import hoomd
from hoomd.heat_transfer import heatflux, correlate
import numpy as np

###------------------------###
# time stepping size [τ]
dt = 0.002
# temperature [ε/kB]
kT = 0.71
# 
nlog = 1_000
ndump = 1_000
nrun = 200_000
###------------------------###

# initialize simulation
cpu = hoomd.device.CPU()
simulation = hoomd.Simulation(device=cpu, seed=1000)
simulation.create_state_from_gsd("pos_sol.gsd")

# integrator
integrator = hoomd.md.Integrator(dt=dt)

# pair-wise interaction
cell = hoomd.md.nlist.Cell(buffer=0.4)
## epsilon
eps_ArAr = 1.0
## sigma
sig_ArAr = 1.0
## coefficient (for surface anisotropy)
lj = hoomd.md.pair.LJ(nlist=cell, default_r_cut=2.5)
lj.params[("A","A")] = dict(epsilon=eps_ArAr, sigma=sig_ArAr)
integrator.forces.append(lj)

# ==== NVT ====
nvt = hoomd.md.methods.ConstantVolume(
    filter=hoomd.filter.All(), 
    thermostat=hoomd.md.methods.thermostats.MTTK(kT=kT, tau=100.0*dt)
)

integrator.methods.append(nvt)
simulation.operations.integrator = integrator

# initialize velocity
simulation.state.thermalize_particle_momenta(filter=hoomd.filter.All(), kT=kT)

simulation.run(100_000)

# ==== switch to NVE ====
integrator.methods.pop(0)
nve = hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All())
integrator.methods.append(nve)


# thermodynamical properties to be computed
simulation.always_compute_pressure = True
thermo = hoomd.md.compute.ThermodynamicQuantities(hoomd.filter.All())
simulation.operations.computes.append(thermo)

# Add custom action
heat_flux = heatflux.ComputeheatFlux(hoomd.filter.All())
simulation.operations.computes.append(heat_flux)

# # logging
logger = hoomd.logging.Logger(categories=["sequence", "scalar"])
logger.add(heat_flux, quantities=["heatflux"])
#logger.add(lj, quantities=["energy"])

s = 1
p = 1000
d = s*p
corr = correlate.Correlator(logger,
                            sample_interval=s,
                            output_interval=d,
                            max_lag = p)
simulation.operations.writers.append(corr)

#stream_writer = hoomd.write.Table(
#     trigger=hoomd.trigger.Periodic(period=nlog),
#     logger=logger,
#     output=open("log-output.txt", mode="w"),
#     pretty=True,
#     max_header_len = 10
#)
#simulation.operations.writers.append(stream_writer)

#import stream
#seq_writer = stream.TextSequenceWriter(
#      filename="heat_flux.dat",
#      logger=logger,
#)
#stream_writer = hoomd.write.CustomWriter(action=seq_writer, trigger=hoomd.trigger.Periodic(period=nlog))
#simulation.operations.writers.append(stream_writer)

# # dump
#dump_writer = hoomd.write.GSD(
#     trigger=hoomd.trigger.Periodic(period=ndump),
#     filename="equil.gsd",
#     mode="wb",
#)
#simulation.operations.writers.append(dump_writer)

# run
simulation.run(nrun)
