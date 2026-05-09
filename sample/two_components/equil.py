import hoomd
from hoomd.heat_transfer import heatflux, correlate
import hflux

kB = 1.380649e-23             # [J/K]
kCal2J = 4184.0/6.02214076e23 # [J]
A2m = 1e-10                   # [m]
fs2s = 1e-15                  # [s]

###------------------------###
eps_Ar_real = 0.2381 #[kcal/mol]
sig_Ar_real = 0.3405 #[nm]
# time stepping size [τ]
dt = 0.002
# temperature [ε/kB]
T_real = 85 #[K]
kT = T_real/(eps_Ar_real*kCal2J/kB)   #85[K] --> 0.71[ε/kB]

# 
nlog = 1_000
ndump = 1_000
nrun = 200_000
###------------------------###

# initialize simulation
device = hoomd.device.GPU()
simulation = hoomd.Simulation(device=device, seed=1000)
simulation.create_state_from_gsd("equil.gsd")

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
integrator.methods.pop(0)
nve = hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All())
integrator.methods.append(nve)
simulation.operations.integrator = integrator

# thermodynamical properties to be computed
simulation.always_compute_pressure = True
thermo = hoomd.md.compute.ThermodynamicQuantities(hoomd.filter.All())
simulation.operations.computes.append(thermo)

# compute partial enthalpy
enthalpy = hflux.ComputeEnthalpyFlux()
h_writer = hoomd.write.CustomWriter(trigger=hoomd.trigger.Periodic(ndump), action=enthalpy)
simulation.operations.writers.append(h_writer)

# compute heat flux
heat_flux = heatflux.ComputeheatFlux(hoomd.filter.All())
simulation.operations.computes.append(heat_flux)

logger = hoomd.logging.Logger(categories=["sequence", "scalar"])
logger.add(heat_flux, quantities=["heatflux"])
logger.add(enthalpy, quantities=["enthalpy_flux"])

# dump
dump_writer = hoomd.write.GSD(
    trigger=hoomd.trigger.Periodic(period=ndump),
    filename="equil.gsd",
    mode="wb",
    logger=logger
)
simulation.operations.writers.append(dump_writer)

# run
simulation.run(nrun)

