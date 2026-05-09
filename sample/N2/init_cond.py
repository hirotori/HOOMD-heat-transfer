from kitmolsim import init
from kitmolsim.writer import write_lammps
import numpy as np
from gsd import hoomd

# rho = N/L^3: L = (N/rho)^(1/3)
N = 2
L = 8.0
r0 = 2**(1./6.)
pos = np.array([[-r0/2, 0.0, 0.0],[r0/2, 0.0, 0.0]])
vel = np.array([[ 5.0, 0.0, 0.0],[0.0, 0.0, 0.0]])
box = np.full(3, fill_value=L)


write_lammps.write_lmp_data("pos_sol.dat", 
                            Lbox=box,
                            Ntotal=N,
                            Natyp=1,
                            pos=pos,
                            velocity=vel,
                            mole_id=np.ones(N),
                            atype=np.ones(N),
                            charges=np.zeros(N))


frame = hoomd.Frame()
file = hoomd.open(name="pos_sol.gsd", mode="w")

frame.configuration.box = np.append(box, [0.,0.,0.])
frame.configuration.step = 0
frame.particles.N = N
frame.particles.types = ["A"]
frame.particles.mass = np.full(shape=N, fill_value=1.0)
frame.particles.position = pos
frame.particles.velocity = vel
frame.particles.typeid = np.zeros(N)
file.append(frame=frame)
file.close()
