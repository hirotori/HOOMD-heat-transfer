from kitmolsim import init
from kitmolsim.writer import write_lammps
import numpy as np
from gsd import hoomd

# rho = N/L^3: L = (N/rho)^(1/3)
N = 512
rho = 0.84
kT = 0.71
L = np.cbrt(float(N)/rho); print(f"Box size = {L:.4f}")
rng = np.random.Generator(np.random.MT19937(seed=332))
pos, box = init.make_defected_fcc(L, rho, seed=440)
vel = rng.normal(loc=0.0, scale=np.sqrt(kT), size=(N,3))

typeid = np.append(np.zeros(N//2), np.ones(N//2))
rng.shuffle(typeid)
mass = np.full(shape=N, fill_value=1.0)
mass[typeid == 1] = 0.25

write_lammps.write_lmp_data("pos_sol.dat", 
                            Lbox=box,
                            Ntotal=N,
                            Natyp=1,
                            pos=pos,
                            mole_id=np.ones(N),
                            atype=typeid+1,
                            charges=np.zeros(N),
                            velocity=vel)


frame = hoomd.Frame()
file = hoomd.open(name="pos_sol.gsd", mode="w")

frame.configuration.box = np.append(box, [0.,0.,0.])
frame.configuration.step = 0
frame.particles.N = N
frame.particles.types = ["A", "B"]
frame.particles.mass = mass
frame.particles.position = pos
frame.particles.velocity = vel
frame.particles.typeid = typeid
file.append(frame=frame)
file.close()

