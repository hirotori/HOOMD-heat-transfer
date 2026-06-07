"""Generate a small elongated pure-Ar Lennard-Jones state for RNEMD.

Units are Lennard-Jones reduced units with argon-like parameters:

- mass m = 1
- sigma = 1
- epsilon = 1
- kB = 1

The system uses a 4 x 4 x 12 FCC lattice, giving 768 particles at rho = 0.84.
This is intentionally small enough for a laptop CPU while still elongated along
z for a readable temperature profile.
"""

import numpy as np
import gsd.hoomd

RHO = 0.84
KT = 0.71
NX, NY, NZ = 4, 4, 12
SEED = 20260605


def make_fcc(nx, ny, nz, rho):
    basis = np.array(
        [[0.0, 0.0, 0.0],
         [0.5, 0.5, 0.0],
         [0.5, 0.0, 0.5],
         [0.0, 0.5, 0.5]],
        dtype=float,
    )
    lattice_constant = (4.0 / rho) ** (1.0 / 3.0)
    box = np.array([nx, ny, nz], dtype=float) * lattice_constant
    positions = []
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                cell = np.array([ix, iy, iz], dtype=float)
                for b in basis:
                    positions.append((cell + b) * lattice_constant - 0.5 * box)
    return np.asarray(positions), box


rng = np.random.default_rng(SEED)
position, box = make_fcc(NX, NY, NZ, RHO)
N = len(position)
velocity = rng.normal(0.0, np.sqrt(KT), size=(N, 3))
velocity -= velocity.mean(axis=0)

frame = gsd.hoomd.Frame()
frame.configuration.box = [box[0], box[1], box[2], 0.0, 0.0, 0.0]
frame.configuration.step = 0
frame.particles.N = N
frame.particles.types = ["Ar"]
frame.particles.typeid = np.zeros(N, dtype=np.uint32)
frame.particles.mass = np.ones(N)
frame.particles.position = position
frame.particles.velocity = velocity

with gsd.hoomd.open("init.gsd", "w") as traj:
    traj.append(frame)

print(f"Wrote init.gsd: N={N}, rho={RHO}, box=({box[0]:.3f}, {box[1]:.3f}, {box[2]:.3f})")
