"""Recompute the thermal-conductivity estimate from RNEMD output files."""

import numpy as np

N_SLABS = 20
HOT_SLAB = N_SLABS // 2
COLD_SLAB = 0


def read_summary(path="rnemd_summary.dat"):
    values = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            key, value = line.split()[:2]
            values[key] = float(value)
    return values


def main():
    profile = np.loadtxt("temperature_profile.dat")
    summary = read_summary()

    z = profile[:, 1]
    temperature = profile[:, 2]
    left = np.arange(COLD_SLAB + 2, HOT_SLAB - 1)
    right = np.arange(HOT_SLAB + 2, N_SLABS - 1)

    slope_left = np.polyfit(z[left], temperature[left], 1)[0]
    slope_right = np.polyfit(z[right], temperature[right], 1)[0]
    gradient = 0.5 * (abs(slope_left) + abs(slope_right))

    flux = summary["summed_exchanged_energy"] / (2.0 * summary["area"] * summary["elapsed_time"])
    conductivity = flux / gradient

    print("Pure Ar RNEMD analysis, LJ reduced units")
    print(f"heat flux              {flux:.8g}")
    print(f"left branch dT/dz      {slope_left:.8g}")
    print(f"right branch dT/dz     {slope_right:.8g}")
    print(f"mean |dT/dz|           {gradient:.8g}")
    print(f"thermal conductivity   {conductivity:.8g}")


if __name__ == "__main__":
    main()
