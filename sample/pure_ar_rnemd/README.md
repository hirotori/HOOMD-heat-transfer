# Pure Ar RNEMD thermal conductivity sample

[English](#english) / [日本語](#日本語)

## English

This sample validates `heat_transfer.heatflow.ReversePerturbationHeatFlow` on a
small one-component Lennard-Jones argon-like liquid. It imposes a heat flux with
the Muller-Plathe reverse perturbation method, measures the slab temperature
profile, and estimates the thermal conductivity.

The system is intentionally modest for a MacBook CPU:

- 768 particles on a `4 x 4 x 12` FCC lattice
- reduced density `rho = 0.84`
- reduced temperature `kT = 0.71`
- 20 slabs along `z`
- 100,000 RNEMD production steps by default

Run:

```bash
python init_cond.py
python run_rnemd.py
python analyze.py
```

Outputs:

- `init.gsd`: initial elongated LJ state
- `rnemd_state.gsd`: sparse trajectory during RNEMD
- `temperature_profile.dat`: columns are `slab`, `z_center`, `T`, and average particle count
- `rnemd_summary.dat`: imposed heat flux, fitted gradient, and conductivity

The reported conductivity is in Lennard-Jones reduced units. The run is meant as
an implementation and workflow check, not as a publication-quality estimate. For
better statistics, increase `N_RNEMD_STEPS`, run independent seeds, and fit only
the linear regions away from the hot and cold slabs.

## 日本語

このサンプルは、`heat_transfer.heatflow.ReversePerturbationHeatFlow` を pure Ar
相当の一成分 Lennard-Jones 液体で検証するためのものです。Muller-Plathe の
reverse perturbation 法で熱流束を与え、slabごとの温度プロファイルを測り、熱伝導率を見積もります。

MacBookのCPUでも回せるように、系は小さめにしています。

- `4 x 4 x 12` FCC格子、768粒子
- reduced density `rho = 0.84`
- reduced temperature `kT = 0.71`
- `z`方向に20 slabs
- デフォルトでRNEMD production 100,000 steps

実行方法:

```bash
python init_cond.py
python run_rnemd.py
python analyze.py
```

出力:

- `init.gsd`: 初期構造
- `rnemd_state.gsd`: RNEMD中の疎なtrajectory
- `temperature_profile.dat`: `slab`, `z_center`, `T`, 平均粒子数
- `rnemd_summary.dat`: 与えた熱流束、温度勾配、熱伝導率

熱伝導率はLennard-Jones reduced unitsです。このサンプルは実装とワークフロー確認用で、出版品質の推定値ではありません。精度を上げるには、`N_RNEMD_STEPS`を増やし、seed違いの独立runを取り、hot/cold slab近傍を避けて線形領域をfitしてください。
