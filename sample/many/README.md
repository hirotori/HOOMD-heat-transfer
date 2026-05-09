# One-component Lennard-Jones sample

[English](#english) / [日本語](#日本語)

## English

This sample contains a one-component Lennard-Jones liquid with 512 particles at reduced density `rho = 0.84` and temperature `kT = 0.71`.

Use this case to:

- run a homogeneous-fluid heat-flux calculation;
- accumulate the heat-flux autocorrelation with `heat_transfer.correlate.Correlator`;
- compare the HOOMD workflow with the included LAMMPS input.

Files:

- `init_cond.py`: generates the FCC-like initial state, `pos_sol.gsd`, and `pos_sol.dat`.
- `equil.py`: runs HOOMD equilibration, switches from NVT to NVE, computes heat flux, and writes correlation files.
- `equil.in`: LAMMPS input for comparison with `compute heat/flux`.
- `pos_sol.gsd`, `pos_sol.dat`: generated initial configurations.
- `equil.gsd`: prepared HOOMD state/trajectory.

## 日本語

512粒子、還元密度 `rho = 0.84`、温度 `kT = 0.71` の一成分 Lennard-Jones 液体サンプルです。

このケースは以下の確認に使います。

- 均一流体で熱流束を計算する
- `heat_transfer.correlate.Correlator` で熱流束自己相関を蓄積する
- 同梱の LAMMPS 入力と HOOMD の計算手順を比較する

ファイル:

- `init_cond.py`: FCC に近い初期構造、`pos_sol.gsd`、`pos_sol.dat` を生成します。
- `equil.py`: HOOMD で NVT 平衡化後に NVE へ切り替え、熱流束と相関関数を計算します。
- `equil.in`: LAMMPS の `compute heat/flux` と比較するための入力ファイルです。
- `pos_sol.gsd`, `pos_sol.dat`: 生成済み初期構造です。
- `equil.gsd`: 準備済みの HOOMD 状態・軌跡です。
