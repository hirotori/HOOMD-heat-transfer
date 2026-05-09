# N2 sample

[English](#english) / [日本語](#日本語)

## English

This is a two-particle Lennard-Jones sanity-check case. The initial configuration places two particles near the LJ minimum and gives one particle a finite velocity.

Use this case to:

- inspect how kinetic and virial terms contribute in a minimal system;
- compare HOOMD output with the included LAMMPS-style data/trajectory files;
- debug sign conventions and unit conversions before running larger systems.

Files:

- `init_cond.py`: generates `pos_sol.gsd` and `pos_sol.dat`.
- `pos_sol.gsd`: HOOMD initial configuration.
- `pos_sol.dat`: LAMMPS data file for cross-checking.
- `n2.lammpstrj`: small LAMMPS-style trajectory/reference output.
- `equil.gsd`: short prepared trajectory/state.

## 日本語

二粒子 Lennard-Jones 系の確認用サンプルです。初期配置では、2粒子を LJ ポテンシャルの極小付近に置き、一方の粒子に有限の速度を与えています。

このケースは以下の確認に使います。

- 最小構成で、運動項とビリアル項が熱流束にどう寄与するかを見る
- HOOMD の出力を、同梱の LAMMPS 形式データや軌跡と比較する
- 大きな系を走らせる前に、符号規約や単位換算を確認する

ファイル:

- `init_cond.py`: `pos_sol.gsd` と `pos_sol.dat` を生成します。
- `pos_sol.gsd`: HOOMD 用の初期構造です。
- `pos_sol.dat`: 比較用の LAMMPS data ファイルです。
- `n2.lammpstrj`: 小さな LAMMPS 形式の軌跡・参照出力です。
- `equil.gsd`: 短い準備済み状態です。
