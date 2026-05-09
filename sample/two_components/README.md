# Binary Lennard-Jones sample

[English](#english) / [日本語](#日本語)

## English

This sample contains a binary Lennard-Jones mixture with 512 particles. Half of the particles are type `A`, half are type `B`, and type `B` has a smaller mass.

Use this case to:

- check heat-flux calculations in a mixture;
- inspect the partial-enthalpy contribution used with `include_enthalpy=True`;
- compare the compiled `ComputeheatFlux` implementation with the pure-Python reference action in `hflux.py`.

Files:

- `init_cond.py`: generates `pos_sol.gsd` and `pos_sol.dat` with randomized particle types.
- `equil.py`: runs the binary-mixture simulation and logs heat-flux data.
- `hflux.py`: pure-Python reference implementation for partial enthalpy flux.
- `n512.gsd`, `pos_sol.gsd`, `equil.gsd`: prepared HOOMD states/trajectories.
- `pos_sol.dat`: LAMMPS-style data file.

## 日本語

512粒子の二成分 Lennard-Jones 混合系サンプルです。粒子の半分が type `A`、残り半分が type `B` で、type `B` には小さい質量を与えています。

このケースは以下の確認に使います。

- 混合系での熱流束計算を確認する
- `include_enthalpy=True` で使う部分エンタルピー寄与を確認する
- コンパイル済みの `ComputeheatFlux` 実装と、`hflux.py` の pure-Python 参照実装を比較する

ファイル:

- `init_cond.py`: 粒子タイプをランダムに割り当てた `pos_sol.gsd` と `pos_sol.dat` を生成します。
- `equil.py`: 二成分混合系を走らせ、熱流束データをログ出力します。
- `hflux.py`: 部分エンタルピー流束の pure-Python 参照実装です。
- `n512.gsd`, `pos_sol.gsd`, `equil.gsd`: 準備済みの HOOMD 状態・軌跡です。
- `pos_sol.dat`: LAMMPS 形式の data ファイルです。
