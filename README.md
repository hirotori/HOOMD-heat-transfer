# HOOMD-blue heat transfer
[![DOI](https://zenodo.org/badge/1233641288.svg)](https://doi.org/10.5281/zenodo.21255016)

[English](#english) / [日本語](#日本語)

## English

`hoomd-heat-transfer` is a HOOMD-blue component for studying thermal conductivity and heat flow with molecular dynamics simulations. It provides microscopic heat-flux calculations for equilibrium molecular dynamics, time-correlation calculations based on the Green-Kubo formula, and nonequilibrium simulations that generate a temperature gradient for estimating thermal conductivity.

The package provides:

- `heat_transfer.heatflux.ComputeheatFlux`: logs total, kinetic, virial, and optional partial-enthalpy heat flux.
- `heat_transfer.heatflow.ReversePerturbationHeatFlow`: a HOOMD `Updater` that imposes heat flow with the Muller-Plathe reverse perturbation method.
- `heat_transfer.correlate.Correlator`: a HOOMD `Writer` that samples logged numeric quantities and writes autocorrelation data.
- `heat_transfer.tablewriter.SequenceTable`: a HOOMD `Writer` that writes scalar and one-dimensional sequence logger quantities as table columns.
- CPU and GPU implementations of the heat-flux compute.
- Example systems for Green-Kubo and RNEMD thermal-transport workflows.

### Features

`ComputeheatFlux` reports the sequence quantities:

- `heatflux`: total heat flux, `kinetic_heatflux + virial_heatflux - enthalpy_flux`
- `kinetic_heatflux`: convective energy flux
- `virial_heatflux`: virial contribution
- `enthalpy_flux`: partial-enthalpy contribution for multicomponent systems, enabled with `include_enthalpy=True`

On GPU devices, the component uses `ComputeHeatFluxGPU` when the extension is built with GPU support. When `include_enthalpy=True`, the partial-enthalpy term follows the same definition as the CPU implementation.

`ReversePerturbationHeatFlow` divides the box into slabs along `x`, `y`, or `z`, then exchanges velocities between selected particles in a cold slab and a hot slab. It logs:

- `summed_exchanged_energy`: total exchanged kinetic energy accumulated by the updater

For strict energy and momentum conservation in RNEMD runs, use a filter containing particles of a single mass. GPU is not supported yet.

`Correlator` and `SequenceTable` both accept numeric scalar quantities and one-dimensional numeric sequence quantities from a HOOMD `Logger`. Sequence values are expanded into separate scalar channels or columns.

### Requirements

- HOOMD-blue 4.0 or newer. This project has been tested with HOOMD-blue 4.8.0.
- CMake 3.9 or newer.
- A Python environment that can import the same HOOMD-blue installation used by CMake.
- For GPU builds, a GPU-enabled HOOMD-blue build and the corresponding HIP/CUDA toolchain.

### Build

Activate the Python environment that contains HOOMD-blue, then configure, build, and install the component.

```bash
source /path/to/hoomd-venv/bin/activate
cmake -B build/hoomd-heat-transfer -S . \
  -DPython_EXECUTABLE=/path/to/hoomd-venv/bin/python
cmake --build build/hoomd-heat-transfer
cmake --install build/hoomd-heat-transfer
```

If CMake cannot find HOOMD-blue or its dependencies automatically, pass the package directories explicitly:

```bash
cmake -B build/hoomd-heat-transfer -S . \
  -DPython_EXECUTABLE=/path/to/hoomd-venv/bin/python \
  -DHOOMD_DIR=/path/to/hoomd-venv/lib/cmake/hoomd \
  -Dpybind11_DIR=/path/to/hoomd-venv/share/cmake/pybind11 \
  -DEigen3_DIR=/path/to/hoomd-venv/share/eigen3/cmake
```

### Usage

#### Green-Kubo heat-flux autocorrelation

```python
import hoomd
from hoomd.heat_transfer import heatflux, correlate

device = hoomd.device.GPU()
simulation = hoomd.Simulation(device=device, seed=1000)

# Set up the state, forces, and integrator before adding the compute.
simulation.always_compute_pressure = True

heat_flux = heatflux.ComputeheatFlux(
    filter=hoomd.filter.All(),
    include_enthalpy=True,
)
simulation.operations.computes.append(heat_flux)

logger = hoomd.logging.Logger(categories=["sequence"])
logger.add(heat_flux, quantities=["heatflux", "kinetic_heatflux", "virial_heatflux"])

correlator = correlate.Correlator(
    logger=logger,
    sample_interval=1,
    output_interval=1000,
    max_lag=1000,
)
simulation.operations.writers.append(correlator)
```

When using parameters named `s`, `p`, and `d`, pass them as
`sample_interval=s`, `output_interval=d`, and `max_lag=p`.

The correlator writes files named `correlation_<timestep>.dat` in the current working directory.

**Note** Long heat-flux autocorrelation calculations on CPU may cause an OS restart. Running on GPU is recommended. Alternatively, write the heat-flux time series to a file as shown below, then compute the autocorrelation after all simulations have finished.

To write a heat-flux time series instead of, or in addition to, autocorrelation data:

```python
from hoomd.heat_transfer import tablewriter

with open("heatflux_timeseries.dat", "w") as output:
    writer = tablewriter.SequenceTable(
        logger=logger,
        trigger=hoomd.trigger.Periodic(10),
        output=output,
    )
    simulation.operations.writers.append(writer)
    simulation.run(10000)
```

#### Muller-Plathe RNEMD heat flow

```python
import hoomd
from hoomd.heat_transfer import heatflow

target = hoomd.variant.Ramp(0.0, 1.0, 0, 100000)
rnemd = heatflow.ReversePerturbationHeatFlow(
    filter=hoomd.filter.All(),
    heat_flux_target=target,
    slab_direction="z",
    n_slabs=20,
    cold_slab=0,
    hot_slab=10,
)
simulation.operations.add(rnemd)
```

The imposed heat flux can be estimated from `rnemd.summed_exchanged_energy` divided by twice the slab area and the elapsed simulation time.

### Samples

See [sample/README.md](sample/README.md) for the purpose of each example case:

- `sample/N2`: a two-particle sanity-check system.
- `sample/many`: a one-component Lennard-Jones fluid for heat-flux autocorrelation.
- `sample/pure_ar_rnemd`: a pure-Ar Lennard-Jones RNEMD case for imposing a heat flux and estimating thermal conductivity.
- `sample/two_components`: a binary Lennard-Jones mixture with a partial-enthalpy flux reference.

### Tests

After installing the component into the HOOMD-blue environment:

```bash
python -m pytest -q /path/to/hoomd/site-packages/hoomd/heat_transfer/pytest
```

The test suite covers basic import/version checks, zero and uniform-velocity heat-flux cases, a two-particle Lennard-Jones check, the RNEMD heat-flow updater, and the correlator accumulation logic.

### Status

This repository is under active development. The public API and sample files may change while the heat-flux and correlation workflows are refined.

## 日本語

`hoomd-heat-transfer` は、分子動力学シミュレーションを用いて熱伝導率や熱の流れを研究するためのHOOMD-blue コンポーネントです。平衡分子動力学を用いた微視的熱流束の計算、Green-Kubo 公式に基づいた時間相関関数の計算のほか、温度勾配を生成して熱伝導率を算出する非平衡シミュレーションを提供します。 

このパッケージには次の機能が含まれています。
- `heat_transfer.heatflux.ComputeheatFlux`: 全熱流束、運動項、ビリアル項、必要に応じて部分エンタルピー項をログ出力する
- `heat_transfer.heatflow.ReversePerturbationHeatFlow`: Muller-Plathe の reverse perturbation 法で熱流束を与える HOOMD `Updater`
- `heat_transfer.correlate.Correlator`: HOOMD の `Logger` から数値列をサンプリングし、自己相関を出力する `Writer`
- `heat_transfer.tablewriter.SequenceTable`: scalar と一次元 sequence の logger quantity を表形式で出力する `Writer`
- 熱流束計算の CPU 実装と GPU 実装
- Green-Kubo と RNEMD の熱輸送ワークフロー用サンプル

### 機能

`ComputeheatFlux` は以下の sequence quantity を提供します。

- `heatflux`: 全熱流束、`kinetic_heatflux + virial_heatflux - enthalpy_flux`
- `kinetic_heatflux`: エネルギー輸送の対流項
- `virial_heatflux`: ビリアル寄与
- `enthalpy_flux`: 多成分系のための部分エンタルピー流束。`include_enthalpy=True` で有効化

GPU デバイス上では、GPU 対応でビルドされている場合に `ComputeHeatFluxGPU` を使用します。`include_enthalpy=True` の場合も、CPU 実装と同じ定義で部分エンタルピー項を計算します。

`ReversePerturbationHeatFlow` はシミュレーションボックスを `x`、`y`、または `z` 方向の slab に分け、cold slab と hot slab の粒子速度を交換します。以下の scalar quantity をログ出力できます。

- `summed_exchanged_energy`: updater が蓄積した交換済み運動エネルギー

RNEMD でエネルギーと運動量を厳密に保存したい場合は、同じ質量の粒子だけを含む filter を使ってください。GPUはまだ対応していません。

`Correlator` と `SequenceTable` は、HOOMD `Logger` から得られる数値 scalar と一次元の数値 sequence を扱えます。sequence は、相関関数の各チャンネルまたは表の各列に展開されます。

### 必要なもの

- HOOMD-blue 4.0 以降。このプロジェクトでは HOOMD-blue 4.8.0 で動作確認しています。
- CMake 3.9 以降
- CMake が参照する HOOMD-blue と同じものを import できる Python 環境
- GPU ビルドの場合は、GPU 対応の HOOMD-blue と対応する HIP/CUDA ツールチェイン

### ビルド

HOOMD-blue が入っている Python 環境を有効化してから、configure、build、install を行います。

```bash
source /path/to/hoomd-venv/bin/activate
cmake -B build/hoomd-heat-transfer -S . \
  -DPython_EXECUTABLE=/path/to/hoomd-venv/bin/python
cmake --build build/hoomd-heat-transfer
cmake --install build/hoomd-heat-transfer
```

CMake が HOOMD-blue や依存パッケージを自動で見つけられない場合は、以下のようにパスを明示してください。

```bash
cmake -B build/hoomd-heat-transfer -S . \
  -DPython_EXECUTABLE=/path/to/hoomd-venv/bin/python \
  -DHOOMD_DIR=/path/to/hoomd-venv/lib/cmake/hoomd \
  -Dpybind11_DIR=/path/to/hoomd-venv/share/cmake/pybind11 \
  -DEigen3_DIR=/path/to/hoomd-venv/share/eigen3/cmake
```

### 使い方

#### Green-Kubo 熱流束自己相関

```python
import hoomd
from hoomd.heat_transfer import heatflux, correlate

device = hoomd.device.GPU()
simulation = hoomd.Simulation(device=device, seed=1000)

# 状態、力場、積分器を設定した後に compute を追加します。
simulation.always_compute_pressure = True

heat_flux = heatflux.ComputeheatFlux(
    filter=hoomd.filter.All(),
    include_enthalpy=True,
)
simulation.operations.computes.append(heat_flux)

logger = hoomd.logging.Logger(categories=["sequence"])
logger.add(heat_flux, quantities=["heatflux", "kinetic_heatflux", "virial_heatflux"])

correlator = correlate.Correlator(
    logger=logger,
    sample_interval=1,
    output_interval=1000,
    max_lag=1000,
)
simulation.operations.writers.append(correlator)
```

`s`, `p`, `d` という変数名を使う場合は、`sample_interval=s`,
`output_interval=d`, `max_lag=p` として渡してください。

`Correlator` はカレントディレクトリに `correlation_<timestep>.dat` という名前で相関関数を書き出します。

**注意** CPU上で熱流束の自己相関を長時間計算すると、OS再起動が生じる場合があります。GPU上で計算することを推奨します。あるいは以下に示すように、熱流束をファイルに書き出し、シミュレーションが全て終わった後で自己相関を計算する手順も利用可能です。

熱流束の時系列を、相関関数とは別に、または相関関数と同時に書き出す場合は `SequenceTable` を使います。

```python
from hoomd.heat_transfer import tablewriter

with open("heatflux_timeseries.dat", "w") as output:
    writer = tablewriter.SequenceTable(
        logger=logger,
        trigger=hoomd.trigger.Periodic(10),
        output=output,
    )
    simulation.operations.writers.append(writer)
    simulation.run(10000)
```

#### Muller-Plathe RNEMD 熱流

```python
import hoomd
from hoomd.heat_transfer import heatflow

target = hoomd.variant.Ramp(0.0, 1.0, 0, 100000)
rnemd = heatflow.ReversePerturbationHeatFlow(
    filter=hoomd.filter.All(),
    heat_flux_target=target,
    slab_direction="z",
    n_slabs=20,
    cold_slab=0,
    hot_slab=10,
)
simulation.operations.add(rnemd)
```

与えた熱流束は、`rnemd.summed_exchanged_energy` を slab 断面積の2倍と経過シミュレーション時間で割ることで見積もれます。

### サンプル

各サンプルの目的は [sample/README.md](sample/README.md) を参照してください。

- `sample/N2`: 二粒子系による簡単な確認用ケース
- `sample/many`: 一成分 Lennard-Jones 流体の熱流束自己相関計算
- `sample/pure_ar_rnemd`: 熱流束を与えて熱伝導率を見積もる pure Ar Lennard-Jones RNEMD ケース
- `sample/two_components`: 部分エンタルピー流束の参照計算を含む二成分 Lennard-Jones 混合系

### テスト

コンポーネントを HOOMD-blue 環境に install した後、以下を実行します。

```bash
python -m pytest -q /path/to/hoomd/site-packages/hoomd/heat_transfer/pytest
```

テストでは、import と version、ゼロ熱流束、一様速度、二粒子 Lennard-Jones 系、RNEMD heat-flow updater、相関関数の蓄積を確認しています。

### 開発状況

このリポジトリは開発中です。熱流束と相関関数のワークフローを整備している段階のため、公開 API やサンプルファイルは今後変更される可能性があります。
