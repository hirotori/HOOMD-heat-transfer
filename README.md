# HOOMD-blue heat transfer

[English](#english) / [日本語](#日本語)

## English

`hoomd-heat-transfer` is a HOOMD-blue component for computing microscopic heat flux in molecular dynamics simulations and accumulating time-correlation functions for Green-Kubo style analysis.

The package provides:

- `heat_transfer.heatflux.ComputeheatFlux`: a HOOMD `Compute` that logs total, kinetic, virial, and optional partial-enthalpy heat flux.
- `heat_transfer.correlate.Correlator`: a HOOMD `Writer` that samples logged numeric quantities and writes autocorrelation data.
- CPU and GPU implementations of the heat-flux compute.
- Example systems for simple Lennard-Jones fluids and binary mixtures.

### Features

`ComputeheatFlux` reports the sequence quantities:

- `heatflux`: total heat flux, `kinetic_heatflux + virial_heatflux - enthalpy_flux`
- `kinetic_heatflux`: convective energy flux
- `virial_heatflux`: virial contribution
- `enthalpy_flux`: partial-enthalpy contribution, enabled with `include_enthalpy=True`

On GPU devices, the component uses `ComputeHeatFluxGPU` when the extension is built with GPU support. When `include_enthalpy=True`, the partial-enthalpy term follows the same definition as the CPU implementation.

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
    trigger=hoomd.trigger.Periodic(1),
    output_interval=1000,
    max_lag=1000,
)
simulation.operations.writers.append(correlator)
```

The correlator writes files named `correlation_<timestep>.dat` in the current working directory.

### Samples

See [sample/README.md](sample/README.md) for the purpose of each example case:

- `sample/N2`: a two-particle sanity-check system.
- `sample/many`: a one-component Lennard-Jones fluid for heat-flux autocorrelation.
- `sample/two_components`: a binary Lennard-Jones mixture with a partial-enthalpy flux reference.

### Tests

After installing the component into the HOOMD-blue environment:

```bash
python -m pytest -q /path/to/hoomd/site-packages/hoomd/heat_transfer/pytest
```

The test suite covers basic import/version checks, zero and uniform-velocity heat-flux cases, a two-particle Lennard-Jones check, and the correlator accumulation logic.

### Status

This repository is under active development. The public API and sample files may change while the heat-flux and correlation workflows are refined.

## 日本語

`hoomd-heat-transfer` は、分子動力学シミュレーションにおける微視的熱流束の計算と、Green-Kubo 型解析に使う時間相関関数の蓄積を行うための HOOMD-blue コンポーネントです。

このパッケージには次の機能が含まれています。

- `heat_transfer.heatflux.ComputeheatFlux`: 全熱流束、運動項、ビリアル項、必要に応じて部分エンタルピー項をログ出力する HOOMD `Compute`
- `heat_transfer.correlate.Correlator`: HOOMD の `Logger` から数値列をサンプリングし、自己相関を出力する `Writer`
- 熱流束計算の CPU 実装と GPU 実装
- Lennard-Jones 流体および二成分混合系のサンプル

### 機能

`ComputeheatFlux` は以下の sequence quantity を提供します。

- `heatflux`: 全熱流束、`kinetic_heatflux + virial_heatflux - enthalpy_flux`
- `kinetic_heatflux`: エネルギー輸送の対流項
- `virial_heatflux`: ビリアル寄与
- `enthalpy_flux`: 部分エンタルピー流束。`include_enthalpy=True` で有効化

GPU デバイス上では、GPU 対応でビルドされている場合に `ComputeHeatFluxGPU` を使用します。`include_enthalpy=True` の場合も、CPU 実装と同じ定義で部分エンタルピー項を計算します。

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
    trigger=hoomd.trigger.Periodic(1),
    output_interval=1000,
    max_lag=1000,
)
simulation.operations.writers.append(correlator)
```

`Correlator` はカレントディレクトリに `correlation_<timestep>.dat` という名前で相関関数を書き出します。

### サンプル

各サンプルの目的は [sample/README.md](sample/README.md) を参照してください。

- `sample/N2`: 二粒子系による簡単な確認用ケース
- `sample/many`: 一成分 Lennard-Jones 流体の熱流束自己相関計算
- `sample/two_components`: 部分エンタルピー流束の参照計算を含む二成分 Lennard-Jones 混合系

### テスト

コンポーネントを HOOMD-blue 環境に install した後、以下を実行します。

```bash
python -m pytest -q /path/to/hoomd/site-packages/hoomd/heat_transfer/pytest
```

テストでは、import と version、ゼロ熱流束、一様速度、二粒子 Lennard-Jones 系、相関関数の蓄積を確認しています。

### 開発状況

このリポジトリは開発中です。熱流束と相関関数のワークフローを整備している段階のため、公開 API やサンプルファイルは今後変更される可能性があります。
