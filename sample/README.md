# Samples

[English](#english) / [日本語](#日本語)

## English

This directory contains small systems used to check heat-flux calculations and correlation workflows.

| Directory | Purpose |
| --- | --- |
| `N2` | Two-particle sanity-check system. It is useful for inspecting simple pair-interaction and velocity contributions by hand or against another MD code. |
| `many` | One-component Lennard-Jones fluid. This is the main sample for computing heat-flux autocorrelation in a homogeneous liquid. |
| `pure_ar_rnemd` | Small pure Ar Lennard-Jones RNEMD case for imposing a heat flux, measuring a temperature gradient, and estimating thermal conductivity. |
| `two_components` | Binary Lennard-Jones mixture with different particle masses. This case is useful for checking the partial-enthalpy contribution used with `include_enthalpy=True`. |

Generated files such as `.gsd`, `.dat`, and LAMMPS input/data files are included as convenient starting points. Regenerate them with the matching `init_cond.py` script when you want to change system size, density, composition, or initial velocities.

## 日本語

このディレクトリには、熱流束計算と相関関数計算の確認に使う小さな系を置いています。

| ディレクトリ | 目的 |
| --- | --- |
| `N2` | 二粒子の確認用ケースです。単純なペア相互作用と速度に由来する寄与を、手計算や他の MD コードと比較しやすい構成です。 |
| `many` | 一成分 Lennard-Jones 流体です。均一液体中の熱流束自己相関を計算する主なサンプルです。 |
| `pure_ar_rnemd` | pure Ar相当の小さな Lennard-Jones RNEMD ケースです。熱流束を与え、温度勾配を測り、熱伝導率を見積もります。 |
| `two_components` | 質量の異なる二成分 Lennard-Jones 混合系です。`include_enthalpy=True` で使う部分エンタルピー項の確認に使います。 |

`.gsd`、`.dat`、LAMMPS 入力ファイルなどの生成済みファイルは、すぐ試せる初期データとして含めています。粒子数、密度、組成、初期速度を変える場合は、各ケースの `init_cond.py` で再生成してください。
