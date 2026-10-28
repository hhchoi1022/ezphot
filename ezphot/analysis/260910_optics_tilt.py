#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
260910_optics_tilt.py

Instead of only the below-400 slope, change the optics slope GLOBALLY: take a smooth (Akima) curve of the same
shape through the 7 nodes and tilt the whole thing blue<->red by a multiplicative linear ramp
  factor(lambda) = 1 + g * (lambda - 600) / 300,   g in {-0.30, -0.15, 0, +0.15, +0.30}
(g>0 boosts red / suppresses blue).  Rebuild S = QE x optics x atm, run the same closed loop
(beam + lab delta_f, corrected XP) and compare BOTH the field colour term and the CALSPEC residual.

Reuses 260910_optics_smoothing.closed_loop / load_inputs (lab delta_f is applied inside).  Figure F16.

  MPLCONFIGDIR=/tmp/ezphot-matplotlib /home/hhchoi1022/miniconda3/envs/too/bin/python \\
      ezphot/analysis/260910_optics_tilt.py
"""
from __future__ import annotations

import importlib
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt     # noqa: E402
import numpy as np                  # noqa: E402
import pandas as pd                 # noqa: E402

warnings.filterwarnings("ignore")
sys.path.insert(0, "/lyman/data1/factory_hhchoi/code/ezphot/ezphot/analysis")
OS = importlib.import_module("260910_optics_smoothing")

SRC = OS.SRC
TS = OS.TS
GRID = OS.GRID


def tilt(base, grid, g):
    return np.clip(base * (1.0 + g * (grid - 600.0) / 300.0), 0.0, None)


def main():
    atm = pd.read_csv(TS / "atmosphere_elsauce.csv")
    grid = atm.lam_air.to_numpy(); sel = (grid >= 330) & (grid <= 1010)
    grid = grid[sel]; t_atm = atm.trans.to_numpy()[sel]
    qe = pd.read_csv(SRC / "QE.csv"); t_qe = np.interp(grid, qe.iloc[:, 0], qe.iloc[:, 1])
    st, cf, is_da, cal_spec, flux_cor = OS.load_inputs()

    base = OS.optics_on(grid, method="akima", blue_mult=1.0)          # smooth base of the same shape
    gs = {"-30% (blue up)": -0.30, "-15%": -0.15, "0 (base)": 0.0, "+15%": 0.15, "+30% (red up)": 0.30}
    optics = {k: tilt(base, grid, g) for k, g in gs.items()}
    variants = list(gs)
    res = OS.closed_loop(st, cf, is_da, cal_spec, flux_cor, grid, t_qe, t_atm, optics, variants)
    res.to_csv(TS / "optics_tilt.csv", index=False)

    piv = res.pivot_table(index="filter", columns="variant", values=["a_bprp", "da_other", "median_all"])
    print(piv.round(4).to_string())
    for metric in ["a_bprp", "da_other", "median_all"]:
        sub = res.dropna(subset=[metric])
        base_v = sub[sub.variant == "0 (base)"].set_index("filter")[metric]
        print(f"\n{metric}: median |.| by tilt  " +
              "  ".join(f"{k}={sub[sub.variant==k][metric].abs().median():.4f}" for k in variants))
        for k in ["-30% (blue up)", "+30% (red up)"]:
            d = (sub[sub.variant == k].set_index("filter")[metric] - base_v).dropna()
            print(f"   {k} - base: max|delta| {d.abs().max():.4f}, rms {np.sqrt((d**2).mean()):.4f}")

    col = plt.get_cmap("coolwarm")(np.linspace(0, 1, len(variants)))
    cmap = dict(zip(variants, col))
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    ax = axes[0]
    gg = np.linspace(340, 1010, 400)
    bb = OS.optics_on(gg, method="akima", blue_mult=1.0)
    for k, g in gs.items():
        ax.plot(gg, tilt(bb, gg, g), "-", color=cmap[k], lw=1.7, label=k)
    ax.plot(OS.OX, OS.OY, "ks", ms=8, label="data nodes")
    ax.set(xlim=(340, 1010), ylim=(0.35, 1.05), xlabel="wavelength [nm]", ylabel="optics total",
           title="(a) global optics tilt (smooth base, same shape)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    for ax, metric, ylab, ttl in [(axes[1], "a_bprp", "field colour term a(BP-RP)", "(b) field colour term"),
                                   (axes[2], "median_all", "CALSPEC median [mag]", "(c) CALSPEC median (corrected)")]:
        for k in variants:
            s = res[res.variant == k].dropna(subset=[metric]).sort_values("pivot")
            ax.plot(s["pivot"], s[metric], "o-", ms=4, color=cmap[k], label=k)
        ax.axhline(0, color="k", lw=0.5); ax.set(xlabel="pivot [nm]", ylabel=ylab, title=ttl)
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle("F16  global optics slope (blue<->red tilt, beam + lab delta_f, corrected XP): colour term + CALSPEC")
    fig.tight_layout(); fig.savefig(TS / "figures" / "F16_optics_tilt.png", dpi=150)
    print("\nsaved -> optics_tilt.csv, figures/F16_optics_tilt.png")


if __name__ == "__main__":
    main()
